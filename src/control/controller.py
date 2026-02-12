"""Main control loop for the Unitree G1 humanoid.

Handles policy inference, command building (PD control law), safety integration,
velocity commands, key handling, policy reloading, and telemetry.

Shared core between Metal and Docker plans — input dispatch differs.
"""
from __future__ import annotations

import glob
import os
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from src.config import (
    Config,
    G1_29DOF_JOINTS,
    G1_23DOF_JOINTS,
    Q_HOME_29DOF,
    Q_HOME_23DOF,
    STANDBY_KP_29DOF,
    STANDBY_KD_29DOF,
)
from src.control.safety import SafetyController, SystemState
from src.policy.base import PolicyInterface, detect_policy_format
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from src.robot.base import RobotCommand, RobotInterface, RobotState

if TYPE_CHECKING:
    from src.logging.logger import DataLogger


class Controller:
    """Main control loop: reads state, runs policy, sends commands.

    Args:
        robot: Robot backend (sim or real).
        policy: Neural network policy.
        safety: Safety state machine.
        joint_mapper: Joint ordering mapper.
        obs_builder: Observation builder (None for BeyondMimic, which builds its own).
        config: Full configuration.
        logger: Optional data logger.
        policy_dir: Optional directory of ONNX files for N/P key cycling.
        max_steps: Auto-terminate after this many steps (0 = disabled).
        max_duration: Auto-terminate after this many seconds (0 = disabled).
    """

    def __init__(
        self,
        robot: RobotInterface,
        policy: PolicyInterface,
        safety: SafetyController,
        joint_mapper: JointMapper,
        obs_builder: Optional[ObservationBuilder],
        config: Config,
        logger: Optional['DataLogger'] = None,
        policy_dir: Optional[str] = None,
        max_steps: int = 0,
        max_duration: float = 0.0,
    ):
        self.robot = robot
        self.policy = policy
        self.safety = safety
        self.joint_mapper = joint_mapper
        self.obs_builder = obs_builder
        self.config = config
        self._logger = logger

        # Control parameters
        self._dt = 1.0 / config.control.policy_frequency
        self._time_step = 0  # BeyondMimic trajectory index

        # Gains — expand scalars to per-joint arrays
        n_ctrl = joint_mapper.n_controlled
        self._kp = self._expand_gain(config.control.kp, n_ctrl)
        self._kd = self._expand_gain(config.control.kd, n_ctrl)
        self._ka = self._expand_gain(config.control.ka, n_ctrl)
        self._kd_damp = config.control.kd_damp

        # Home positions in controlled-joint order
        variant = config.robot.variant
        q_home_dict = config.control.q_home
        if q_home_dict is None:
            q_home_dict = Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
        self._q_home = np.array(
            [q_home_dict[j] for j in joint_mapper.controlled_joints]
        )

        # Velocity command (thread-safe via copy-on-write)
        self._velocity_command = np.zeros(3)
        self._vel_lock = threading.Lock()

        # Telemetry (thread-safe via dict replacement)
        self._telemetry: Dict = {
            "policy_hz": 0.0,
            "sim_hz": 0.0,
            "inference_ms": 0.0,
            "loop_ms": 0.0,
            "base_height": 0.0,
            "base_vel": np.zeros(3),
            "system_state": "idle",
            "step_count": 0,
        }
        self._telemetry_lock = threading.Lock()

        # Control thread
        self._running = False
        self._policy_active = False
        self._thread: Optional[threading.Thread] = None

        # Policy directory cycling
        self._policy_dir = policy_dir
        self._policy_files: List[str] = []
        self._policy_index: int = 0
        if policy_dir and os.path.isdir(policy_dir):
            self._policy_files = sorted(glob.glob(os.path.join(policy_dir, "*.onnx")))

        # Auto-termination
        self._max_steps = max_steps
        self._max_duration = max_duration

        # BeyondMimic end-of-trajectory handling
        self._completing_cycle = False  # extending ref past trajectory end
        self._cycle_period: int = 0
        self._cycle_target_step: int = 0  # step at which cycle completion ends
        self._interpolating = False
        self._interp_start_pos: Optional[np.ndarray] = None
        self._interp_step: int = 0
        self._interp_total_steps: int = int(2.0 * config.control.policy_frequency)  # 2 seconds
        self._interp_start_ref_q: Optional[np.ndarray] = None
        self._interp_start_ref_dq: Optional[np.ndarray] = None
        self._interp_end_ref_q: Optional[np.ndarray] = None
        self._interp_end_ref_dq: Optional[np.ndarray] = None


        # Stdout status
        self._last_print_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the control loop thread (idempotent).

        The loop runs continuously: in IDLE/STOPPED it holds the home pose;
        in RUNNING it executes the policy.  Policy state is reset on the
        IDLE → RUNNING transition inside the loop.
        """
        if self._running:
            return
        self._running = True
        self._time_step = 0
        self._completing_cycle = False
        self._interpolating = False
        self._policy_active = False
        self.policy.reset()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the control loop and wait for the thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        """Set velocity command (thread-safe)."""
        with self._vel_lock:
            self._velocity_command = np.array([vx, vy, yaw_rate])

    def get_velocity_command(self) -> np.ndarray:
        """Get current velocity command (thread-safe). Returns [vx, vy, yaw_rate]."""
        with self._vel_lock:
            return self._velocity_command.copy()

    def get_telemetry(self) -> dict:
        """Get latest telemetry (thread-safe, non-blocking)."""
        with self._telemetry_lock:
            return dict(self._telemetry)

    def reload_policy(self, policy_path: str) -> None:
        """Load a new ONNX policy. Stops control loop if running.

        Thread-safe. After reload, the policy and internal state are reset.

        Raises:
            ValueError: If the path is invalid or model cannot be loaded.
        """
        was_running = self._running
        if was_running:
            self.safety.stop()
            self.stop()

        # Attempt load — if it fails, original policy is preserved
        self.policy.load(policy_path)
        self.policy.reset()
        self._time_step = 0
        self._completing_cycle = False
        self._interpolating = False

    def handle_key(self, key: str) -> None:
        """Handle keyboard input (from MuJoCo viewer or CLI).

        Key map (avoids MuJoCo viewer letter-key conflicts):
            Space      — start / stop
            Backspace  — e-stop
            Enter      — clear e-stop
            Delete     — reset (Fn+Backspace on Mac)
            Up/Down    — vx +/-
            Left/Right — vy +/-
            ,/.        — yaw +/-
            /          — zero velocity
            -/=        — prev/next policy

        Called from the main thread. Thread-safe.
        """
        if key == "space":
            if self.safety.state in (SystemState.IDLE, SystemState.STOPPED):
                self.safety.start()
                # Control loop is already running (hold pose mode);
                # it will detect the RUNNING transition and activate policy.
                if not self._running:
                    self.start()
            elif self.safety.state == SystemState.RUNNING:
                self.safety.stop()
        elif key == "backspace":
            self.safety.estop()
        elif key == "enter":
            self.safety.clear_estop()
        elif key == "delete":
            self.robot.reset()
        elif key == "up":
            self._adjust_velocity(0, 0.1, -1.0, 1.0)
        elif key == "down":
            self._adjust_velocity(0, -0.1, -1.0, 1.0)
        elif key == "left":
            self._adjust_velocity(1, 0.1, -0.5, 0.5)
        elif key == "right":
            self._adjust_velocity(1, -0.1, -0.5, 0.5)
        elif key == "comma":
            self._adjust_velocity(2, 0.1, -1.0, 1.0)
        elif key == "period":
            self._adjust_velocity(2, -0.1, -1.0, 1.0)
        elif key == "slash":
            with self._vel_lock:
                self._velocity_command = np.zeros(3)
        elif key == "equal":
            self._cycle_policy(1)
        elif key == "minus":
            self._cycle_policy(-1)

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(self, state: RobotState, action: np.ndarray) -> RobotCommand:
        """Build a RobotCommand from policy action output.

        Both IsaacLab and BeyondMimic use the same control law from training:
            tau = Kp * (default_q + Ka * action - q) - Kd * dq

        The ONNX ``joint_pos`` / ``joint_vel`` outputs are reference trajectory
        data from constant-table lookups (observation-independent).  They feed
        the ``command`` and ``motion_anchor_*`` observation terms but are NOT
        used as PD targets — the learned actor network output (``action``) is
        what drives the robot through the training control law.
        """
        n_total = self.joint_mapper.n_total
        n_ctrl = self.joint_mapper.n_controlled

        target_pos = np.zeros(n_total)
        target_vel = np.zeros(n_total)
        target_tau = np.zeros(n_total)
        kp = np.zeros(n_total)
        kd = np.zeros(n_total)

        ctrl_idx = self.joint_mapper.controlled_indices
        non_ctrl_idx = self.joint_mapper.non_controlled_indices

        # Determine if BeyondMimic
        from src.policy.beyondmimic_policy import BeyondMimicPolicy
        is_beyondmimic = isinstance(self.policy, BeyondMimicPolicy)

        if is_beyondmimic:
            bm: BeyondMimicPolicy = self.policy
            # Use metadata gains if available, else config
            ctrl_kp = bm.stiffness if bm.stiffness is not None else self._kp
            ctrl_kd = bm.damping if bm.damping is not None else self._kd
            ctrl_ka = bm.action_scale if bm.action_scale is not None else self._ka

            # Training control law: target = default_q + Ka * action
            ctrl_target_pos = bm.default_joint_pos + ctrl_ka * action
            ctrl_target_vel = np.zeros(n_ctrl)
        else:
            ctrl_kp = self._kp
            ctrl_kd = self._kd
            ctrl_ka = self._ka

            ctrl_target_pos = self._q_home + ctrl_ka * action
            ctrl_target_vel = np.zeros(n_ctrl)

        target_pos[ctrl_idx] = ctrl_target_pos
        target_vel[ctrl_idx] = ctrl_target_vel
        kp[ctrl_idx] = ctrl_kp
        kd[ctrl_idx] = ctrl_kd

        # Non-controlled joints: damping mode
        if len(non_ctrl_idx) > 0:
            target_pos[non_ctrl_idx] = state.joint_positions[non_ctrl_idx]
            kd[non_ctrl_idx] = self._kd_damp

        return RobotCommand(
            joint_positions=target_pos,
            joint_velocities=target_vel,
            joint_torques=target_tau,
            kp=kp,
            kd=kd,
        )

    def _build_interpolation_command(self, state: RobotState) -> RobotCommand:
        """Build command for BeyondMimic end-of-trajectory interpolation to q_home."""
        alpha = self._interp_step / self._interp_total_steps
        n_total = self.joint_mapper.n_total
        ctrl_idx = self.joint_mapper.controlled_indices
        non_ctrl_idx = self.joint_mapper.non_controlled_indices

        target_pos = np.zeros(n_total)
        target_vel = np.zeros(n_total)
        target_tau = np.zeros(n_total)
        kp = np.zeros(n_total)
        kd = np.zeros(n_total)

        # Linear interpolation: start_pos -> q_home
        interp_pos = (1.0 - alpha) * self._interp_start_pos + alpha * self._q_home
        target_pos[ctrl_idx] = interp_pos
        kp[ctrl_idx] = self._kp
        kd[ctrl_idx] = self._kd

        if len(non_ctrl_idx) > 0:
            target_pos[non_ctrl_idx] = state.joint_positions[non_ctrl_idx]
            kd[non_ctrl_idx] = self._kd_damp

        return RobotCommand(
            joint_positions=target_pos,
            joint_velocities=target_vel,
            joint_torques=target_tau,
            kp=kp,
            kd=kd,
        )

    def _build_hold_command(self, state: RobotState) -> RobotCommand:
        """Build command to hold the robot at home pose (IDLE/STOPPED state).

        Uses high standby PD gains (Kp=150-350 legs, 40 arms) from the
        BeyondMimic deployment reference.  These are much higher than the
        walking policy gains and sufficient to hold the robot upright in
        MuJoCo without gravity compensation.

        Torques are applied via ``qfrc_applied`` (see ``SimRobot.send_command``)
        which bypasses actuator ctrlrange limits in the XML.
        """
        n_total = self.joint_mapper.n_total
        ctrl_idx = self.joint_mapper.controlled_indices
        non_ctrl_idx = self.joint_mapper.non_controlled_indices

        target_pos = np.zeros(n_total)
        target_vel = np.zeros(n_total)
        target_tau = np.zeros(n_total)
        kp = np.zeros(n_total)
        kd = np.zeros(n_total)

        # Determine home pose (BeyondMimic default_joint_pos or config q_home)
        from src.policy.beyondmimic_policy import BeyondMimicPolicy
        if isinstance(self.policy, BeyondMimicPolicy):
            bm: BeyondMimicPolicy = self.policy
            hold_pos = bm.default_joint_pos
        else:
            hold_pos = self._q_home

        target_pos[ctrl_idx] = hold_pos

        # Use high standby gains from BeyondMimic deployment reference.
        # These are keyed by config joint names (controlled_joints order).
        ctrl_joints = self.joint_mapper.controlled_joints
        standby_kp = np.array(
            [STANDBY_KP_29DOF.get(j, 100.0) for j in ctrl_joints],
            dtype=np.float64,
        )
        standby_kd = np.array(
            [STANDBY_KD_29DOF.get(j, 5.0) for j in ctrl_joints],
            dtype=np.float64,
        )
        kp[ctrl_idx] = standby_kp
        kd[ctrl_idx] = standby_kd

        # Non-controlled joints: damping mode
        if len(non_ctrl_idx) > 0:
            target_pos[non_ctrl_idx] = state.joint_positions[non_ctrl_idx]
            kd[non_ctrl_idx] = self._kd_damp

        return RobotCommand(
            joint_positions=target_pos,
            joint_velocities=target_vel,
            joint_torques=target_tau,
            kp=kp,
            kd=kd,
        )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """Main control loop — runs in a background thread."""
        last_action = np.zeros(self.joint_mapper.n_controlled)
        step_count = 0
        loop_start_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            try:
                # 1. E-stop: send damping and keep sim alive
                if self.safety.state == SystemState.ESTOP:
                    state = self.robot.get_state()
                    cmd = self.safety.get_damping_command(state)
                    self.robot.send_command(cmd)
                    self.robot.step()
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 2. IDLE/STOPPED: hold the robot at home pose.
                #    This keeps the robot standing under gravity while
                #    the user hasn't yet pressed Space to start the policy.
                if self.safety.state != SystemState.RUNNING:
                    self._policy_active = False
                    state = self.robot.get_state()

                    # BeyondMimic: use the policy frozen at time_step=0
                    # for active balance.  Simple PD hold can't stabilize
                    # the undamped BM training model because the ankle
                    # torque limit saturates before the pitching moment
                    # is arrested.  The trained actor network adjusts
                    # actions based on observations to keep the base upright.
                    from src.policy.beyondmimic_policy import BeyondMimicPolicy
                    if isinstance(self.policy, BeyondMimicPolicy):
                        bm: BeyondMimicPolicy = self.policy
                        if hasattr(self.robot, 'get_body_state') and bm.anchor_body_name:
                            anchor_pos, anchor_quat = self.robot.get_body_state(
                                bm.anchor_body_name
                            )
                        else:
                            anchor_pos = state.base_position
                            anchor_quat = state.imu_quaternion
                        obs = bm.build_observation(state, anchor_pos, anchor_quat)
                        action = bm.get_action(obs, time_step=0)
                        cmd = self._build_command(state, action)
                    else:
                        cmd = self._build_hold_command(state)

                    self.robot.send_command(cmd)
                    self.robot.step()
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 3. Detect IDLE/STOPPED → RUNNING transition
                if not self._policy_active:
                    self._policy_active = True
                    self._time_step = 0
                    self._completing_cycle = False
                    self._interpolating = False
                    self.policy.reset()
                    last_action = np.zeros(self.joint_mapper.n_controlled)
                    step_count = 0
                    loop_start_time = time.perf_counter()
                    self._last_print_time = 0.0

                    from src.policy.beyondmimic_policy import BeyondMimicPolicy
                    if isinstance(self.policy, BeyondMimicPolicy):
                        bm: BeyondMimicPolicy = self.policy
                        # Prefetch reference data for time_step=0 so the
                        # observation builder has _prev_target_q/dq.
                        # No teleport — hold mode already used the policy
                        # at time_step=0, so the robot is already near the
                        # reference starting pose.
                        bm.prefetch_reference(0)
                        print("[controller] Policy activated — starting from hold pose.")
                    else:
                        print("[controller] Policy activated.")

                # 4. Get robot state
                state = self.robot.get_state()

                # 4a. Gait cycle completion: extend the reference past
                #     the end of the trajectory using the periodic gait
                #     pattern.  Runs the policy at (time_step - period) to
                #     get the equivalent reference from one cycle back.
                if self._completing_cycle:
                    from src.policy.beyondmimic_policy import BeyondMimicPolicy
                    bm: BeyondMimicPolicy = self.policy
                    ref_ts = self._time_step - self._cycle_period
                    if hasattr(self.robot, 'get_body_state') and bm.anchor_body_name:
                        anchor_pos, anchor_quat = self.robot.get_body_state(
                            bm.anchor_body_name
                        )
                    else:
                        anchor_pos = state.base_position
                        anchor_quat = state.imu_quaternion
                    obs = bm.build_observation(state, anchor_pos, anchor_quat)
                    action = bm.get_action(obs, time_step=ref_ts)
                    cmd = self._build_command(state, action)
                    cmd = self.safety.clamp_command(cmd)
                    self.robot.send_command(cmd)
                    self.robot.step()
                    self._time_step += 1
                    if self._time_step >= self._cycle_target_step:
                        self._completing_cycle = False
                        # Cache current reference as interpolation start
                        self._interp_start_ref_q = bm._prev_target_q.copy()
                        self._interp_start_ref_dq = bm._prev_target_dq.copy()
                        # Fetch frame-0 as interpolation target
                        bm.prefetch_reference(0)
                        self._interp_end_ref_q = bm._prev_target_q.copy()
                        self._interp_end_ref_dq = bm._prev_target_dq.copy()
                        self._interpolating = True
                        self._interp_step = 0
                        print(
                            "[controller] Gait cycle complete. "
                            "Interpolating to stance."
                        )
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 4b. Smooth interpolation from double-support to stance.
                #     BeyondMimic: interpolate the command reference from the
                #     cycle-completion pose to frame 0 while the policy at
                #     time_step=0 provides active balance.
                #     Non-BM: PD interpolation from last position to q_home.
                if self._interpolating:
                    from src.policy.beyondmimic_policy import BeyondMimicPolicy
                    if isinstance(self.policy, BeyondMimicPolicy):
                        bm: BeyondMimicPolicy = self.policy
                        alpha = self._interp_step / self._interp_total_steps

                        # Interpolate command reference for observation
                        bm._prev_target_q = (
                            (1.0 - alpha) * self._interp_start_ref_q
                            + alpha * self._interp_end_ref_q
                        )
                        bm._prev_target_dq = (
                            (1.0 - alpha) * self._interp_start_ref_dq
                            + alpha * self._interp_end_ref_dq
                        )

                        if hasattr(self.robot, 'get_body_state') and bm.anchor_body_name:
                            anchor_pos, anchor_quat = self.robot.get_body_state(
                                bm.anchor_body_name
                            )
                        else:
                            anchor_pos = state.base_position
                            anchor_quat = state.imu_quaternion
                        obs = bm.build_observation(state, anchor_pos, anchor_quat)
                        action = bm.get_action(obs, time_step=0)
                        cmd = self._build_command(state, action)
                    else:
                        cmd = self._build_interpolation_command(state)
                    cmd = self.safety.clamp_command(cmd)
                    self.robot.send_command(cmd)
                    self.robot.step()
                    self._interp_step += 1
                    if self._interp_step >= self._interp_total_steps:
                        self._interpolating = False
                        print("[controller] Returned to stance. Switching to hold mode.")
                        self.safety.stop()
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 5. Build observation and run policy inference
                inference_start = time.perf_counter()
                if self.obs_builder is not None:
                    vel_cmd = self.get_velocity_command()
                    obs = self.obs_builder.build(state, last_action, vel_cmd)
                    action = self.policy.get_action(obs)
                else:
                    # BeyondMimic: single ONNX call per step matching the
                    # motion_tracking_controller C++ sim2sim deployment.
                    #
                    # 1. Build obs using PREVIOUS step's cached trajectory
                    #    (_prev_target_q/dq for "command" term, etc.)
                    # 2. Run ONNX with current time_step → gets action +
                    #    new trajectory data (target_q/dq, body poses)
                    # 3. Post-increment time_step (matching C++ timeStep_++)
                    from src.policy.beyondmimic_policy import BeyondMimicPolicy
                    bm: BeyondMimicPolicy = self.policy

                    # Get anchor body (e.g. torso_link) state from sim
                    if hasattr(self.robot, 'get_body_state') and bm.anchor_body_name:
                        anchor_pos, anchor_quat = self.robot.get_body_state(
                            bm.anchor_body_name
                        )
                    else:
                        anchor_pos = state.base_position
                        anchor_quat = state.imu_quaternion
                    obs = bm.build_observation(state, anchor_pos, anchor_quat)
                    action = bm.get_action(obs, time_step=self._time_step)
                    self._time_step += 1

                    # Detect end of reference trajectory
                    if self._time_step >= bm.trajectory_length:
                        period, extra = self._compute_cycle_extension(bm)
                        self._cycle_period = period
                        self._cycle_target_step = self._time_step + extra
                        self._completing_cycle = True
                        print(
                            f"[controller] End of trajectory at step "
                            f"{self._time_step}. Completing gait cycle: "
                            f"{extra} extra steps (period={period}) to "
                            f"step {self._cycle_target_step}."
                        )
                inference_time = time.perf_counter() - inference_start

                # Safety: replace NaN/Inf in policy output with zeros
                if not np.all(np.isfinite(action)):
                    print("[controller] WARNING: NaN/Inf in policy action, using zeros")
                    action = np.zeros_like(action)

                # 6. Build and clamp command
                cmd = self._build_command(state, action)
                cmd = self.safety.clamp_command(cmd)

                # 7. Send command
                self.robot.send_command(cmd)

                # 8. Step simulation
                self.robot.step()

                # 9. Store for next iteration
                last_action = action.copy()
                step_count += 1

                # 10. Log
                if self._logger is not None:
                    loop_time = time.perf_counter() - loop_start
                    self._logger.log_step(
                        timestamp=time.time(),
                        robot_state=state,
                        observation=obs,
                        action=action,
                        command=cmd,
                        system_state=self.safety.state,
                        velocity_command=self.get_velocity_command(),
                        timing={
                            "inference_ms": inference_time * 1000.0,
                            "loop_ms": loop_time * 1000.0,
                        },
                    )

                # 11. Update telemetry
                self._update_telemetry(state, inference_time, loop_start, step_count)

                # 12. Print status at 1 Hz
                now = time.perf_counter()
                if now - self._last_print_time >= 1.0:
                    vel_cmd = self.get_velocity_command()
                    print(
                        f"[controller] state={self.safety.state.value} "
                        f"step={step_count} "
                        f"policy_hz={1.0 / max(now - loop_start, 1e-6):.1f} "
                        f"vel_cmd=[{vel_cmd[0]:.1f}, {vel_cmd[1]:.1f}, {vel_cmd[2]:.1f}]"
                    )
                    self._last_print_time = now

                # 13. Check auto-termination
                if self._max_steps > 0 and step_count >= self._max_steps:
                    self.safety.stop()
                    self._running = False
                    break
                if self._max_duration > 0 and (now - loop_start_time) >= self._max_duration:
                    self.safety.stop()
                    self._running = False
                    break

            except Exception as exc:
                # Any exception in the control loop triggers E-stop
                print(f"[controller] EXCEPTION in control loop: {exc}")
                self.safety.estop()

            self._sleep_until_next_tick(loop_start)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sleep_until_next_tick(self, loop_start: float) -> None:
        """Sleep to maintain policy frequency."""
        elapsed = time.perf_counter() - loop_start
        remaining = self._dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _update_telemetry(
        self, state: RobotState, inference_time: float, loop_start: float, step_count: int
    ) -> None:
        """Update telemetry dict (thread-safe via replacement)."""
        loop_time = time.perf_counter() - loop_start
        telem = {
            "policy_hz": 1.0 / max(loop_time, 1e-6),
            "sim_hz": self.config.control.sim_frequency,
            "inference_ms": inference_time * 1000.0,
            "loop_ms": loop_time * 1000.0,
            "base_height": state.base_position[2],
            "base_vel": state.base_velocity.copy(),
            "system_state": self.safety.state.value,
            "step_count": step_count,
        }
        with self._telemetry_lock:
            self._telemetry = telem

    def _adjust_velocity(self, index: int, delta: float, vmin: float, vmax: float) -> None:
        """Adjust one component of velocity command, clamped to [vmin, vmax]."""
        with self._vel_lock:
            vc = self._velocity_command.copy()
            vc[index] = np.clip(vc[index] + delta, vmin, vmax)
            self._velocity_command = vc

    def _cycle_policy(self, direction: int) -> None:
        """Cycle to next/previous policy in policy_dir."""
        if not self._policy_files:
            return
        self._policy_index = (self._policy_index + direction) % len(self._policy_files)
        path = self._policy_files[self._policy_index]
        try:
            self.reload_policy(path)
            print(f"[controller] Loaded policy: {os.path.basename(path)}")
        except Exception as exc:
            print(f"[controller] Failed to load policy: {exc}")

    def _compute_cycle_extension(self, bm) -> tuple:
        """Compute gait cycle period and extra steps to reach best transition.

        Analyzes the reference trajectory to detect the steady-state gait
        period and find the double-support phase with minimum reference
        joint velocity — the most stable point where both feet are on
        the ground and the base momentum is lowest.

        Returns:
            (period, extra_steps): Gait period and steps to extend.
        """
        traj_len = bm.trajectory_length
        obs_dummy = np.zeros((1, bm.observation_dim), dtype=np.float32)
        tq_name = bm._output_names[bm._target_q_idx]
        tdq_name = bm._output_names[bm._target_dq_idx]

        def get_ref_q(ts: int) -> np.ndarray:
            return bm._session.run(
                [tq_name],
                {"obs": obs_dummy, "time_step": np.array([[ts]], dtype=np.float32)},
            )[0].flatten()

        def get_ref_dq(ts: int) -> np.ndarray:
            return bm._session.run(
                [tdq_name],
                {"obs": obs_dummy, "time_step": np.array([[ts]], dtype=np.float32)},
            )[0].flatten()

        # Detect gait period from auto-correlation of the first joint
        # (left hip pitch) in the last 150 steps.
        n_analyze = min(150, traj_len)
        start_ts = traj_len - n_analyze
        hip_vals = np.array([get_ref_q(ts)[0] for ts in range(start_ts, traj_len)])

        best_period, best_corr = 20, 1e9
        for lag in range(20, 80):
            if lag >= len(hip_vals):
                break
            diff = hip_vals[lag:] - hip_vals[:-lag]
            corr = np.mean(diff ** 2)
            if corr < best_corr:
                best_corr = corr
                best_period = lag

        period = best_period

        # Find the double-support phase with minimum reference joint velocity.
        # This is where both feet are stably on the ground — the safest
        # point to begin the transition back to stance.
        cycle_start = traj_len - period
        best_phase, best_vel = 0, np.inf
        for phase in range(period):
            ts = cycle_start + phase
            if ts >= traj_len:
                break
            vel = np.linalg.norm(get_ref_dq(ts))
            if vel < best_vel:
                best_vel = vel
                best_phase = phase

        # Compute how many extra steps from current time_step to best phase.
        current_phase = (self._time_step - 1 - cycle_start) % period
        extra = (best_phase - current_phase) % period
        if extra < 3:
            extra += period

        return period, extra

    @staticmethod
    def _expand_gain(value, n: int) -> np.ndarray:
        """Expand a scalar or list gain to an (n,) array."""
        if isinstance(value, (int, float)):
            return np.full(n, value, dtype=np.float64)
        return np.array(value, dtype=np.float64)

    @property
    def is_running(self) -> bool:
        return self._running
