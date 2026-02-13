"""Main control loop for the Unitree G1 humanoid.

Handles policy inference, command building (PD control law), safety integration,
velocity commands, key handling, policy reloading, and telemetry.

Supports dual-policy mode: a default policy (IsaacLab velocity tracking) for
stance/idle and an active policy (BM, IL, etc.) for execution.

Shared core between Metal and Docker plans -- input dispatch differs.
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
    ISAACLAB_KD_29DOF,
    ISAACLAB_KP_29DOF,
    Q_HOME_29DOF,
    Q_HOME_23DOF,
    STANDBY_KP_29DOF,
    STANDBY_KD_29DOF,
)
from src.control.safety import ControlMode, SafetyController, SystemState
from src.policy.base import PolicyInterface
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from src.robot.base import RobotCommand, RobotInterface, RobotState

if TYPE_CHECKING:
    from src.logging.logger import DataLogger


class Controller:
    """Main control loop: reads state, runs policy, sends commands.

    Supports dual-policy mode:

    - **Default policy** (IsaacLab velocity tracking): runs in IDLE/STOPPED
      states with zero velocity command for balanced stance.
    - **Active policy** (``--policy``): runs in RUNNING state (BM, IL, etc.).

    If no default policy is provided, IDLE/STOPPED uses static PD hold
    at home pose (standby gains).

    Args:
        robot: Robot backend (sim or real).
        policy: Active neural network policy (the ``--policy``).
        safety: Safety state machine.
        joint_mapper: Joint ordering mapper for the active policy.
        obs_builder: Observation builder for the active policy
            (``None`` for BeyondMimic, which builds its own).
        config: Full configuration.
        logger: Optional data logger.
        policy_dir: Optional directory of ONNX files for ``-/=`` key cycling.
        max_steps: Auto-terminate after this many RUNNING steps (0 = disabled).
        max_duration: Auto-terminate after this many seconds (0 = disabled).
        default_policy: Optional IsaacLab policy for stance/velocity tracking.
        default_obs_builder: Observation builder for the default policy.
        default_joint_mapper: Joint mapper for the default policy.
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
        default_policy: Optional[PolicyInterface] = None,
        default_obs_builder: Optional[ObservationBuilder] = None,
        default_joint_mapper: Optional[JointMapper] = None,
    ):
        self.robot = robot
        self.policy = policy  # Active policy
        self.safety = safety
        self.joint_mapper = joint_mapper  # Active policy mapper
        self.obs_builder = obs_builder  # Active policy obs builder
        self.config = config
        self._logger = logger

        # Default policy for stance (optional)
        self._default_policy = default_policy
        self._default_obs_builder = default_obs_builder
        self._default_mapper = default_joint_mapper

        # Control mode: DEFAULT if default policy is available, else HOLD
        self._control_mode = (
            ControlMode.DEFAULT if default_policy else ControlMode.HOLD
        )

        # Control parameters
        self._dt = 1.0 / config.control.policy_frequency
        self._time_step = 0  # BeyondMimic trajectory index

        # Active policy gains -- expand scalars to per-joint arrays
        n_ctrl = joint_mapper.n_controlled
        self._kp = self._expand_gain(config.control.kp, n_ctrl)
        self._kd = self._expand_gain(config.control.kd, n_ctrl)
        self._ka = self._expand_gain(config.control.ka, n_ctrl)
        self._kd_damp = config.control.kd_damp

        # Active policy home positions in controlled-joint order
        variant = config.robot.variant
        q_home_dict = config.control.q_home
        if q_home_dict is None:
            q_home_dict = Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
        self._q_home = np.array(
            [q_home_dict[j] for j in joint_mapper.controlled_joints]
        )

        # Default policy gains and home (if available)
        if default_joint_mapper is not None:
            n_def = default_joint_mapper.n_controlled
            def_joints = default_joint_mapper.controlled_joints
            self._default_kp = np.array(
                [ISAACLAB_KP_29DOF.get(j, 40.0) for j in def_joints]
            )
            self._default_kd = np.array(
                [ISAACLAB_KD_29DOF.get(j, 2.5) for j in def_joints]
            )
            self._default_ka = self._expand_gain(config.policy.default_ka, n_def)
            self._default_q_home = np.array(
                [q_home_dict[j] for j in def_joints]
            )
            self._default_last_action = np.zeros(n_def)
        else:
            self._default_last_action = np.zeros(0)

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
            self._policy_files = sorted(
                glob.glob(os.path.join(policy_dir, "*.onnx"))
            )

        # Auto-termination
        self._max_steps = max_steps
        self._max_duration = max_duration

        # Stdout status
        self._last_print_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the control loop thread (idempotent).

        The loop runs continuously: in IDLE/STOPPED it runs the default
        policy (stance) or static hold; in RUNNING it executes the active
        policy.  Policy state is reset on the IDLE->RUNNING transition.
        """
        if self._running:
            return
        self._running = True
        self._time_step = 0
        self._policy_active = False
        self._control_mode = (
            ControlMode.DEFAULT if self._default_policy else ControlMode.HOLD
        )
        self.policy.reset()
        if self._default_policy is not None:
            self._default_policy.reset()
            self._default_last_action = np.zeros(
                self._default_mapper.n_controlled
            )
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the control loop and wait for the thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_velocity_command(
        self, vx: float, vy: float, yaw_rate: float
    ) -> None:
        """Set velocity command (thread-safe)."""
        with self._vel_lock:
            self._velocity_command = np.array([vx, vy, yaw_rate])

    def get_velocity_command(self) -> np.ndarray:
        """Get current velocity command (thread-safe).

        Returns:
            ``[vx, vy, yaw_rate]``
        """
        with self._vel_lock:
            return self._velocity_command.copy()

    def get_telemetry(self) -> dict:
        """Get latest telemetry (thread-safe, non-blocking)."""
        with self._telemetry_lock:
            return dict(self._telemetry)

    def reload_policy(self, policy_path: str) -> None:
        """Load a new ONNX active policy. Stops control loop if running.

        Thread-safe. After reload, the policy and internal state are reset.

        Raises:
            ValueError: If the path is invalid or model cannot be loaded.
        """
        was_running = self._running
        if was_running:
            self.safety.stop()
            self.stop()

        # Attempt load -- if it fails, original policy is preserved
        self.policy.load(policy_path)
        self.policy.reset()
        self._time_step = 0

    def handle_key(self, key: str) -> None:
        """Handle keyboard input (from MuJoCo viewer or CLI).

        Key map (avoids MuJoCo viewer letter-key conflicts):
            Space      -- start / stop
            Backspace  -- e-stop
            Enter      -- clear e-stop
            Delete     -- reset (Fn+Backspace on Mac)
            Up/Down    -- vx +/-
            Left/Right -- vy +/-
            ,/.        -- yaw +/-
            /          -- zero velocity
            -/=        -- prev/next policy

        Called from the main thread. Thread-safe.
        """
        if key == "space":
            if self.safety.state in (SystemState.IDLE, SystemState.STOPPED):
                self.safety.start()
                # Control loop is already running (default/hold mode);
                # it will detect RUNNING and activate the active policy.
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

    def _build_command(
        self, state: RobotState, action: np.ndarray
    ) -> RobotCommand:
        """Build a RobotCommand from active policy action output.

        Both IsaacLab and BeyondMimic use the same control law from training::

            tau = Kp * (default_q + Ka * action - q) - Kd * dq

        The ONNX ``joint_pos`` / ``joint_vel`` outputs are reference trajectory
        data from constant-table lookups (observation-independent).  They feed
        the ``command`` and ``motion_anchor_*`` observation terms but are NOT
        used as PD targets -- the learned actor network output (``action``) is
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
            ctrl_ka = (
                bm.action_scale if bm.action_scale is not None else self._ka
            )

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

    def _build_default_command(
        self, state: RobotState, action: np.ndarray
    ) -> RobotCommand:
        """Build a RobotCommand from default policy action output.

        Uses IsaacLab training gains (ISAACLAB_KP/KD_29DOF) and
        config.policy.default_ka, mapped through the default policy's
        joint mapper.
        """
        mapper = self._default_mapper
        n_total = mapper.n_total

        target_pos = np.zeros(n_total)
        target_vel = np.zeros(n_total)
        target_tau = np.zeros(n_total)
        kp = np.zeros(n_total)
        kd = np.zeros(n_total)

        ctrl_idx = mapper.controlled_indices
        non_ctrl_idx = mapper.non_controlled_indices

        ctrl_target_pos = self._default_q_home + self._default_ka * action
        target_pos[ctrl_idx] = ctrl_target_pos
        kp[ctrl_idx] = self._default_kp
        kd[ctrl_idx] = self._default_kd

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
        """Build command to hold the robot at home pose (static PD).

        Uses high standby PD gains (Kp=150-350 legs, 40 arms) from the
        BeyondMimic deployment reference.  Fallback when no default policy
        is loaded.

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

        target_pos[ctrl_idx] = self._q_home

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
        """Main control loop -- runs in a background thread.

        State machine:

        - **ESTOP**: damping mode (kp=0, kd=kd_damp)
        - **IDLE/STOPPED + DEFAULT**: default policy with zero velocity (stance)
        - **IDLE/STOPPED + HOLD**: static PD at home pose (standby gains)
        - **RUNNING + ACTIVE_POLICY**: active policy executes
            - BM: when trajectory ends, auto-return to DEFAULT + STOPPED
            - IL: runs until Space / max_steps / max_duration
        """
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

                # 2. IDLE/STOPPED: run default policy (stance) or static hold
                if self.safety.state != SystemState.RUNNING:
                    self._policy_active = False
                    self._control_mode = (
                        ControlMode.DEFAULT
                        if self._default_policy
                        else ControlMode.HOLD
                    )
                    state = self.robot.get_state()

                    if self._control_mode == ControlMode.DEFAULT:
                        # Default policy: IL velocity tracking with zero vel
                        vel_cmd = np.zeros(3)
                        obs = self._default_obs_builder.build(
                            state, self._default_last_action, vel_cmd
                        )
                        action = self._default_policy.get_action(obs)
                        self._default_last_action = action.copy()
                        cmd = self._build_default_command(state, action)
                    else:
                        # Static PD hold at home pose
                        cmd = self._build_hold_command(state)

                    self.robot.send_command(cmd)
                    self.robot.step()
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 3. Detect IDLE/STOPPED -> RUNNING transition
                if not self._policy_active:
                    self._policy_active = True
                    self._control_mode = ControlMode.ACTIVE_POLICY
                    self._time_step = 0
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
                        # No teleport -- default policy already maintained
                        # stance near the starting pose.
                        bm.prefetch_reference(0)
                        print(
                            "[controller] Active policy started (BeyondMimic)."
                        )
                    else:
                        print("[controller] Active policy started.")

                # 4. Get robot state
                state = self.robot.get_state()

                # 5. Build observation and run policy inference
                inference_start = time.perf_counter()
                if self.obs_builder is not None:
                    # IsaacLab active policy
                    vel_cmd = self.get_velocity_command()
                    obs = self.obs_builder.build(state, last_action, vel_cmd)
                    action = self.policy.get_action(obs)
                else:
                    # BeyondMimic active policy: single ONNX call per step
                    # matching motion_tracking_controller C++ sim2sim.
                    #
                    # 1. Build obs using PREVIOUS step's cached trajectory
                    # 2. Run ONNX with current time_step -> action + ref data
                    # 3. Post-increment time_step (matching C++ timeStep_++)
                    from src.policy.beyondmimic_policy import BeyondMimicPolicy

                    bm = self.policy
                    if hasattr(self.robot, "get_body_state") and bm.anchor_body_name:
                        anchor_pos, anchor_quat = self.robot.get_body_state(
                            bm.anchor_body_name
                        )
                    else:
                        anchor_pos = state.base_position
                        anchor_quat = state.imu_quaternion
                    obs = bm.build_observation(
                        state, anchor_pos, anchor_quat
                    )
                    action = bm.get_action(
                        obs, time_step=self._time_step
                    )
                    self._time_step += 1

                    # Detect end of reference trajectory
                    if self._time_step >= bm.trajectory_length:
                        print(
                            f"[controller] BeyondMimic trajectory ended at "
                            f"step {self._time_step}. Returning to "
                            f"{'default policy' if self._default_policy else 'hold'}."
                        )
                        self._control_mode = (
                            ControlMode.DEFAULT
                            if self._default_policy
                            else ControlMode.HOLD
                        )
                        self.safety.stop()
                        # Still apply this last action below

                inference_time = time.perf_counter() - inference_start

                # Safety: replace NaN/Inf in policy output with zeros
                if not np.all(np.isfinite(action)):
                    print(
                        "[controller] WARNING: NaN/Inf in policy action, "
                        "using zeros"
                    )
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
                self._update_telemetry(
                    state, inference_time, loop_start, step_count
                )

                # 12. Print status at 1 Hz
                now = time.perf_counter()
                if now - self._last_print_time >= 1.0:
                    vel_cmd = self.get_velocity_command()
                    mode_str = self._control_mode.value
                    print(
                        f"[controller] state={self.safety.state.value} "
                        f"mode={mode_str} step={step_count} "
                        f"policy_hz="
                        f"{1.0 / max(now - loop_start, 1e-6):.1f} "
                        f"vel_cmd=[{vel_cmd[0]:.1f}, {vel_cmd[1]:.1f}, "
                        f"{vel_cmd[2]:.1f}]"
                    )
                    self._last_print_time = now

                # 13. Check auto-termination
                if self._max_steps > 0 and step_count >= self._max_steps:
                    self.safety.stop()
                    self._running = False
                    break
                if self._max_duration > 0 and (
                    now - loop_start_time
                ) >= self._max_duration:
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
        self,
        state: RobotState,
        inference_time: float,
        loop_start: float,
        step_count: int,
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

    def _adjust_velocity(
        self, index: int, delta: float, vmin: float, vmax: float
    ) -> None:
        """Adjust one component of velocity command, clamped to [vmin, vmax]."""
        with self._vel_lock:
            vc = self._velocity_command.copy()
            vc[index] = np.clip(vc[index] + delta, vmin, vmax)
            self._velocity_command = vc

    def _cycle_policy(self, direction: int) -> None:
        """Cycle to next/previous active policy in policy_dir."""
        if not self._policy_files:
            return
        self._policy_index = (self._policy_index + direction) % len(
            self._policy_files
        )
        path = self._policy_files[self._policy_index]
        try:
            self.reload_policy(path)
            print(f"[controller] Loaded policy: {os.path.basename(path)}")
        except Exception as exc:
            print(f"[controller] Failed to load policy: {exc}")

    @staticmethod
    def _expand_gain(value, n: int) -> np.ndarray:
        """Expand a scalar or list gain to an (n,) array."""
        if isinstance(value, (int, float)):
            return np.full(n, value, dtype=np.float64)
        return np.array(value, dtype=np.float64)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def control_mode(self) -> ControlMode:
        """Current control mode."""
        return self._control_mode
