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

from src.config import Config, G1_29DOF_JOINTS, G1_23DOF_JOINTS, Q_HOME_29DOF, Q_HOME_23DOF
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

        # BeyondMimic end-of-trajectory interpolation
        self._interpolating = False
        self._interp_start_pos: Optional[np.ndarray] = None
        self._interp_step: int = 0
        self._interp_total_steps: int = int(2.0 * config.control.policy_frequency)  # 2 seconds

        # Stdout status
        self._last_print_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the control loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._time_step = 0
        self._interpolating = False
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
        self._interpolating = False

    def handle_key(self, key: str) -> None:
        """Handle keyboard input (from MuJoCo viewer or CLI).

        Called from the main thread. Thread-safe.
        """
        if key == "space":
            if self.safety.state in (SystemState.IDLE, SystemState.STOPPED):
                self.safety.start()
                self.start()
            elif self.safety.state == SystemState.RUNNING:
                self.safety.stop()
                self.stop()
        elif key == "e":
            self.safety.estop()
        elif key == "c":
            self.safety.clear_estop()
        elif key == "r":
            self.robot.reset()
        elif key == "w":
            self._adjust_velocity(0, 0.1, -1.0, 1.0)
        elif key == "s":
            self._adjust_velocity(0, -0.1, -1.0, 1.0)
        elif key == "a":
            self._adjust_velocity(1, 0.1, -0.5, 0.5)
        elif key == "d":
            self._adjust_velocity(1, -0.1, -0.5, 0.5)
        elif key == "q":
            self._adjust_velocity(2, 0.1, -1.0, 1.0)
        elif key == "z":
            self._adjust_velocity(2, -0.1, -1.0, 1.0)
        elif key == "x":
            with self._vel_lock:
                self._velocity_command = np.zeros(3)
        elif key == "n":
            self._cycle_policy(1)
        elif key == "p":
            self._cycle_policy(-1)

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(self, state: RobotState, action: np.ndarray) -> RobotCommand:
        """Build a RobotCommand from policy action output.

        For IsaacLab:
            target_pos = q_home + Ka * action
            kp, kd from config; dq_target = 0, tau = 0

        For BeyondMimic:
            target_pos = target_q + Ka * action
            kp, kd from metadata (or config); dq_target = target_dq, tau = 0
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

            ctrl_target_pos = bm.target_q + ctrl_ka * action
            ctrl_target_vel = bm.target_dq.copy()
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

                # 2. Not RUNNING: keep sim alive but don't send commands
                if self.safety.state != SystemState.RUNNING:
                    self.robot.step()
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 3. Get robot state
                state = self.robot.get_state()

                # 4. Handle BeyondMimic end-of-trajectory interpolation
                if self._interpolating:
                    cmd = self._build_interpolation_command(state)
                    cmd = self.safety.clamp_command(cmd)
                    self.robot.send_command(cmd)
                    self.robot.step()
                    self._interp_step += 1
                    if self._interp_step >= self._interp_total_steps:
                        self._interpolating = False
                        self.safety.stop()
                        self._running = False
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 5. Build observation and run policy inference
                inference_start = time.perf_counter()
                if self.obs_builder is not None:
                    vel_cmd = self.get_velocity_command()
                    obs = self.obs_builder.build(state, last_action, vel_cmd)
                    action = self.policy.get_action(obs)
                else:
                    # BeyondMimic: policy builds its own observations
                    from src.policy.beyondmimic_policy import BeyondMimicPolicy
                    bm: BeyondMimicPolicy = self.policy
                    # Get anchor body state from sim
                    anchor_pos = state.base_position
                    anchor_quat = state.imu_quaternion
                    obs = bm.build_observation(state, anchor_pos, anchor_quat)
                    action = bm.get_action(obs, time_step=self._time_step)
                    self._time_step += 1
                inference_time = time.perf_counter() - inference_start

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

    @staticmethod
    def _expand_gain(value, n: int) -> np.ndarray:
        """Expand a scalar or list gain to an (n,) array."""
        if isinstance(value, (int, float)):
            return np.full(n, value, dtype=np.float64)
        return np.array(value, dtype=np.float64)

    @property
    def is_running(self) -> bool:
        return self._running
