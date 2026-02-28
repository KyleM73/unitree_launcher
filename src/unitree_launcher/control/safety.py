"""Safety controller with state machine, damping, orientation check, and limit clamping.

Implements the safety system from SPEC section 7: E-stop behavior, damping mode,
state machine transitions, and command limit enforcement.
"""
from __future__ import annotations

import logging
import threading
from enum import Enum

import numpy as np

from unitree_launcher.config import (
    Config,
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    JOINT_LIMITS_23DOF,
    JOINT_LIMITS_29DOF,
    TORQUE_LIMITS_23DOF,
    TORQUE_LIMITS_29DOF,
    VELOCITY_LIMITS_23DOF,
    VELOCITY_LIMITS_29DOF,
)
from unitree_launcher.robot.base import RobotCommand, RobotState

logger = logging.getLogger(__name__)


class SystemState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ESTOP = "estop"


class ControlMode(Enum):
    HOLD = "hold"            # Static PD at home pose (gantry / pre-default-policy)
    DEFAULT = "default"      # Default policy running (IL velocity tracking, zero vel = stance)
    ACTIVE_POLICY = "active" # Running the --policy (BM, another IL, etc.)
    DAMPING = "damping"      # Pure velocity damping (kp=0, kd=kd_damp)
    INTERPOLATE = "interpolate"  # Cosine-ramp position + gains over a duration


class SafetyController:
    """Safety state machine with damping, orientation check, and command clamping.

    State transitions:
        IDLE -> RUNNING (start)
        RUNNING -> STOPPED (stop)
        RUNNING -> ESTOP (estop)
        STOPPED -> ESTOP (estop)
        ESTOP -> STOPPED (clear_estop)

    Thread safety: all state transitions are protected by a lock.
    """

    def __init__(self, config: Config, n_dof: int):
        self._control_config = config.control
        self._safety_config = config.safety
        self._n_dof = n_dof
        self._state = SystemState.IDLE
        self._lock = threading.Lock()

        # Build limit arrays from config constants
        variant = config.robot.variant
        if variant == "g1_29dof":
            joints = G1_29DOF_JOINTS
            pos_limits = JOINT_LIMITS_29DOF
            torque_limits = TORQUE_LIMITS_29DOF
            vel_limits = VELOCITY_LIMITS_29DOF
        else:
            joints = G1_23DOF_JOINTS
            pos_limits = JOINT_LIMITS_23DOF
            torque_limits = TORQUE_LIMITS_23DOF
            vel_limits = VELOCITY_LIMITS_23DOF

        self._joints = list(joints)
        self._pos_min = np.array([pos_limits[j][0] for j in joints])
        self._pos_max = np.array([pos_limits[j][1] for j in joints])
        self._torque_max = np.array([torque_limits[j] for j in joints])
        self._vel_max = np.array([vel_limits[j] for j in joints])

    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    def start(self) -> bool:
        """IDLE/STOPPED -> RUNNING. Returns False if transition invalid."""
        with self._lock:
            if self._state in (SystemState.IDLE, SystemState.STOPPED):
                self._state = SystemState.RUNNING
                return True
            return False

    def stop(self) -> bool:
        """RUNNING -> STOPPED. Returns False if transition invalid."""
        with self._lock:
            if self._state == SystemState.RUNNING:
                self._state = SystemState.STOPPED
                return True
            return False

    def estop(self) -> None:
        """Trigger E-stop. Always succeeds from non-IDLE states. Latching."""
        with self._lock:
            if self._state in (SystemState.RUNNING, SystemState.STOPPED, SystemState.ESTOP):
                self._state = SystemState.ESTOP

    def clear_estop(self) -> bool:
        """ESTOP -> STOPPED. Returns False if not in ESTOP state."""
        with self._lock:
            if self._state == SystemState.ESTOP:
                self._state = SystemState.STOPPED
                return True
            return False

    def get_damping_command(self, current_state: RobotState) -> RobotCommand:
        """Generate damping command: target_pos = current_pos, kp=0, kd=kd_damp.

        The resulting torque is: tau = -kd_damp * q_dot (pure velocity damping).
        kd_damp must be high enough to overcome gravity torques (~15 Nm on
        legs) — MuJoCo's per-actuator forcerange automatically clips the
        torque for weaker joints (e.g. wrists at ±5 Nm).
        """
        kd_damp = self._control_config.kd_damp
        return RobotCommand(
            joint_positions=current_state.joint_positions.copy(),
            joint_velocities=np.zeros(self._n_dof),
            joint_torques=np.zeros(self._n_dof),
            kp=np.zeros(self._n_dof),
            kd=np.full(self._n_dof, kd_damp),
        )

    def check_orientation(self, imu_quaternion: np.ndarray) -> tuple:
        """Check if robot orientation is safe (for real robot startup).

        Safe = projected gravity Z component > 0.8 (roughly < 35 deg from vertical).

        Returns:
            (is_safe, message) tuple.
        """
        # Compute projected gravity: R^T @ [0, 0, -1]
        w, x, y, z = imu_quaternion
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        gz = gravity_body[2]

        # Upright robot has projected gravity ≈ [0, 0, -1], so gz ≈ -1.
        # We check abs(gz) > 0.8 but actually we want gz < -0.8 (pointing down).
        # The spec says "Z component > 0.8" which for gravity = [0,0,-1] means
        # the magnitude of the downward component. Let's use gz < -0.8.
        if gz < -0.8:
            return (True, "Orientation OK")
        else:
            angle_deg = np.degrees(np.arccos(np.clip(-gz, -1.0, 1.0)))
            return (False, f"Unsafe orientation: {angle_deg:.1f} deg from vertical (limit ~35 deg)")

    def clamp_command(self, cmd: RobotCommand) -> RobotCommand:
        """Enforce safety limits on a command.

        Clips joint positions, velocities, and torques when the corresponding
        SafetyConfig booleans are enabled. Returns a new command (does not
        modify the input).
        """
        result = RobotCommand(
            joint_positions=cmd.joint_positions.copy(),
            joint_velocities=cmd.joint_velocities.copy(),
            joint_torques=cmd.joint_torques.copy(),
            kp=cmd.kp.copy(),
            kd=cmd.kd.copy(),
        )

        if self._safety_config.joint_position_limits:
            result.joint_positions = np.clip(
                result.joint_positions, self._pos_min, self._pos_max
            )

        if self._safety_config.joint_velocity_limits:
            result.joint_velocities = np.clip(
                result.joint_velocities, -self._vel_max, self._vel_max
            )

        if self._safety_config.torque_limits:
            result.joint_torques = np.clip(
                result.joint_torques, -self._torque_max, self._torque_max
            )

        return result

    def check_state_limits(self, state: RobotState) -> bool:
        """Check measured robot state against safety thresholds.

        Monitors actual joint positions, velocities, and torques (not commands).
        If any measured value exceeds ``fault_threshold`` fraction of its limit,
        triggers E-stop and logs the offending joint.

        Each check respects the corresponding SafetyConfig boolean toggle.
        Skips entirely if in IDLE (pre-operational) or ESTOP (avoids log spam).

        Returns:
            True if a fault was detected (ESTOP triggered), False otherwise.
        """
        if self._state in (SystemState.IDLE, SystemState.ESTOP):
            return False

        threshold = self._safety_config.fault_threshold

        if self._safety_config.joint_position_limits:
            pos = state.joint_positions
            margin = (1.0 - threshold) * (self._pos_max - self._pos_min)
            low = self._pos_min + margin
            high = self._pos_max - margin
            for i in range(self._n_dof):
                if pos[i] < low[i] or pos[i] > high[i]:
                    logger.error(
                        "State fault: %s position %.4f outside threshold "
                        "[%.4f, %.4f] (limits [%.4f, %.4f])",
                        self._joints[i], pos[i], low[i], high[i],
                        self._pos_min[i], self._pos_max[i],
                    )
                    self.estop()
                    return True

        if self._safety_config.joint_velocity_limits:
            vel = state.joint_velocities
            vel_thresh = threshold * self._vel_max
            for i in range(self._n_dof):
                if abs(vel[i]) > vel_thresh[i]:
                    logger.error(
                        "State fault: %s velocity %.4f exceeds threshold "
                        "%.4f (limit %.4f)",
                        self._joints[i], vel[i], vel_thresh[i],
                        self._vel_max[i],
                    )
                    self.estop()
                    return True

        if self._safety_config.torque_limits:
            tau = state.joint_torques
            tau_thresh = threshold * self._torque_max
            for i in range(self._n_dof):
                if abs(tau[i]) > tau_thresh[i]:
                    logger.error(
                        "State fault: %s torque %.4f exceeds threshold "
                        "%.4f (limit %.4f)",
                        self._joints[i], tau[i], tau_thresh[i],
                        self._torque_max[i],
                    )
                    self.estop()
                    return True

        return False
