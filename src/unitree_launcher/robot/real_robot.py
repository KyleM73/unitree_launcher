"""Real robot backend — runs onboard the G1 via unitree_cpp binding.

Wraps ``unitree_cpp.UnitreeController`` (from HansZ8/unitree_cpp) which
handles DDS communication, CRC computation, motor mode management, and
background command re-publishing at control_dt (20ms) entirely in C++.

The C++ layer has a RecurrentThread that keeps publishing the last command
even if Python hiccups — ensuring the robot always receives fresh commands.

Safety note:
    The software E-stop (wireless A-button parsed in ``get_state()``)
    only works while the Python control loop is running.  If Python
    hangs, the C++ layer continues re-publishing its last command.
    The **hardware-level fallback** is the wireless controller's
    **L2+B** combination, which the motor control board acts on
    independently of any software.
"""
from __future__ import annotations

import logging
import struct
import time
from typing import Callable, Optional

import numpy as np

from unitree_launcher.config import Config, _get_joints_for_variant
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState, SdkState

logger = logging.getLogger(__name__)


class RealRobot(RobotInterface):
    """Real robot backend using unitree_cpp (RoboJuDo's C++ binding).

    The C++ layer handles:
    - Background command re-publishing at control_dt (20ms)
    - CRC32 computation
    - Motor mode / mode_machine echo
    - MotionSwitcher service release
    - DDS channel initialization
    """

    def __init__(self, config: Config):
        variant = config.robot.variant

        self._n_dof = len(_get_joints_for_variant(variant))
        self._iface_name = config.network.interface
        self._connected = False
        self._controller = None
        self._safety = None
        self._wireless_handler: Optional[Callable[[bytes], None]] = None

    def set_safety(self, safety) -> None:
        """Set safety controller reference for watchdog E-stop."""
        self._safety = safety

    def set_wireless_handler(self, handler: Callable[[bytes], None]) -> None:
        """Register a callback to receive wireless controller bytes each tick."""
        self._wireless_handler = handler

    def connect(self) -> None:
        """Initialize unitree_cpp controller and verify connection."""
        if self._connected:
            return

        try:
            from unitree_cpp import UnitreeController
        except ImportError as exc:
            raise ImportError(
                "unitree_cpp is not installed.\n"
                "On the G1 robot, build from source:\n"
                "  ./scripts/build_cpp_backend.sh\n"
            ) from exc

        config_dict = {
            "net_if": self._iface_name,
            "control_dt": 0.02,
            "msg_type": "hg",
            "control_mode": "position",
            "hand_type": "NONE",
            "lowcmd_topic": "rt/lowcmd",
            "lowstate_topic": "rt/lowstate",
            "enable_odometry": False,
            "sport_state_topic": "rt/odommodestate",
            "stiffness": [0.0] * self._n_dof,
            "damping": [0.0] * self._n_dof,
            "num_dofs": self._n_dof,
        }
        self._controller = UnitreeController(config_dict)

        # Wait for state data (matching RoboJuDo's self_check pattern)
        for _ in range(30):
            time.sleep(0.1)
            if self._controller.self_check():
                break
        if not self._controller.self_check():
            raise RuntimeError(
                "unitree_cpp self-check failed: no data from robot. "
                "Check Ethernet cable and robot power."
            )

        self._connected = True
        logger.info("RealRobot connected via unitree_cpp on %s", self._iface_name)

        # Print startup orientation
        state = self._controller.get_robot_state()
        quat = state.imu_state.quaternion
        logger.info("IMU quaternion: [%.3f, %.3f, %.3f, %.3f]", *quat)

    def graceful_shutdown(self, damping_duration: float = 0.5) -> None:
        """Send damping command via C++ shutdown, then wait."""
        if not self._connected or self._controller is None:
            return

        logger.info("Graceful shutdown: damping for %.1fs", damping_duration)
        try:
            self._controller.shutdown()
            time.sleep(damping_duration)
        except Exception:
            logger.exception("Error during graceful shutdown")

        self._connected = False

    def disconnect(self) -> None:
        """Mark as disconnected."""
        self._controller = None
        self._connected = False
        logger.info("RealRobot disconnected")

    def get_state(self) -> RobotState:
        """Read current robot state from C++ binding.

        Also extracts wireless controller bytes and:
        1. Passes them to the registered wireless handler (for button/stick parsing)
        2. Checks the A-button directly for immediate E-stop (tightest loop)
        """
        if self._controller is None:
            return RobotState.zeros(self._n_dof)

        state = self._controller.get_robot_state()

        # Extract and process wireless controller data
        remote = state.wireless_remote
        if remote is not None and len(remote) >= 24:
            keys = struct.unpack("H", remote[2:4])[0]
            if keys & (1 << 8) and self._safety is not None:
                logger.critical("Wireless A-button E-stop!")
                self._safety.estop()
                # Send damping via set_gains + step
                self._controller.set_gains(
                    [0.0] * self._n_dof, [8.0] * self._n_dof
                )
                self._controller.step([0.0] * self._n_dof)

            if self._wireless_handler is not None:
                self._wireless_handler(remote)

        # The C++ binding exposes motor_state as a single object with flat
        # arrays (motor_state.q[:], motor_state.dq[:], etc.), not per-motor
        # objects like the Python DDS LowState_.
        ms = state.motor_state
        n_motors = 35

        # Build SdkState from all available LowState_ fields.
        # Use getattr with defaults since the C++ binding may not expose all.
        sdk_state = SdkState(
            tick=int(getattr(state, 'tick', 0)),
            mode_pr=int(getattr(state, 'mode_pr', 0)),
            mode_machine=int(getattr(state, 'mode_machine', 0)),
            motor_ddq=np.array(getattr(ms, 'ddq', [0.0] * n_motors)[:n_motors], dtype=np.float32),
            motor_temperature=np.array(getattr(ms, 'temperature', [[0, 0]] * n_motors)[:n_motors], dtype=np.int16),
            motor_voltage=np.array(getattr(ms, 'vol', [0.0] * n_motors)[:n_motors], dtype=np.float32),
            motor_state_flags=np.array(getattr(ms, 'motorstate', [0] * n_motors)[:n_motors], dtype=np.uint32),
            imu_rpy=np.array(getattr(state.imu_state, 'rpy', [0.0, 0.0, 0.0])[:3], dtype=np.float32),
            imu_temperature=int(getattr(state.imu_state, 'temperature', 0)),
        )

        return RobotState(
            timestamp=time.time(),
            joint_positions=np.array(ms.q[:self._n_dof], dtype=np.float64),
            joint_velocities=np.array(ms.dq[:self._n_dof], dtype=np.float64),
            joint_torques=np.array(ms.tau_est[:self._n_dof], dtype=np.float64),
            imu_quaternion=np.array(state.imu_state.quaternion[:4], dtype=np.float64),
            imu_angular_velocity=np.array(state.imu_state.gyroscope[:3], dtype=np.float64),
            imu_linear_acceleration=np.array(state.imu_state.accelerometer[:3], dtype=np.float64),
            base_position=np.full(3, np.nan),
            base_velocity=np.full(3, np.nan),
            sdk_state=sdk_state,
        )

    def send_command(self, cmd: RobotCommand) -> None:
        """Send command via unitree_cpp: set gains then step with positions."""
        if self._controller is None:
            return

        self._controller.set_gains(cmd.kp.tolist(), cmd.kd.tolist())
        self._controller.step(cmd.joint_positions.tolist())

    def step(self) -> None:
        """No-op for real robot (C++ handles re-publishing)."""
        pass

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Log warning -- cannot reset a physical robot."""
        logger.warning("reset() called on RealRobot -- ignoring.")

    @property
    def n_dof(self) -> int:
        return self._n_dof
