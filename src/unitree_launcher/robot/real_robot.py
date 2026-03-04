"""Real robot backend — runs onboard the G1 via C++ unitree_interface binding.

Wraps the ``unitree_interface`` pybind11 module (from unitree_sdk2) which
handles DDS communication with the motor control board over the internal
Ethernet bus, 500 Hz command re-publishing, CRC computation, and motor
mode management entirely in C++ — eliminating Python GIL jitter.

Mirror mode uses a separate DDS backend for read-only visualization.

Safety note:
    The software E-stop (wireless A-button parsed in ``get_state()``)
    only works while the Python control loop is running.  If Python
    hangs, the robot continues executing its last command (re-published
    at 500 Hz by the C++ binding).  The **hardware-level fallback** is
    the wireless controller's **L2+B** combination, which the motor
    control board acts on independently of any software.

Usage:
    from unitree_launcher.robot.real_robot import RealRobot
    robot = RealRobot(config)
    robot.connect()
    state = robot.get_state()
    robot.send_command(cmd)
    robot.graceful_shutdown()
"""
from __future__ import annotations

import logging
import struct
import time
from typing import Callable, Optional

import numpy as np

from unitree_launcher.config import Config, _get_joints_for_variant
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

logger = logging.getLogger(__name__)


class RealRobot(RobotInterface):
    """Real robot backend — runs onboard via C++ unitree_interface binding.

    The C++ layer handles:
    - 500 Hz command re-publishing (OS-level timer precision)
    - CRC32 computation
    - Motor mode / mode_machine echo
    - DDS channel initialization

    Python only needs to call ``send_command()`` at the policy rate (50 Hz).
    """

    def __init__(self, config: Config):
        variant = config.robot.variant

        self._n_dof = len(_get_joints_for_variant(variant))

        # Network interface name (e.g. "en8", "eth0")
        self._iface_name = config.network.interface

        # C++ interface handle (lazy init in connect())
        self._connected = False
        self._interface = None

        # Optional safety controller reference (for watchdog E-stop)
        self._safety = None

        # Wireless controller callback (set via set_wireless_handler)
        self._wireless_handler: Optional[Callable[[bytes], None]] = None


    def set_safety(self, safety) -> None:
        """Set safety controller reference for watchdog E-stop."""
        self._safety = safety

    def set_wireless_handler(self, handler: Callable[[bytes], None]) -> None:
        """Register a callback to receive wireless controller bytes each tick.

        Called from ``get_state()`` with the raw ``wireless_remote`` bytes
        from the C++ binding's LowState.  Typically wired to
        ``WirelessInput.parse()``.
        """
        self._wireless_handler = handler

    # ---- RobotInterface implementation ----

    def connect(self) -> None:
        """Initialize C++ binding and verify connection."""
        if self._connected:
            return

        try:
            import unitree_interface
        except ImportError as exc:
            raise ImportError(
                "unitree_interface (C++ pybind11 binding) is not installed.\n"
                "On the G1 robot, build from source:\n"
                "  ./scripts/build_cpp_backend.sh\n"
                "See reference/RoboJuDo/docs/g1_onboard_deploy.md for setup."
            ) from exc

        self._interface = unitree_interface.create_robot(
            self._iface_name,
            unitree_interface.RobotType.G1,
            unitree_interface.MessageType.HG,
        )
        self._interface.set_control_mode(unitree_interface.ControlMode.PR)

        # Verify connection by reading first state
        state = self._interface.read_low_state()
        self._connected = True

        logger.info(
            "RealRobot connected on interface %s (C++ binding) — "
            "mode_machine=%d",
            self._iface_name,
            getattr(state, 'mode_machine', 0),
        )

        # Print startup orientation check
        imu_quat = np.array(state.imu.quat[:4])
        logger.info("IMU quaternion: [%.3f, %.3f, %.3f, %.3f]", *imu_quat)

    def graceful_shutdown(self, damping_duration: float = 0.5) -> None:
        """Send damping commands then disconnect.

        The C++ binding continues re-publishing the damping command at 500 Hz
        during the sleep, so the robot sees a smooth deceleration.
        """
        if not self._connected:
            return

        logger.info(
            "Graceful shutdown: sending damping commands for %.1fs",
            damping_duration,
        )

        try:
            cmd = RobotCommand.damping(self._n_dof)
            self.send_command(cmd)
            # C++ re-publishes during sleep
            time.sleep(damping_duration)
        except Exception:
            logger.exception("Error during graceful shutdown damping")

        self.disconnect()

    def disconnect(self) -> None:
        """Mark as disconnected. C++ binding cleans up on garbage collection."""
        self._interface = None
        self._connected = False
        logger.info("RealRobot disconnected")

    def get_state(self) -> RobotState:
        """Read current robot state from C++ binding.

        Also extracts wireless controller bytes and:
        1. Passes them to the registered wireless handler (for button/stick parsing)
        2. Checks the A-button directly for immediate E-stop (tightest loop)
        """
        if self._interface is None:
            return RobotState.zeros(self._n_dof)

        state = self._interface.read_low_state()

        # Extract and process wireless controller data
        remote = getattr(state, 'wireless_remote', None)
        if remote is not None and len(remote) >= 24:
            # Tight A-button check: parse button field and trigger E-stop
            # immediately, before any policy inference runs.
            # A-button is bit 8 in the 16-bit field at bytes 2-3.
            keys = struct.unpack("H", bytes(remote[2:4]))[0]
            if keys & (1 << 8) and self._safety is not None:
                logger.critical("Wireless A-button E-stop!")
                self._safety.estop()
                self.send_command(RobotCommand.damping(self._n_dof))

            # Forward full bytes to the wireless input controller
            if self._wireless_handler is not None:
                self._wireless_handler(bytes(remote))

        return RobotState(
            timestamp=time.time(),
            joint_positions=np.array(state.motor.q[:self._n_dof], dtype=np.float64),
            joint_velocities=np.array(state.motor.dq[:self._n_dof], dtype=np.float64),
            joint_torques=np.array(state.motor.tau_est[:self._n_dof], dtype=np.float64),
            imu_quaternion=np.array(state.imu.quat[:4], dtype=np.float64),
            imu_angular_velocity=np.array(state.imu.omega[:3], dtype=np.float64),
            imu_linear_acceleration=np.array(state.imu.accel[:3], dtype=np.float64),
            # Real robot has no world-frame base pose
            base_position=np.full(3, np.nan),
            base_velocity=np.full(3, np.nan),
        )

    def send_command(self, cmd: RobotCommand) -> None:
        """Send command via C++ binding. No CRC, no mode echo needed."""
        if self._interface is None:
            return

        motor_cmd = self._interface.create_zero_command()
        motor_cmd.q_target = cmd.joint_positions.tolist()
        motor_cmd.dq_target = cmd.joint_velocities.tolist()
        motor_cmd.tau_ff = cmd.joint_torques.tolist()
        motor_cmd.kp = cmd.kp.tolist()
        motor_cmd.kd = cmd.kd.tolist()
        self._interface.write_low_command(motor_cmd)

    def step(self) -> None:
        """No-op for real robot (hardware runs PD at ~500 Hz)."""
        pass

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Log warning -- cannot reset a physical robot."""
        logger.warning(
            "reset() called on RealRobot -- cannot reset physical robot. Ignoring."
        )

    @property
    def n_dof(self) -> int:
        return self._n_dof
