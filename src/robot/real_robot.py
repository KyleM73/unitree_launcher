"""Real robot backend via DDS communication.

Implements the RobotInterface for the physical Unitree G1 robot. Communicates
over CycloneDDS using unitree_hg IDL types (LowCmd_ / LowState_).

Threading model:
    DDS subscriber thread:  receives LowState_ callbacks, copies into _latest_state
    Control loop thread:    get_state() -> policy -> send_command()
    Watchdog:               checked each get_state(), triggers E-stop on timeout

SAFETY WARNING: All development should happen in simulation first. Real robot
testing should start with the robot hanging from a support harness.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from src.compat import patch_unitree_threading, resolve_network_interface
from src.config import Config, _get_joints_for_variant
from src.robot.base import RobotCommand, RobotInterface, RobotState

logger = logging.getLogger(__name__)

# DDS IDL motor count for unitree_hg messages
_NUM_MOTOR_IDL_HG = 35

# PMSM servo mode flag for motor commands
_MOTOR_MODE_SERVO = 0x01

# Watchdog timeout: trigger E-stop if no state received within this window
_WATCHDOG_TIMEOUT_S = 0.1  # 100 ms

# Connect timeout: fail if no state message within this window
_CONNECT_TIMEOUT_S = 5.0


class RealRobot(RobotInterface):
    """Real robot backend communicating over DDS.

    Uses unitree_hg IDL types with domain_id=0 (Unitree default for real robot).
    """

    def __init__(self, config: Config):
        self._config = config
        variant = config.robot.variant

        self._cfg_joints = _get_joints_for_variant(variant)
        self._n_dof = len(self._cfg_joints)

        # DDS state (lazy init in connect())
        self._connected = False
        self._cmd_pub = None
        self._state_sub = None
        self._low_cmd_msg = None
        self._crc = None

        # Latest state from DDS subscriber callback (protected by lock)
        self._state_lock = threading.Lock()
        self._latest_state: Optional[RobotState] = None
        self._last_state_time: float = 0.0
        self._state_received = threading.Event()

        # Optional safety controller reference (set externally for watchdog)
        self._safety = None

    def set_safety(self, safety) -> None:
        """Set safety controller reference for watchdog E-stop."""
        self._safety = safety

    # ---- RobotInterface implementation ----

    def connect(self) -> None:
        """Initialize DDS and verify connection by waiting for first state message."""
        if self._connected:
            return

        patch_unitree_threading()

        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        # Real robot uses domain_id=0
        iface = resolve_network_interface(self._config.network.interface)
        ChannelFactoryInitialize(0, iface)

        # Command publisher
        self._low_cmd_msg = unitree_hg_msg_dds__LowCmd_()
        self._cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._cmd_pub.Init()

        # State subscriber with callback
        self._state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_sub.Init(handler=self._on_low_state, queueLen=1)

        # CRC calculator
        self._crc = CRC()

        # Wait for first state message
        self._state_received.clear()
        if not self._state_received.wait(timeout=_CONNECT_TIMEOUT_S):
            raise TimeoutError(
                f"No state message received within {_CONNECT_TIMEOUT_S}s. "
                "Is the robot powered on and connected?"
            )

        self._connected = True
        logger.info("RealRobot connected on interface %s (domain_id=0)", iface)

        # Print startup orientation check
        state = self.get_state()
        logger.info(
            "IMU quaternion: [%.3f, %.3f, %.3f, %.3f]",
            *state.imu_quaternion,
        )

    def disconnect(self) -> None:
        """Clean up DDS resources."""
        if self._state_sub is not None:
            self._state_sub.Close()
            self._state_sub = None
        self._cmd_pub = None
        self._connected = False
        logger.info("RealRobot disconnected")

    def get_state(self) -> RobotState:
        """Return latest state from DDS subscription. Thread-safe.

        Also checks watchdog — if state is stale, triggers E-stop.
        """
        with self._state_lock:
            if self._latest_state is None:
                return RobotState.zeros(self._n_dof)

            # Watchdog check
            elapsed = time.monotonic() - self._last_state_time
            if elapsed > _WATCHDOG_TIMEOUT_S and self._safety is not None:
                logger.warning(
                    "Watchdog: no state for %.1f ms, triggering E-stop",
                    elapsed * 1000,
                )
                self._safety.estop()

            return self._latest_state.copy()

    def send_command(self, cmd: RobotCommand) -> None:
        """Publish LowCmd_ to rt/lowcmd via DDS.

        Sets motor_cmd[i].mode = 0x01 (PMSM servo mode) for each controlled
        joint and computes CRC32 before publishing.
        """
        if self._cmd_pub is None or self._low_cmd_msg is None:
            return

        msg = self._low_cmd_msg

        for cfg_i in range(self._n_dof):
            motor = msg.motor_cmd[cfg_i]
            motor.mode = _MOTOR_MODE_SERVO
            motor.q = float(cmd.joint_positions[cfg_i])
            motor.dq = float(cmd.joint_velocities[cfg_i])
            motor.tau = float(cmd.joint_torques[cfg_i])
            motor.kp = float(cmd.kp[cfg_i])
            motor.kd = float(cmd.kd[cfg_i])

        # Compute and set CRC32
        msg.crc = self._crc.Crc(msg)

        self._cmd_pub.Write(msg)

    def step(self) -> None:
        """No-op for real robot (physics runs on hardware)."""
        pass

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Log warning — cannot reset a physical robot."""
        logger.warning(
            "reset() called on RealRobot — cannot reset physical robot. Ignoring."
        )

    @property
    def n_dof(self) -> int:
        return self._n_dof

    # ---- Private ----

    def _on_low_state(self, msg) -> None:
        """DDS callback: convert LowState_ to RobotState and store."""
        now = time.monotonic()

        state = RobotState(
            timestamp=time.time(),
            joint_positions=np.array(
                [msg.motor_state[i].q for i in range(self._n_dof)],
                dtype=np.float64,
            ),
            joint_velocities=np.array(
                [msg.motor_state[i].dq for i in range(self._n_dof)],
                dtype=np.float64,
            ),
            joint_torques=np.array(
                [msg.motor_state[i].tau_est for i in range(self._n_dof)],
                dtype=np.float64,
            ),
            imu_quaternion=np.array(
                [msg.imu_state.quaternion[j] for j in range(4)],
                dtype=np.float64,
            ),
            imu_angular_velocity=np.array(
                [msg.imu_state.gyroscope[j] for j in range(3)],
                dtype=np.float64,
            ),
            imu_linear_acceleration=np.array(
                [msg.imu_state.accelerometer[j] for j in range(3)],
                dtype=np.float64,
            ),
            # Real robot has no world-frame base pose from DDS
            base_position=np.full(3, np.nan),
            base_velocity=np.full(3, np.nan),
        )

        with self._state_lock:
            self._latest_state = state
            self._last_state_time = now

        # Signal first message received (for connect() wait)
        self._state_received.set()
