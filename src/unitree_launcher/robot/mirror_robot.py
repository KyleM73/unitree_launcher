"""Read-only DDS subscriber for mirroring real robot state.

NOT used for robot control — see ``real_robot.py`` for the onboard C++
backend. This module subscribes to LowState_ over Ethernet so diagnostic
scripts (e.g., ``uv run mirror``) can visualize the robot in MuJoCo.

Communicates over CycloneDDS using unitree_hg IDL types (LowState_).
Requires ``unitree_sdk2py`` and ``cyclonedds`` packages.

SAFETY WARNING: All development should happen in simulation first. Real robot
testing should start with the robot hanging from a support harness.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from unitree_launcher.compat import RecurrentThread, patch_unitree_b2_import, patch_unitree_crc, patch_unitree_threading, resolve_network_interface
from unitree_launcher.config import Config, _get_joints_for_variant
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

logger = logging.getLogger(__name__)

# DDS IDL motor count for unitree_hg messages
_NUM_MOTOR_IDL_HG = 35

# PMSM servo mode flag for motor commands
_MOTOR_MODE_SERVO = 0x01

# Watchdog timeout: trigger E-stop if no state received within this window
_WATCHDOG_TIMEOUT_S = 0.1  # 100 ms

# Connect timeout: fail if no state message within this window
_CONNECT_TIMEOUT_S = 5.0


class MirrorRobot(RobotInterface):
    """Read-only DDS subscriber for mirroring real robot state into sim.

    Reads LowState_ from the real robot over Ethernet via Python DDS.
    Used by ``scripts/mirror_real_robot.py`` to visualize the robot's
    actual joint positions in a MuJoCo viewer. NOT for motor control —
    see ``real_robot.RealRobot`` for onboard deployment.
    """

    def __init__(self, config: Config):
        self._config = config
        variant = config.robot.variant

        self._n_dof = len(_get_joints_for_variant(variant))

        # DDS state (lazy init in connect())
        self._connected = False
        self._cmd_pub = None
        self._state_sub = None
        self._low_cmd_msg = None
        self._crc = None
        self._publish_thread: Optional[RecurrentThread] = None

        # Command lock: protects _low_cmd_msg and _cmd_ready.
        # Lock ordering: _cmd_lock -> _state_lock (never reversed).
        self._cmd_lock = threading.Lock()
        self._cmd_ready = False

        # Latest state from DDS subscriber callback (protected by lock)
        self._state_lock = threading.Lock()
        self._latest_state: Optional[RobotState] = None
        self._last_state_time: float = 0.0
        self._state_received = threading.Event()

        # Hardware identification from LowState (protected by _state_lock)
        self._mode_machine: int = 0
        self._version: tuple[int, int] = (0, 0)

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

        patch_unitree_b2_import()
        patch_unitree_threading()
        patch_unitree_crc()

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

        # Start 500 Hz command re-publish thread
        self._publish_thread = RecurrentThread(
            interval=0.002, target=self._publish_cmd, name="cmd_publish"
        )
        self._publish_thread.Start()

        self._connected = True

        # Log hardware identification
        with self._state_lock:
            mode_machine = self._mode_machine
            version = self._version
        logger.info(
            "MirrorRobot connected on interface %s (domain_id=0) — "
            "mode_machine=%d, version=(%d, %d)",
            iface, mode_machine, version[0], version[1],
        )

        # Print startup orientation check
        state = self.get_state()
        logger.info(
            "IMU quaternion: [%.3f, %.3f, %.3f, %.3f]",
            *state.imu_quaternion,
        )

    def graceful_shutdown(self, damping_duration: float = 0.5) -> None:
        """Send damping commands then disconnect. Safe to call from signal handlers.

        Sends zero-torque damping commands for *damping_duration* seconds so the
        robot decelerates smoothly before the command stream stops. Idempotent —
        calling multiple times is harmless.

        Args:
            damping_duration: How long to send damping commands (seconds).
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

            # Let the 500 Hz publish thread re-send the damping command
            deadline = time.monotonic() + damping_duration
            while time.monotonic() < deadline:
                time.sleep(0.01)
        except Exception:
            logger.exception("Error during graceful shutdown damping")

        self.disconnect()

    def disconnect(self) -> None:
        """Clean up DDS resources. Shuts down publish thread first."""
        if self._publish_thread is not None:
            self._publish_thread.Shutdown()
            self._publish_thread = None
        if self._state_sub is not None:
            self._state_sub.Close()
            self._state_sub = None
        self._cmd_pub = None
        self._connected = False
        self._cmd_ready = False
        logger.info("MirrorRobot disconnected")

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
        """Build LowCmd_ and publish immediately + mark ready for 500Hz re-publish.

        Sets motor_cmd[i].mode = 0x01 (PMSM servo mode) for each controlled
        joint, zeros out non-controlled slots 29-34, echoes mode_machine from
        the latest LowState, sets mode_pr=0, and computes CRC32 before
        publishing.
        """
        if self._cmd_pub is None or self._low_cmd_msg is None:
            return

        # Read mode_machine from latest state (brief lock)
        with self._state_lock:
            mode_machine = self._mode_machine

        with self._cmd_lock:
            msg = self._low_cmd_msg

            # Echo mode_machine from LowState
            msg.mode_machine = mode_machine

            # Pitch/roll ankle mode: 0 = position control
            msg.mode_pr = 0

            # Fill controlled joints
            for cfg_i in range(self._n_dof):
                motor = msg.motor_cmd[cfg_i]
                motor.mode = _MOTOR_MODE_SERVO
                motor.q = float(cmd.joint_positions[cfg_i])
                motor.dq = float(cmd.joint_velocities[cfg_i])
                motor.tau = float(cmd.joint_torques[cfg_i])
                motor.kp = float(cmd.kp[cfg_i])
                motor.kd = float(cmd.kd[cfg_i])

            # Zero out non-controlled motor slots (29-34)
            for i in range(self._n_dof, _NUM_MOTOR_IDL_HG):
                motor = msg.motor_cmd[i]
                motor.mode = 0
                motor.q = 0.0
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = 0.0
                motor.kd = 0.0

            # Compute and set CRC32
            msg.crc = self._crc.Crc(msg)

            self._cmd_ready = True

            # Publish immediately (first send, don't wait for 500Hz tick)
            self._cmd_pub.Write(msg)

    def step(self) -> None:
        """No-op for real robot (physics runs on hardware)."""
        pass

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Log warning — cannot reset a physical robot."""
        logger.warning(
            "reset() called on MirrorRobot — cannot reset physical robot. Ignoring."
        )

    @property
    def n_dof(self) -> int:
        return self._n_dof

    # ---- Private ----

    def _publish_cmd(self) -> None:
        """500 Hz re-publish target. No-op until first send_command() sets _cmd_ready."""
        with self._cmd_lock:
            if not self._cmd_ready or self._cmd_pub is None:
                return
            self._cmd_pub.Write(self._low_cmd_msg)

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
            self._mode_machine = getattr(msg, 'mode_machine', 0)
            version = getattr(msg, 'version', None)
            if version is not None:
                self._version = (int(version[0]), int(version[1]))

        # Signal first message received (for connect() wait)
        self._state_received.set()
