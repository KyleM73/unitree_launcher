#!/usr/bin/env python3
"""Live MuJoCo mirror of the Unitree G1 real robot.

Subscribes to the robot's LowState_ over DDS (Ethernet) and updates a
MuJoCo visualization to match the real joint positions and IMU orientation.
No physics simulation — MuJoCo is used purely for display.

Usage:
    uv run mirror --gui --interface en8

    --interface: Network interface connected to the robot (default: "en8").
                 Use "auto" for loopback (testing).

This standalone script is an alternative to ``uv run mirror``.
Run via: uv run python scripts/mirror_real_robot.py --interface en8
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import mujoco.viewer
import numpy as np

from unitree_launcher.compat import patch_unitree_b2_import, patch_unitree_threading, resolve_network_interface
patch_unitree_b2_import()
from unitree_launcher.mirror import RealtimeMirror
from unitree_launcher.robot.base import RobotState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mirror")

_N_DOF = 29
_CONNECT_TIMEOUT_S = 10.0


class RobotMirror:
    """Subscribes to DDS LowState_ and mirrors into a display-only MuJoCo model."""

    def __init__(self, interface: str, record_path: str | None = None):
        self._interface = interface
        self._record_path = record_path
        self._mirror = RealtimeMirror()

        # Latest state from DDS (protected by lock)
        self._lock = threading.Lock()
        self._latest_state = RobotState.zeros(_N_DOF)
        self._state_received = threading.Event()
        self._state_count = 0

        self._shutdown = threading.Event()

    def connect(self):
        """Initialize DDS subscriber and wait for first state."""
        patch_unitree_threading()

        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        iface = resolve_network_interface(self._interface)
        logger.info("Initializing DDS on interface '%s' (domain_id=0)", iface)
        ChannelFactoryInitialize(0, iface)

        self._state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_sub.Init(handler=self._on_low_state, queueLen=1)

        logger.info("Waiting for first LowState_ message (timeout=%ds)...", _CONNECT_TIMEOUT_S)
        if not self._state_received.wait(timeout=_CONNECT_TIMEOUT_S):
            raise TimeoutError(
                f"No state received within {_CONNECT_TIMEOUT_S}s. "
                "Is the robot powered on and connected?"
            )
        logger.info("Connected — receiving state from robot.")

    def _on_low_state(self, msg) -> None:
        """DDS callback: extract joint positions, velocities, and IMU."""
        state = RobotState(
            timestamp=time.time(),
            joint_positions=np.array(
                [msg.motor_state[i].q for i in range(_N_DOF)], dtype=np.float64,
            ),
            joint_velocities=np.array(
                [msg.motor_state[i].dq for i in range(_N_DOF)], dtype=np.float64,
            ),
            joint_torques=np.array(
                [msg.motor_state[i].tau_est for i in range(_N_DOF)], dtype=np.float64,
            ),
            imu_quaternion=np.array(
                [msg.imu_state.quaternion[j] for j in range(4)], dtype=np.float64,
            ),
            imu_angular_velocity=np.array(
                [msg.imu_state.gyroscope[j] for j in range(3)], dtype=np.float64,
            ),
            imu_linear_acceleration=np.array(
                [msg.imu_state.accelerometer[j] for j in range(3)], dtype=np.float64,
            ),
            base_position=np.full(3, np.nan),
            base_velocity=np.full(3, np.nan),
        )

        with self._lock:
            self._latest_state = state
            self._state_count += 1

        self._state_received.set()

    def run(self):
        """Launch the MuJoCo viewer and update loop."""
        self.connect()

        # Video recorder (uses the mirror's display-only model).
        recorder = None
        if self._record_path:
            from unitree_launcher.recording import VideoRecorder, normalize_record_path
            recorder = VideoRecorder(
                normalize_record_path(self._record_path),
                self._mirror.model, self._mirror.data,
            )

        logger.info("Launching MuJoCo viewer...")
        with mujoco.viewer.launch_passive(self._mirror.model, self._mirror.data) as viewer:
            logger.info("Viewer running. Close the window or press Ctrl+C to exit.")

            last_count = 0
            try:
                while viewer.is_running() and not self._shutdown.is_set():
                    with self._lock:
                        state = self._latest_state.copy()
                        count = self._state_count

                    self._mirror.update(state)
                    viewer.sync()
                    # Capture after sync — the DDS callback runs on its own
                    # thread so this never blocks incoming state updates.
                    if recorder:
                        recorder.capture()

                    if count >= last_count + 500:
                        logger.info("Received %d state messages", count)
                        last_count = count

                    time.sleep(1 / 100)
            finally:
                if recorder:
                    recorder.close()

        logger.info("Viewer closed.")

    def shutdown(self):
        self._shutdown.set()


def main():
    parser = argparse.ArgumentParser(
        description="Mirror the Unitree G1 real robot in MuJoCo (display only)."
    )
    parser.add_argument(
        "--interface",
        default="en8",
        help="Network interface connected to the robot (default: en8 — USB Ethernet on macOS). "
             "Use 'auto' for loopback.",
    )
    parser.add_argument(
        "--record", nargs="?", const="recording.mp4", default=None,
        metavar="PATH",
        help="Record video to MP4 (default: recording.mp4)",
    )
    args = parser.parse_args()

    mirror = RobotMirror(interface=args.interface, record_path=args.record)

    def signal_handler(sig, frame):
        logger.info("Caught signal %d, shutting down...", sig)
        mirror.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        mirror.run()
    except TimeoutError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")


if __name__ == "__main__":
    main()
