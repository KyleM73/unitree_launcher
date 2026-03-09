"""Mirror bridge: publish robot state over TCP for remote visualization.

When running on the robot, the C++ backend reads motor state on eth0.
This bridge reads state from the C++ backend and serves it over a TCP
socket so that a remote mirror client can visualize the robot over WiFi.

No DDS Python packages required on the robot — uses plain TCP sockets.

Protocol: each message is a 4-byte big-endian length prefix followed by
JSON-encoded state (joint positions, velocities, torques, IMU data).
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from typing import Optional, List

logger = logging.getLogger(__name__)

MIRROR_PORT = 9870


class MirrorBridge:
    """Serve robot state over TCP for remote mirror clients."""

    def __init__(self, port: int = MIRROR_PORT):
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._robot = None
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()

    def start(self, robot) -> None:
        """Start the bridge server in a background daemon thread."""
        self._robot = robot
        self._thread = threading.Thread(
            target=self._run, name="mirror-bridge", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the bridge to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _accept_clients(self, server: socket.socket) -> None:
        """Accept loop running in a separate daemon thread."""
        server.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                client, addr = server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._clients_lock:
                    self._clients.append(client)
                logger.info("Mirror client connected: %s:%d", *addr)
            except socket.timeout:
                continue
            except OSError:
                break

    def _run(self) -> None:
        """Bridge thread: read C++ state, broadcast to TCP clients."""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", self._port))
            server.listen(4)

            accept_thread = threading.Thread(
                target=self._accept_clients, args=(server,),
                name="mirror-accept", daemon=True,
            )
            accept_thread.start()

            logger.info("Mirror bridge: serving on port %d", self._port)

            dt = 0.02  # 50 Hz

            while not self._stop_event.is_set():
                if self._robot._controller is None:
                    self._stop_event.wait(timeout=0.1)
                    continue

                try:
                    state = self._robot._controller.get_robot_state()
                    ms = state.motor_state
                    n = min(35, len(ms.q))
                    payload = {
                        "q": [float(ms.q[i]) for i in range(n)],
                        "dq": [float(ms.dq[i]) for i in range(n)],
                        "tau": [float(ms.tau_est[i]) for i in range(n)],
                        "quat": list(state.imu_state.quaternion[:4]),
                        "gyro": list(state.imu_state.gyroscope[:3]),
                        "accel": list(state.imu_state.accelerometer[:3]),
                    }
                    data = json.dumps(payload).encode()
                    frame = struct.pack(">I", len(data)) + data

                    with self._clients_lock:
                        dead = []
                        for client in self._clients:
                            try:
                                client.sendall(frame)
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                dead.append(client)
                        for client in dead:
                            self._clients.remove(client)
                            client.close()
                except Exception:
                    pass  # Robot may be shutting down

                self._stop_event.wait(timeout=dt)

        except Exception:
            logger.exception("Mirror bridge failed to start")
        finally:
            with self._clients_lock:
                for client in self._clients:
                    client.close()
                self._clients.clear()
            try:
                server.close()
            except Exception:
                pass
