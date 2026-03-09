"""Mirror mode: display real robot state in MuJoCo viewer.

Provides ``RealtimeMirror`` (display-only MuJoCo model) and ``run_mirror()``
(the ``uv run mirror`` entry point).

Two transport modes:
- **DDS** (Ethernet): uses ``MirrorRobot`` to subscribe to ``rt/lowstate``
- **TCP** (WiFi, ``--peer``): connects to the mirror bridge TCP server on the robot
"""
from __future__ import annotations

import json
import socket
import struct
from pathlib import Path

import mujoco
import numpy as np

from unitree_launcher.config import G1_29DOF_JOINTS, G1_29DOF_MUJOCO_JOINTS
from unitree_launcher.robot.base import RobotState

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_N_DOF = 29


class RealtimeMirror:
    """Display-only MuJoCo model that mirrors robot state."""

    def __init__(self, base_height: float = 0.793):
        """
        Args:
            base_height: Z position for the floating base (default: standing height).
        """
        scene_xml = _ASSETS_DIR / "robots" / "g1" / "scene_29dof.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self._base_height = base_height

        # Build joint mapping: config index -> MuJoCo qpos/qvel addr
        self._qpos_addrs = np.zeros(_N_DOF, dtype=np.intp)
        self._dof_addrs = np.zeros(_N_DOF, dtype=np.intp)
        for cfg_idx, cfg_name in enumerate(G1_29DOF_JOINTS):
            mj_joint_name = G1_29DOF_MUJOCO_JOINTS[cfg_name]
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco._enums.mjtObj.mjOBJ_JOINT, mj_joint_name
            )
            self._qpos_addrs[cfg_idx] = self.model.jnt_qposadr[jnt_id]
            self._dof_addrs[cfg_idx] = self.model.jnt_dofadr[jnt_id]

        # Freejoint address
        fj_id = mujoco.mj_name2id(
            self.model, mujoco._enums.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        self._fj_addr = self.model.jnt_qposadr[fj_id]

    def update(self, state: RobotState) -> None:
        """Set display model to match a RobotState and run forward kinematics.

        Uses base_position from state if available (non-NaN, e.g. from sim
        logs or estimator). Falls back to fixed height for mirror mode where
        base position is unknown.
        """
        for i in range(_N_DOF):
            self.data.qpos[self._qpos_addrs[i]] = state.joint_positions[i]
            self.data.qvel[self._dof_addrs[i]] = state.joint_velocities[i]

        fj = self._fj_addr
        bp = state.base_position
        if np.isfinite(bp[0]):
            self.data.qpos[fj + 0] = bp[0]
            self.data.qpos[fj + 1] = bp[1]
            self.data.qpos[fj + 2] = bp[2]
        else:
            self.data.qpos[fj + 0] = 0.0
            self.data.qpos[fj + 1] = 0.0
            self.data.qpos[fj + 2] = self._base_height
        self.data.qpos[fj + 3] = state.imu_quaternion[0]  # w
        self.data.qpos[fj + 4] = state.imu_quaternion[1]  # x
        self.data.qpos[fj + 5] = state.imu_quaternion[2]  # y
        self.data.qpos[fj + 6] = state.imu_quaternion[3]  # z

        mujoco.mj_forward(self.model, self.data)


class TcpStateReader:
    """Read robot state from the mirror bridge TCP server."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._buf = b""

    def connect(self, timeout: float = 5.0) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((self._host, self._port))
        self._sock.settimeout(0.1)

    def disconnect(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def get_state(self) -> RobotState | None:
        """Read the next state frame. Returns None if no data available."""
        try:
            chunk = self._sock.recv(8192)
            if not chunk:
                return None
            self._buf += chunk
        except socket.timeout:
            pass
        except OSError:
            return None

        # Parse length-prefixed frames, keep only the latest
        state = None
        while len(self._buf) >= 4:
            msg_len = struct.unpack(">I", self._buf[:4])[0]
            if len(self._buf) < 4 + msg_len:
                break
            payload = json.loads(self._buf[4:4 + msg_len])
            self._buf = self._buf[4 + msg_len:]

            n = min(len(payload["q"]), _N_DOF)
            state = RobotState(
                timestamp=0.0,
                joint_positions=np.array(payload["q"][:n], dtype=np.float64),
                joint_velocities=np.array(payload["dq"][:n], dtype=np.float64),
                joint_torques=np.array(payload["tau"][:n], dtype=np.float64),
                imu_quaternion=np.array(payload["quat"][:4], dtype=np.float64),
                imu_angular_velocity=np.array(payload["gyro"][:3], dtype=np.float64),
                imu_linear_acceleration=np.array(payload["accel"][:3], dtype=np.float64),
                base_position=np.full(3, np.nan),
                base_velocity=np.full(3, np.nan),
            )

        return state


def run_mirror(args) -> None:
    """Run the mirror viewer: display real robot state.

    Two transport modes:
    - ``--peer HOST``: TCP connection to mirror bridge on the robot (WiFi)
    - No ``--peer``: DDS subscription via MirrorRobot (Ethernet)
    """
    import logging
    import time

    from unitree_launcher.mirror_bridge import MIRROR_PORT

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    logger = logging.getLogger("mirror")

    mirror_display = RealtimeMirror()
    peer = getattr(args, 'peer', None)

    if peer:
        # TCP mode: connect to mirror bridge on the robot
        reader = TcpStateReader(peer, MIRROR_PORT)
        logger.info("Connecting to mirror bridge at %s:%d", peer, MIRROR_PORT)
        reader.connect()
        logger.info("Connected (TCP)")

        def get_state():
            return reader.get_state()

        cleanup = reader.disconnect
    else:
        # DDS mode: subscribe via MirrorRobot (Ethernet)
        from unitree_launcher.compat import patch_unitree_b2_import, patch_unitree_crc, patch_unitree_threading
        from unitree_launcher.robot.mirror_robot import MirrorRobot
        from unitree_launcher.config import load_config

        config = load_config(args.config)
        config.network.interface = args.interface
        config.network.domain_id = 0

        patch_unitree_b2_import()
        patch_unitree_threading()
        patch_unitree_crc()

        robot = MirrorRobot(config)
        logger.info("Connecting to robot on interface %s (DDS)", args.interface)
        robot.connect()
        logger.info("Connected (DDS)")

        def get_state():
            return robot.get_state()

        cleanup = robot.disconnect

    last_state = None

    try:
        if args.gui:
            logger.info("Launching MuJoCo viewer")
            with mujoco.viewer.launch_passive(mirror_display.model, mirror_display.data) as viewer:
                while viewer.is_running():
                    state = get_state()
                    if state is not None:
                        last_state = state
                    if last_state is not None:
                        mirror_display.update(last_state)
                    viewer.sync()
                    time.sleep(0.02)

        elif args.viser:
            from unitree_launcher.viz.viser_viewer import ViserViewer
            logger.info("Launching viser viewer on port %d", args.port)
            viser_viewer = ViserViewer(mirror_display.model, port=args.port)
            viser_viewer.setup()
            try:
                while True:
                    state = get_state()
                    if state is not None:
                        last_state = state
                    if last_state is not None:
                        mirror_display.update(last_state)
                    viser_viewer.sync(mirror_display.data)
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass
            finally:
                viser_viewer.close()

    finally:
        cleanup()
        logger.info("Mirror stopped")
