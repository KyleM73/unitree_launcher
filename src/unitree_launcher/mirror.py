"""Mirror mode: display real robot state in MuJoCo viewer.

Provides ``RealtimeMirror`` (display-only MuJoCo model) and ``run_mirror()``
(the ``uv run mirror`` entry point). Subscribes to the real robot's DDS
state over Ethernet via ``MirrorRobot`` and renders in gui or viser.
"""
from __future__ import annotations

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
        """Set display model to match a RobotState and run forward kinematics."""
        for i in range(_N_DOF):
            self.data.qpos[self._qpos_addrs[i]] = state.joint_positions[i]
            self.data.qvel[self._dof_addrs[i]] = state.joint_velocities[i]

        fj = self._fj_addr
        self.data.qpos[fj + 0] = 0.0
        self.data.qpos[fj + 1] = 0.0
        self.data.qpos[fj + 2] = self._base_height
        self.data.qpos[fj + 3] = state.imu_quaternion[0]  # w
        self.data.qpos[fj + 4] = state.imu_quaternion[1]  # x
        self.data.qpos[fj + 5] = state.imu_quaternion[2]  # y
        self.data.qpos[fj + 6] = state.imu_quaternion[3]  # z

        mujoco.mj_forward(self.model, self.data)


def run_mirror(args) -> None:
    """Run the mirror viewer: subscribe to real robot DDS and display.

    Supports --gui (MuJoCo GLFW viewer) and --viser (web viewer).
    Called by ``uv run mirror``. Requires unitree_sdk2py (Python DDS) to
    read LowState_ from the real robot over Ethernet.
    """
    import logging
    import time

    from unitree_launcher.compat import patch_unitree_b2_import, patch_unitree_crc, patch_unitree_threading
    from unitree_launcher.robot.mirror_robot import MirrorRobot
    from unitree_launcher.config import load_config

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    logger = logging.getLogger("mirror")

    config = load_config(args.config)
    config.network.interface = args.interface
    config.network.domain_id = 0

    # Patches needed for Python DDS
    patch_unitree_b2_import()
    patch_unitree_threading()
    patch_unitree_crc()

    mirror_display = RealtimeMirror()
    robot = MirrorRobot(config)

    logger.info("Connecting to robot on interface %s", args.interface)
    robot.connect()
    logger.info("Connected")

    try:
        if args.gui:
            logger.info("Launching MuJoCo viewer")
            with mujoco.viewer.launch_passive(mirror_display.model, mirror_display.data) as viewer:
                while viewer.is_running():
                    state = robot.get_state()
                    mirror_display.update(state)
                    viewer.sync()
                    time.sleep(0.02)

        elif args.viser:
            from unitree_launcher.viz.viser_viewer import ViserViewer
            logger.info("Launching viser viewer on port %d", args.port)
            viser_viewer = ViserViewer(mirror_display.model, port=args.port)
            viser_viewer.setup()
            try:
                while True:
                    state = robot.get_state()
                    mirror_display.update(state)
                    viser_viewer.sync(mirror_display.data)
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass
            finally:
                viser_viewer.close()

    finally:
        robot.disconnect()
        logger.info("Mirror stopped")
