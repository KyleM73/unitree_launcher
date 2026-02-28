"""Display-only MuJoCo model that mirrors robot state.

Loads a separate MuJoCo scene (no physics) and updates joint positions,
velocities, and base orientation from a RobotState. Used by scripts that
need to visualize the real robot's state in MuJoCo.
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
