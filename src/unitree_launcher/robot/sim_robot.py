"""MuJoCo simulation robot backend.

Pure MuJoCo — no DDS, no unitree_sdk2py. For mirroring real robot state
into sim, see ``mirror_robot.py`` and ``scripts/mirror_real_robot.py``.

Threading model (when viewer is active):
    Main thread:           drain key queue -> handle_key() -> lock -> viewer.sync() -> unlock
    Control loop thread:   get_state() -> policy -> send_command() -> lock -> mj_step() -> unlock
    Viewer thread:         GLFW rendering + event loop -> key_callback enqueues to SimpleQueue

A threading.Lock (self._lock) protects mjData during mj_step() and viewer.sync().
The key_callback must never acquire the lock (fires on viewer thread -> deadlock risk).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

from unitree_launcher.config import (
    Config,
    G1_29DOF_MUJOCO_JOINTS,
    G1_23DOF_MUJOCO_JOINTS,
    _get_joints_for_variant,
)
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

# Asset root relative to this file
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets"


class SimRobot(RobotInterface):
    """MuJoCo simulation backend."""

    def __init__(self, config: Config):
        variant = config.robot.variant

        # Load MuJoCo scene
        variant_suffix = variant.split("_")[1]  # "29dof" or "23dof"
        scene_path = _ASSETS_DIR / "robots" / "g1" / f"scene_{variant_suffix}.xml"
        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data = mujoco.MjData(self._model)

        # Set physics timestep from config
        self._model.opt.timestep = 1.0 / config.control.sim_frequency

        # Joint mapping: config indices -> MuJoCo actuator indices
        self._cfg_joints = _get_joints_for_variant(variant)
        self._n_dof = len(self._cfg_joints)
        self._num_motor = self._model.nu  # Always 29 for G1

        mujoco_map = (
            G1_29DOF_MUJOCO_JOINTS if variant == "g1_29dof"
            else G1_23DOF_MUJOCO_JOINTS
        )

        # Build config-to-MuJoCo actuator index mapping
        self._cfg_to_mj = np.zeros(self._n_dof, dtype=np.intp)
        self._qpos_addr = np.zeros(self._n_dof, dtype=np.intp)
        self._dof_addr = np.zeros(self._n_dof, dtype=np.intp)

        for cfg_idx, cfg_name in enumerate(self._cfg_joints):
            mj_joint_name = mujoco_map[cfg_name]
            actuator_name = mj_joint_name.replace("_joint", "")
            mj_act_idx = mujoco.mj_name2id(
                self._model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if mj_act_idx == -1:
                raise RuntimeError(
                    f"Actuator '{actuator_name}' not found in MuJoCo model"
                )
            self._cfg_to_mj[cfg_idx] = mj_act_idx

            # Precompute qpos/qvel addresses for reset
            jnt_id = mujoco.mj_name2id(
                self._model, mujoco._enums.mjtObj.mjOBJ_JOINT, mj_joint_name
            )
            self._qpos_addr[cfg_idx] = self._model.jnt_qposadr[jnt_id]
            self._dof_addr[cfg_idx] = self._model.jnt_dofadr[jnt_id]

        # Substeps per policy tick
        self._substeps = config.control.sim_frequency // config.control.policy_frequency

        # Thread safety lock (protects mj_data during mj_step / viewer.sync).
        self._lock = threading.Lock()

        # Pending command for per-substep PD (set by send_command, applied in step)
        self._pending_cmd: Optional[RobotCommand] = None

        # Optional per-substep callback (e.g. for elastic band forces)
        self._substep_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]] = None

        # Save initial state for reset
        mujoco.mj_forward(self._model, self._data)
        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()

    # ---- RobotInterface implementation ----

    def connect(self) -> None:
        """No-op for SimRobot (pure MuJoCo, no external connections)."""
        pass

    def disconnect(self) -> None:
        """No-op for SimRobot."""
        pass

    def get_state(self) -> RobotState:
        """Read current state from MuJoCo.

        Joint positions/velocities are read directly from ``qpos``/``qvel``
        (current, post-integration — no sensor lag).  Torques from
        ``actuator_force``.  IMU and frame data from remaining sensors.

        Sensor layout (after removing joint sensors):
          [0:4]   framequat (imu_quat)
          [4:7]   gyro (imu_gyro)
          [7:10]  accelerometer (imu_acc)
          [10:13] framepos (frame_pos)
          [13:16] framelinvel (frame_vel)
          [16:20] framequat (secondary_imu_quat)
          [20:23] gyro (secondary_imu_gyro)
          [23:26] accelerometer (secondary_imu_acc)
        """
        sd = self._data.sensordata

        return RobotState(
            timestamp=self._data.time,
            joint_positions=self._data.qpos[self._qpos_addr].copy(),
            joint_velocities=self._data.qvel[self._dof_addr].copy(),
            joint_torques=self._data.actuator_force[self._cfg_to_mj].copy(),
            imu_quaternion=sd[0:4].copy(),
            imu_angular_velocity=sd[4:7].copy(),
            imu_linear_acceleration=sd[7:10].copy(),
            base_position=sd[10:13].copy(),
            base_velocity=sd[13:16].copy(),
        )

    def send_command(self, cmd: RobotCommand) -> None:
        """Store command for the next ``step()`` call.

        At the start of ``step()``, the command's target positions are
        written to ``data.ctrl`` and kp/kd are applied to the position
        actuator's gainprm/biasprm.  MuJoCo computes PD forces internally
        at each physics substep, reading the latest ``qpos``/``qvel``.
        """
        self._pending_cmd = cmd

    def _apply_cmd(self, cmd: RobotCommand) -> None:
        """Apply command through MuJoCo position actuators.

        Sets target positions via ``data.ctrl`` and updates per-actuator
        kp/kv (``gainprm``/``biasprm``) from the command's kp/kd arrays.
        MuJoCo computes the PD force internally at each physics substep:

            actuator_force = kp * (ctrl - q) - kv * dq

        Force is clamped by the actuator's ``forcerange`` (matching motor
        torque limits).  Torque feedforward is added via ``qfrc_applied``.

        Non-controlled joints (23-DOF mode) keep their XML-default kp/kv.
        """
        self._data.qfrc_applied[:] = 0.0

        for cfg_i in range(self._n_dof):
            mj_i = self._cfg_to_mj[cfg_i]

            # Set target position for the position actuator
            self._data.ctrl[mj_i] = float(cmd.joint_positions[cfg_i])

            # Update actuator gains: kp -> gainprm[0], -kp -> biasprm[1], -kd -> biasprm[2]
            kp = float(cmd.kp[cfg_i])
            kd = float(cmd.kd[cfg_i])
            self._model.actuator_gainprm[mj_i, 0] = kp
            self._model.actuator_biasprm[mj_i, 1] = -kp
            self._model.actuator_biasprm[mj_i, 2] = -kd

            # Feedforward torque via qfrc_applied (not subject to actuator forcerange)
            tau_ff = float(cmd.joint_torques[cfg_i])
            if tau_ff != 0.0:
                self._data.qfrc_applied[self._dof_addr[cfg_i]] = tau_ff

    def get_gravity_compensation(self) -> np.ndarray:
        """Get per-joint gravity compensation torques in config order.

        Uses MuJoCo's ``qfrc_bias`` (Coriolis + gravitational forces).
        At low velocities this is dominated by gravity compensation.
        """
        grav = np.zeros(self._n_dof, dtype=np.float64)
        for cfg_i in range(self._n_dof):
            dof_idx = self._dof_addr[cfg_i]
            grav[cfg_i] = self._data.qfrc_bias[dof_idx]
        return grav

    def set_substep_callback(
        self, fn: Callable[[mujoco.MjModel, mujoco.MjData], None] | None
    ) -> None:
        """Register a callback invoked before each physics substep.

        Use this to apply external forces (e.g. elastic band) without
        reaching into SimRobot internals.  Pass ``None`` to clear.
        """
        self._substep_callback = fn

    def step(self) -> None:
        """Advance simulation by one policy timestep (multiple physics substeps).

        Position actuators compute PD forces internally at each ``mj_step``
        call, reading the latest ``qpos``/``qvel``.  We only need to set
        ``ctrl`` and actuator gains once per policy step.

        If a substep callback is registered (via ``set_substep_callback``),
        it is called before each ``mj_step`` — used for external forces
        like the elastic band.
        """
        cmd = self._pending_cmd
        cb = self._substep_callback
        with self._lock:
            if cmd is not None:
                self._apply_cmd(cmd)
            for _ in range(self._substeps):
                if cb is not None:
                    cb(self._model, self._data)
                mujoco.mj_step(self._model, self._data)

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Reset MuJoCo to initial configuration or a specified state."""
        with self._lock:
            self._data.qpos[:] = self._initial_qpos
            self._data.qvel[:] = 0.0
            self._data.ctrl[:] = 0.0

            if initial_state is not None:
                # Map config joint positions/velocities to MuJoCo qpos/qvel
                for cfg_i in range(self._n_dof):
                    self._data.qpos[self._qpos_addr[cfg_i]] = (
                        initial_state.joint_positions[cfg_i]
                    )
                    self._data.qvel[self._dof_addr[cfg_i]] = (
                        initial_state.joint_velocities[cfg_i]
                    )

            self._data.time = 0.0
            mujoco.mj_forward(self._model, self._data)

    def set_home_positions(self, positions_native_order: np.ndarray) -> None:
        """Update stored initial qpos from config-order joint positions and reset.

        Call this after loading a policy to ensure the robot starts in the
        policy's expected default pose.
        """
        for cfg_i in range(self._n_dof):
            self._initial_qpos[self._qpos_addr[cfg_i]] = positions_native_order[cfg_i]
        self.reset()

    def set_armature(self, armature_native_order: np.ndarray) -> None:
        """Set per-joint armature values in the MuJoCo model.

        Armature adds to the diagonal of the mass matrix and must match
        the training simulator.  Values are in config/robot-native order.
        """
        for cfg_i in range(self._n_dof):
            dof_idx = self._dof_addr[cfg_i]
            self._model.dof_armature[dof_idx] = armature_native_order[cfg_i]

    def set_actuator_gains(self, kp: float | np.ndarray, kv: float | np.ndarray) -> None:
        """Set per-joint actuator kp/kv in the MuJoCo model.

        Updates the position actuator's gainprm/biasprm for each controlled
        joint.  This is used to match the training environment's gains when
        switching policies (e.g. BeyondMimic vs IsaacLab).

        Args:
            kp: Position gain — scalar or array in config order.
            kv: Velocity gain — scalar or array in config order.
        """
        kp_scalar = np.isscalar(kp)
        kv_scalar = np.isscalar(kv)
        for cfg_i in range(self._n_dof):
            mj_i = self._cfg_to_mj[cfg_i]
            _kp = float(kp) if kp_scalar else float(kp[cfg_i])
            _kv = float(kv) if kv_scalar else float(kv[cfg_i])
            self._model.actuator_gainprm[mj_i, 0] = _kp
            self._model.actuator_biasprm[mj_i, 1] = -_kp
            self._model.actuator_biasprm[mj_i, 2] = -_kv

    def set_reference_pose(
        self,
        joint_positions_policy: np.ndarray,
        pelvis_pos: np.ndarray,
        pelvis_quat_wxyz: np.ndarray,
        joint_mapper: 'JointMapper',
    ) -> None:
        """Reset robot to match the reference trajectory's starting pose.

        In training, the environment teleports the robot to the reference
        motion's pose at each reset.  This method replicates that behavior
        for MuJoCo, setting both the root body (freejoint) and joint positions.

        Args:
            joint_positions_policy: Reference joint positions in policy order.
            pelvis_pos: Reference pelvis world position ``(3,)``.
            pelvis_quat_wxyz: Reference pelvis quaternion ``(4,)`` in wxyz.
            joint_mapper: Maps policy order to robot-native order.
        """
        joint_native = joint_mapper.policy_to_robot(joint_positions_policy)
        with self._lock:
            # Set root body (freejoint): [tx, ty, tz, qw, qx, qy, qz]
            self._data.qpos[0:3] = pelvis_pos
            self._data.qpos[3:7] = pelvis_quat_wxyz

            # Set joint positions
            for cfg_i in range(self._n_dof):
                self._data.qpos[self._qpos_addr[cfg_i]] = joint_native[cfg_i]

            # Zero velocities and controls
            self._data.qvel[:] = 0.0
            self._data.ctrl[:] = 0.0
            self._data.time = 0.0
            mujoco.mj_forward(self._model, self._data)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    # ---- Viewer integration ----

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Access MuJoCo model (needed by viewer)."""
        return self._model

    @property
    def mj_data(self) -> mujoco.MjData:
        """Access MuJoCo data (needed by viewer)."""
        return self._data

    @property
    def lock(self) -> threading.Lock:
        """Access the data lock (needed by viewer for thread-safe sync)."""
        return self._lock

    def get_body_state(self, body_name: str) -> tuple:
        """Get a named body's world position and quaternion from MuJoCo.

        Returns:
            (position(3,), quaternion_wxyz(4,))
        """
        body_id = mujoco.mj_name2id(
            self._model, mujoco._enums.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
        return (
            self._data.xpos[body_id].copy(),
            self._data.xquat[body_id].copy(),
        )

