"""MuJoCo simulation robot backend.

Implements the RobotInterface for the MuJoCo physics simulator with
DDS bridge for state publishing.

Threading model (when viewer is active):
    Main thread:           drain key queue -> handle_key() -> lock -> viewer.sync() -> unlock
    Control loop thread:   get_state() -> policy -> send_command() -> lock -> mj_step() -> unlock
    Viewer thread:         GLFW rendering + event loop -> key_callback enqueues to SimpleQueue
    DDS publishing thread: lock -> snapshot sensor data -> unlock -> publish LowState_

A threading.Lock (self._lock) protects mjData during mj_step() and viewer.sync().
The key_callback must never acquire the lock (fires on viewer thread -> deadlock risk).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

from unitree_launcher.compat import RecurrentThread, resolve_network_interface, patch_unitree_threading
from unitree_launcher.config import (
    Config,
    G1_29DOF_MUJOCO_JOINTS,
    G1_23DOF_MUJOCO_JOINTS,
    _get_joints_for_variant,
)
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

logger = logging.getLogger(__name__)

# DDS IDL motor count for unitree_hg messages
_NUM_MOTOR_IDL_HG = 35

# Asset root relative to this file
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets"


class SimRobot(RobotInterface):
    """MuJoCo simulation backend with optional DDS bridge."""

    def __init__(self, config: Config):
        self._config = config
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

        # Non-controlled MuJoCo actuator indices (for 23-DOF mode)
        controlled_set = set(self._cfg_to_mj.tolist())
        self._non_controlled_mj = np.array(
            [i for i in range(self._num_motor) if i not in controlled_set],
            dtype=np.intp,
        )

        # Sensor data layout: pos[0:nm], vel[nm:2*nm], torque[2*nm:3*nm], IMU, frame
        self._dim_motor_sensor = 3 * self._num_motor

        # Substeps per policy tick
        self._substeps = config.control.sim_frequency // config.control.policy_frequency

        # Thread safety lock (protects mj_data during mj_step / viewer.sync).
        self._lock = threading.Lock()

        # Pending command for per-substep PD (set by send_command, applied in step)
        self._pending_cmd: Optional[RobotCommand] = None

        # Optional per-substep callback (e.g. for elastic band forces)
        self._substep_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]] = None

        # DDS state (lazy init in connect())
        self._dds_initialized = False
        self._state_pub_thread: Optional[RecurrentThread] = None
        self._low_state_pub = None
        self._low_state_msg = None

        # Save initial state for reset
        mujoco.mj_forward(self._model, self._data)
        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()

    # ---- RobotInterface implementation ----

    def connect(self) -> None:
        """Initialize DDS and start state publishing thread."""
        if self._dds_initialized:
            return

        patch_unitree_threading()

        from unitree_sdk2py.core.channel import (
            ChannelPublisher,
            ChannelFactoryInitialize,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

        iface = resolve_network_interface(self._config.network.interface)
        ChannelFactoryInitialize(self._config.network.domain_id, iface)

        self._low_state_msg = unitree_hg_msg_dds__LowState_()
        self._low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
        self._low_state_pub.Init()

        self._state_pub_thread = RecurrentThread(
            interval=self._model.opt.timestep,
            target=self._publish_low_state,
            name="sim_lowstate",
        )
        self._state_pub_thread.Start()

        self._dds_initialized = True
        logger.info("SimRobot DDS connected on interface %s", iface)

    def disconnect(self) -> None:
        """Stop DDS publishing thread."""
        if self._state_pub_thread is not None:
            self._state_pub_thread.Shutdown()
            self._state_pub_thread = None
        self._dds_initialized = False
        logger.info("SimRobot disconnected")

    def get_state(self) -> RobotState:
        """Read current state from MuJoCo sensor data.

        Note: sensordata is computed during ``mj_step1`` (before integration),
        so it lags ``qpos``/``qvel`` by one substep.  This is acceptable for
        observation building (matches real sensor latency), but NOT for the
        PD control law — see ``_apply_pd()`` which reads ``qpos``/``qvel``.
        """
        sd = self._data.sensordata
        mj = self._cfg_to_mj
        nm = self._num_motor
        dms = self._dim_motor_sensor

        return RobotState(
            timestamp=self._data.time,
            joint_positions=sd[mj].copy(),
            joint_velocities=sd[nm + mj].copy(),
            joint_torques=sd[2 * nm + mj].copy(),
            imu_quaternion=sd[dms + 0 : dms + 4].copy(),
            imu_angular_velocity=sd[dms + 4 : dms + 7].copy(),
            imu_linear_acceleration=sd[dms + 7 : dms + 10].copy(),
            base_position=sd[dms + 10 : dms + 13].copy(),
            base_velocity=sd[dms + 13 : dms + 16].copy(),
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
        joint_native = joint_mapper.action_to_robot(joint_positions_policy)
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

    # ---- Metal-specific: viewer integration ----

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

    # ---- Private ----

    def _publish_low_state(self) -> None:
        """Publish LowState_ DDS message with current sensor data."""
        if self._low_state_msg is None or self._low_state_pub is None:
            return

        with self._lock:
            sd = self._data.sensordata
            nm = self._num_motor
            dms = self._dim_motor_sensor

            for i in range(nm):
                self._low_state_msg.motor_state[i].q = float(sd[i])
                self._low_state_msg.motor_state[i].dq = float(sd[nm + i])
                self._low_state_msg.motor_state[i].tau_est = float(sd[2 * nm + i])

            # IMU
            for j in range(4):
                self._low_state_msg.imu_state.quaternion[j] = float(sd[dms + j])
            for j in range(3):
                self._low_state_msg.imu_state.gyroscope[j] = float(sd[dms + 4 + j])
            for j in range(3):
                self._low_state_msg.imu_state.accelerometer[j] = float(
                    sd[dms + 7 + j]
                )

        self._low_state_pub.Write(self._low_state_msg)
