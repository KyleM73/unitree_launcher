"""MuJoCo simulation robot backend.

Implements the RobotInterface for the MuJoCo physics simulator with
DDS bridge for state publishing.

Threading model:
    Control loop thread:  get_state() -> policy -> send_command() -> step() -> sleep
    DDS publishing thread: periodically reads mj_data and publishes LowState_ (read-only)
    [Metal] MuJoCo viewer: runs in main thread, calls viewer.sync() via launch_passive
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from src.compat import RecurrentThread, resolve_network_interface, patch_unitree_threading
from src.config import (
    Config,
    G1_29DOF_MUJOCO_JOINTS,
    G1_23DOF_MUJOCO_JOINTS,
    _get_joints_for_variant,
)
from src.robot.base import RobotCommand, RobotInterface, RobotState

logger = logging.getLogger(__name__)

# DDS IDL motor count for unitree_hg messages
_NUM_MOTOR_IDL_HG = 35

# Asset root relative to this file
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


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

        # Non-controlled MuJoCo actuator indices (for damping in 23-DOF mode)
        controlled_set = set(self._cfg_to_mj.tolist())
        self._non_controlled_mj = np.array(
            [i for i in range(self._num_motor) if i not in controlled_set],
            dtype=np.intp,
        )

        # Sensor data layout: pos[0:nm], vel[nm:2*nm], torque[2*nm:3*nm], IMU, frame
        self._dim_motor_sensor = 3 * self._num_motor

        # Substeps per policy tick
        self._substeps = config.control.sim_frequency // config.control.policy_frequency

        # Thread safety lock (protects mj_data)
        self._lock = threading.Lock()

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
        """Read current state from MuJoCo sensor data."""
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
        """Apply impedance control law to MuJoCo ctrl array.

        For each controlled joint i (in config order):
            ctrl[mj_i] = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)

        Non-controlled joints (23-DOF mode) get passive damping:
            ctrl[mj_i] = -kd_damp * dq
        """
        sd = self._data.sensordata
        nm = self._num_motor

        for cfg_i in range(self._n_dof):
            mj_i = self._cfg_to_mj[cfg_i]
            q_actual = sd[mj_i]
            dq_actual = sd[nm + mj_i]
            self._data.ctrl[mj_i] = (
                cmd.joint_torques[cfg_i]
                + cmd.kp[cfg_i] * (cmd.joint_positions[cfg_i] - q_actual)
                + cmd.kd[cfg_i] * (cmd.joint_velocities[cfg_i] - dq_actual)
            )

        # Damp non-controlled joints
        kd_damp = self._config.control.kd_damp
        for mj_i in self._non_controlled_mj:
            dq = sd[nm + mj_i]
            self._data.ctrl[mj_i] = -kd_damp * dq

    def step(self) -> None:
        """Advance simulation by one policy timestep (multiple physics substeps)."""
        with self._lock:
            for _ in range(self._substeps):
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
