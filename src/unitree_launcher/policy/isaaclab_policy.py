"""IsaacLab velocity-tracking policy.

Self-contained: owns observation building, ONNX inference, action smoothing,
control law, and gain management. The Runtime calls ``step(state, vel_cmd)``
and receives a complete ``RobotCommand``.

Observation format (IsaacLab PolicyCfg order):
    [base_lin_vel (3)?, base_ang_vel (3), projected_gravity (3),
     velocity_commands (3), joint_pos_rel (n), joint_vel (n), actions (n)]

Control law:
    target_pos = q_home + Ka * action
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnxruntime as ort

from unitree_launcher.config import (
    BM_ACTION_SCALE_29DOF,
    Config,
    ISAACLAB_KD_29DOF,
    ISAACLAB_KP_29DOF,
    Q_HOME_23DOF,
    Q_HOME_29DOF,
)
from unitree_launcher.policy.base import Policy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotState


class IsaacLabPolicy(Policy):
    """ONNX-based IsaacLab locomotion policy.

    Args:
        mapper: Maps between robot-native and policy joint orderings.
        config: Full config for robot variant, gains, home positions.
        use_estimator: Include base_lin_vel in observations.
    """

    def __init__(
        self,
        mapper: JointMapper,
        config: Config,
        use_estimator: bool = True,
    ) -> None:
        super().__init__(mapper, n_dof=mapper.n_robot)
        self._use_estimator = use_estimator

        variant = config.robot.variant
        policy_joints = mapper.policy_joints

        # Per-joint gains from IsaacLab training (armature formula)
        self._kp = np.array(
            [ISAACLAB_KP_29DOF.get(j, 40.0) for j in policy_joints],
            dtype=np.float64,
        )
        self._kd = np.array(
            [ISAACLAB_KD_29DOF.get(j, 2.0) for j in policy_joints],
            dtype=np.float64,
        )

        # Action scale: per-joint Ka (0.25 * effort_limit / kp)
        self._action_scale = np.array(
            [BM_ACTION_SCALE_29DOF.get(j, 0.439) for j in policy_joints],
            dtype=np.float64,
        )

        # Home positions in policy joint order
        q_home_dict = config.control.q_home
        if q_home_dict is None:
            q_home_dict = Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
        self._q_home = np.array(
            [q_home_dict[j] for j in policy_joints], dtype=np.float64
        )

        # Full-robot home positions (for template filling)
        robot_joints = mapper.robot_joints
        self._default_pos_robot = np.array(
            [q_home_dict[j] for j in robot_joints], dtype=np.float64
        )
        self._default_pos_policy = self._q_home.copy()

        # Damping for non-controlled joints
        self._kd_damp = config.control.kd_damp

        # Pre-compute full-robot gain arrays
        self._kp_robot = mapper.fit_gains(self._kp, default=0.0)
        self._kd_robot = mapper.fit_gains(self._kd, default=self._kd_damp)

        # Observation dimension (computed from structure)
        n = mapper.n_policy
        self._obs_dim = (12 if use_estimator else 9) + 2 * n + n

    def load(self, path: str) -> None:
        """Load ONNX model and validate dimensions."""
        try:
            session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ValueError(f"Failed to load ONNX model from {path}: {exc}") from exc

        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"Expected 1 input, got {len(inputs)}: "
                f"{[i.name for i in inputs]}"
            )
        model_obs_dim = inputs[0].shape[1]
        if model_obs_dim != self._obs_dim:
            raise ValueError(
                f"ONNX obs dim {model_obs_dim} != expected {self._obs_dim}"
            )

        outputs = session.get_outputs()
        model_action_dim = outputs[0].shape[1]
        if model_action_dim != self._mapper.n_policy:
            raise ValueError(
                f"ONNX action dim {model_action_dim} != "
                f"n_policy {self._mapper.n_policy}"
            )

        self._session = session

        # Apply ONNX metadata overrides (gains, action scale)
        self._apply_metadata(session)

        # Recompute robot-order gains after metadata overrides
        self._kp_robot = self._mapper.fit_gains(self._kp, default=0.0)
        self._kd_robot = self._mapper.fit_gains(self._kd, default=self._kd_damp)

        self.reset()

    def step(self, state: RobotState, velocity_command: np.ndarray) -> RobotCommand:
        """Build observation, run inference, apply control law, return command."""
        # 1. Build observation
        obs = self._build_observation(state, velocity_command)
        self._last_observation = obs

        # 2. Inference
        raw_action = self._run_inference(obs)

        # 3. Smooth
        action = self._smooth_action(raw_action)

        # 4. Control law: target_q = q_home + Ka * action
        #    (Ka already applied inside _smooth_action via _action_scale)
        target_q_policy = self._q_home + action
        target_q_robot = self._mapper.policy_to_robot(
            target_q_policy, template=self._default_pos_robot
        )

        # 5. Build command
        return self._build_command(
            state, target_q_robot, self._kp_robot.copy(), self._kd_robot.copy()
        )

    def _apply_metadata(self, session: ort.InferenceSession) -> None:
        """Override gains/scale from ONNX metadata if present."""
        md = dict(session.get_modelmeta().custom_metadata_map)
        if not md:
            return

        def _parse_csv(s):
            return np.array([float(x) for x in s.split(",")], dtype=np.float64)

        if "joint_stiffness" in md:
            self._kp = _parse_csv(md["joint_stiffness"])
        if "joint_damping" in md:
            self._kd = _parse_csv(md["joint_damping"])
        if "action_scale" in md:
            self._action_scale = _parse_csv(md["action_scale"])

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    def _build_observation(
        self, state: RobotState, velocity_command: np.ndarray
    ) -> np.ndarray:
        """Build observation vector in IsaacLab PolicyCfg order."""
        parts = []

        R = _quat_to_rotation_matrix(state.imu_quaternion)

        # 1. Base linear velocity in body frame (optional)
        if self._use_estimator:
            body_lin_vel = R.T @ state.base_velocity
            parts.append(body_lin_vel)

        # 2. Base angular velocity (already body frame from IMU gyro)
        parts.append(state.imu_angular_velocity)

        # 3. Projected gravity
        parts.append(R.T @ np.array([0.0, 0.0, -1.0]))

        # 4. Velocity commands
        parts.append(velocity_command)

        # 5. Joint positions relative to home
        joint_pos = self._mapper.robot_to_policy(state.joint_positions)
        parts.append(joint_pos - self._q_home)

        # 6. Joint velocities
        parts.append(self._mapper.robot_to_policy(state.joint_velocities))

        # 7. Last actions (unscaled, in policy order)
        parts.append(self._last_action)

        return np.concatenate(parts)


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix (body->world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
