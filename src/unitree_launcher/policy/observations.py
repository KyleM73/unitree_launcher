"""Observation vector construction for IsaacLab policies.

Builds the concatenated observation vector in IsaacLab PolicyCfg order:
    [base_lin_vel (3)?, base_ang_vel (3), projected_gravity (3),
     velocity_commands (3), joint_pos (n_obs), joint_vel (n_obs),
     actions (n_ctrl)]

base_lin_vel is omitted entirely when use_estimator=False.
"""
from __future__ import annotations

import numpy as np

from unitree_launcher.config import Config, Q_HOME_23DOF, Q_HOME_29DOF
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotState


class ObservationBuilder:
    """Builds observation vectors for IsaacLab-style policies.

    Args:
        joint_mapper: Handles joint ordering / subsetting.
        config: Full config (needs robot.variant and control.q_home).
        use_estimator: If False, omit base_lin_vel from observation.
    """

    def __init__(
        self,
        joint_mapper: JointMapper,
        config: Config,
        use_estimator: bool = True,
    ):
        self._mapper = joint_mapper
        self._use_estimator = use_estimator

        # Build q_home array in observation order
        q_home_dict = config.control.q_home
        if q_home_dict is None:
            q_home_dict = (
                Q_HOME_29DOF if config.robot.variant == "g1_29dof" else Q_HOME_23DOF
            )
        self._q_home_obs = np.array(
            [q_home_dict[j] for j in joint_mapper.observed_joints]
        )

    @property
    def observation_dim(self) -> int:
        """Total observation dimension."""
        base = 2 * self._mapper.n_observed + self._mapper.n_controlled
        return base + (12 if self._use_estimator else 9)

    def build(
        self,
        robot_state: RobotState,
        last_action: np.ndarray,
        velocity_command: np.ndarray,
    ) -> np.ndarray:
        """Build observation vector in IsaacLab PolicyCfg order.

        Args:
            robot_state: Current robot state.
            last_action: Previous policy output, shape ``(n_controlled,)``.
            velocity_command: User velocity command ``[vx, vy, yaw_rate]``.

        Returns:
            Observation vector of shape ``(observation_dim,)``.
        """
        parts = []

        # 1. Base linear velocity in body frame (optional)
        if self._use_estimator:
            body_lin_vel = self.compute_body_velocity_in_body_frame(
                robot_state.base_velocity, robot_state.imu_quaternion
            )
            parts.append(body_lin_vel)

        # 2. Base angular velocity (already in body frame from IMU gyro)
        parts.append(robot_state.imu_angular_velocity)

        # 3. Projected gravity
        parts.append(self.compute_projected_gravity(robot_state.imu_quaternion))

        # 4. Velocity commands
        parts.append(velocity_command)

        # 5. Joint positions relative to home
        joint_pos = self._mapper.robot_to_observation(robot_state.joint_positions)
        parts.append(joint_pos - self._q_home_obs)

        # 6. Joint velocities
        parts.append(
            self._mapper.robot_to_observation(robot_state.joint_velocities)
        )

        # 7. Last actions
        parts.append(last_action)

        return np.concatenate(parts)

    def compute_projected_gravity(self, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Rotate [0, 0, -1] by inverse of IMU quaternion.

        The IMU quaternion rotates body->world. Its transpose (inverse for a
        rotation matrix) rotates world->body, which is what we need to express
        the world-frame gravity vector in the body frame.

        Returns:
            Shape ``(3,)`` gravity vector in body frame.
        """
        R = _quat_to_rotation_matrix(quaternion_wxyz)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def compute_body_velocity_in_body_frame(
        self, world_velocity: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> np.ndarray:
        """Transform world-frame velocity to body frame using IMU quaternion.

        Returns:
            Shape ``(3,)`` velocity in body frame.
        """
        R = _quat_to_rotation_matrix(quaternion_wxyz)
        return R.T @ world_velocity


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix (body->world)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
