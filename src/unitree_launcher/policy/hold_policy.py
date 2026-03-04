"""Static PD hold policy — no neural network.

Holds the robot at its home pose using MTC StandbyController gains.
Used as fallback when no default stance policy is loaded.
"""
from __future__ import annotations

import numpy as np

from unitree_launcher.config import (
    Config,
    Q_HOME_23DOF,
    Q_HOME_29DOF,
    STANDBY_KD_29DOF,
    STANDBY_KP_29DOF,
)
from unitree_launcher.policy.base import Policy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotState


class HoldPolicy(Policy):
    """Static PD hold at home pose. No neural network.

    Uses MTC StandbyController gains (Kp: 150-350 legs, 40 arms;
    Kd: 5 legs, 10 knee, 3 arms).

    Args:
        mapper: Joint mapper for this robot.
        config: Full config for robot variant and home positions.
    """

    def __init__(self, mapper: JointMapper, config: Config) -> None:
        super().__init__(mapper, n_dof=mapper.n_robot)

        variant = config.robot.variant
        policy_joints = mapper.policy_joints
        robot_joints = mapper.robot_joints

        # MTC Standby gains
        self._kp = np.array(
            [STANDBY_KP_29DOF.get(j, 100.0) for j in policy_joints],
            dtype=np.float64,
        )
        self._kd = np.array(
            [STANDBY_KD_29DOF.get(j, 5.0) for j in policy_joints],
            dtype=np.float64,
        )

        # Home positions
        q_home_dict = config.control.q_home
        if q_home_dict is None:
            q_home_dict = Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
        self._q_home = np.array(
            [q_home_dict[j] for j in policy_joints], dtype=np.float64
        )
        self._default_pos_robot = np.array(
            [q_home_dict[j] for j in robot_joints], dtype=np.float64
        )
        self._default_pos_policy = self._q_home.copy()

        self._kd_damp = config.control.kd_damp

        # Pre-compute full-robot gain arrays
        self._kp_robot = mapper.fit_gains(self._kp, default=0.0)
        self._kd_robot = mapper.fit_gains(self._kd, default=self._kd_damp)

    def load(self, path: str) -> None:
        """No-op — HoldPolicy has no neural network."""
        pass

    def step(self, state: RobotState, velocity_command: np.ndarray) -> RobotCommand:
        """Hold at home pose with standby gains."""
        target_q_robot = self._mapper.policy_to_robot(
            self._q_home, template=self._default_pos_robot
        )
        return self._build_command(
            state, target_q_robot, self._kp_robot.copy(), self._kd_robot.copy()
        )
