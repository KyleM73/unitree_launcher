"""Sinusoid tracking policy for gantry arm tests.

Holds all joints at home pose with standby gains while sweeping a single
test joint through a sinusoidal trajectory. Used for sim2real comparison
on the gantry.

The sinusoid swings in the negative direction only::

    target = home + amplitude * (cos(2*pi*f*t) - 1) / 2

This starts at home, swings to ``home - amplitude``, and returns.
"""
from __future__ import annotations

import math

import numpy as np

from unitree_launcher.config import (
    Config,
    JOINT_LIMITS_29DOF,
    Q_HOME_23DOF,
    Q_HOME_29DOF,
    STANDBY_KD_29DOF,
    STANDBY_KP_29DOF,
)
from unitree_launcher.policy.base import Policy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotState


class SinusoidPolicy(Policy):
    """Sinusoid on a single joint, hold everywhere else.

    Args:
        mapper: Joint mapper for this robot.
        config: Full config for robot variant and home positions.
        joint_name: Config-name of the joint to sweep (default: right_shoulder_pitch).
        freq_hz: Sinusoid frequency (default: 0.2 Hz = 5s period).
        amplitude: Sinusoid amplitude in radians. If None, uses quarter
            of the negative ROM from home.
    """

    def __init__(
        self,
        mapper: JointMapper,
        config: Config,
        joint_name: str = "right_shoulder_pitch",
        freq_hz: float = 0.2,
        amplitude: float | None = None,
    ) -> None:
        super().__init__(mapper, n_dof=mapper.n_robot)

        variant = config.robot.variant
        policy_joints = mapper.policy_joints
        robot_joints = mapper.robot_joints

        # Standby gains (stiff PD tracking)
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

        # Sinusoid parameters
        self._joint_name = joint_name
        self._joint_idx = policy_joints.index(joint_name)
        self._freq_hz = freq_hz
        self._dt = 1.0 / config.control.policy_frequency

        home = q_home_dict[joint_name]
        if amplitude is None:
            lower = JOINT_LIMITS_29DOF[joint_name][0]
            amplitude = (home - lower) / 4.0
        self._amplitude = amplitude
        self._home = home

        self._step_count = 0

    def load(self, path: str) -> None:
        """No-op — SinusoidPolicy has no neural network."""
        pass

    def step(self, state: RobotState, velocity_command: np.ndarray) -> RobotCommand:
        """Hold at home + sinusoid on test joint."""
        t = self._step_count * self._dt
        self._step_count += 1

        # Sinusoid: starts at home, swings negative, returns
        # target = home + amplitude * (cos(2*pi*f*t) - 1) / 2
        omega = 2.0 * math.pi * self._freq_hz
        sin_offset = self._amplitude * (math.cos(omega * t) - 1.0) / 2.0
        sin_velocity = -self._amplitude * omega * math.sin(omega * t) / 2.0

        # All joints at home
        q_target = self._q_home.copy()
        dq_target = np.zeros_like(q_target)

        # Override test joint
        q_target[self._joint_idx] = self._home + sin_offset
        dq_target[self._joint_idx] = sin_velocity

        target_q_robot = self._mapper.policy_to_robot(
            q_target, template=self._default_pos_robot
        )
        dq_robot = self._mapper.policy_to_robot(dq_target)

        n = self._mapper.n_robot
        return RobotCommand(
            joint_positions=target_q_robot,
            joint_velocities=dq_robot,
            joint_torques=np.zeros(n),
            kp=self._kp_robot.copy(),
            kd=self._kd_robot.copy(),
        )

    def reset(self) -> None:
        """Reset step counter."""
        super().reset()
        self._step_count = 0

    @property
    def observation_dim(self) -> int:
        return 0

    @property
    def joint_name(self) -> str:
        return self._joint_name

    @property
    def freq_hz(self) -> float:
        return self._freq_hz

    @property
    def amplitude(self) -> float:
        return self._amplitude
