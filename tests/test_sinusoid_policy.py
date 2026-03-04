"""Tests for SinusoidPolicy."""
import math

import numpy as np
import pytest

from unitree_launcher.config import G1_29DOF_JOINTS, Q_HOME_29DOF, load_config
from unitree_launcher.policy.sinusoid_policy import SinusoidPolicy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotState

PROJECT_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))


def _make_state(n_dof=29):
    home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
    return RobotState(
        timestamp=0.0,
        joint_positions=home.copy(),
        joint_velocities=np.zeros(n_dof),
        joint_torques=np.zeros(n_dof),
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_angular_velocity=np.zeros(3),
        imu_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        base_position=np.array([0.0, 0.0, 0.793]),
        base_velocity=np.zeros(3),
    )


class TestSinusoidPolicy:
    def test_step_returns_valid_command(self):
        config = load_config(f"{PROJECT_ROOT}/configs/default.yaml")
        mapper = JointMapper(G1_29DOF_JOINTS)
        policy = SinusoidPolicy(mapper, config)
        state = _make_state()

        cmd = policy.step(state, np.zeros(3))
        assert cmd.joint_positions.shape == (29,)
        assert cmd.kp.shape == (29,)
        assert np.all(np.isfinite(cmd.joint_positions))

    def test_starts_at_home(self):
        config = load_config(f"{PROJECT_ROOT}/configs/default.yaml")
        mapper = JointMapper(G1_29DOF_JOINTS)
        policy = SinusoidPolicy(mapper, config)
        state = _make_state()

        # Step 0: t=0, cos(0)=1, offset = amp*(1-1)/2 = 0 → at home
        cmd = policy.step(state, np.zeros(3))
        home = Q_HOME_29DOF["right_shoulder_pitch"]
        idx = G1_29DOF_JOINTS.index("right_shoulder_pitch")
        assert abs(cmd.joint_positions[idx] - home) < 1e-6

    def test_negative_direction_only(self):
        """Sinusoid target never exceeds home position."""
        config = load_config(f"{PROJECT_ROOT}/configs/default.yaml")
        mapper = JointMapper(G1_29DOF_JOINTS)
        policy = SinusoidPolicy(mapper, config)
        state = _make_state()

        home = Q_HOME_29DOF["right_shoulder_pitch"]
        idx = G1_29DOF_JOINTS.index("right_shoulder_pitch")

        for _ in range(250):  # 5 seconds at 50Hz
            cmd = policy.step(state, np.zeros(3))
            assert cmd.joint_positions[idx] <= home + 1e-6, \
                f"Target {cmd.joint_positions[idx]:.4f} exceeds home {home}"

    def test_reaches_amplitude(self):
        """At half period, target reaches home - amplitude."""
        config = load_config(f"{PROJECT_ROOT}/configs/default.yaml")
        mapper = JointMapper(G1_29DOF_JOINTS)
        policy = SinusoidPolicy(mapper, config, freq_hz=0.2)
        state = _make_state()

        home = Q_HOME_29DOF["right_shoulder_pitch"]
        idx = G1_29DOF_JOINTS.index("right_shoulder_pitch")

        # Half period at 0.2Hz = 2.5s = 125 steps at 50Hz
        min_pos = float("inf")
        for _ in range(250):
            cmd = policy.step(state, np.zeros(3))
            min_pos = min(min_pos, cmd.joint_positions[idx])

        expected_min = home - policy.amplitude
        assert abs(min_pos - expected_min) < 1e-4

    def test_reset_restarts_sinusoid(self):
        config = load_config(f"{PROJECT_ROOT}/configs/default.yaml")
        mapper = JointMapper(G1_29DOF_JOINTS)
        policy = SinusoidPolicy(mapper, config)
        state = _make_state()
        idx = G1_29DOF_JOINTS.index("right_shoulder_pitch")

        # Run some steps
        for _ in range(50):
            policy.step(state, np.zeros(3))

        cmd_before_reset = policy.step(state, np.zeros(3))

        # Reset and step — should be back at home (t=0)
        policy.reset()
        cmd_after_reset = policy.step(state, np.zeros(3))

        home = Q_HOME_29DOF["right_shoulder_pitch"]
        assert abs(cmd_after_reset.joint_positions[idx] - home) < 1e-6

    def test_other_joints_at_home(self):
        """Non-test joints stay at home position."""
        config = load_config(f"{PROJECT_ROOT}/configs/default.yaml")
        mapper = JointMapper(G1_29DOF_JOINTS)
        policy = SinusoidPolicy(mapper, config)
        state = _make_state()
        test_idx = G1_29DOF_JOINTS.index("right_shoulder_pitch")

        # Run to half period (max deviation)
        for _ in range(125):
            cmd = policy.step(state, np.zeros(3))

        home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        for i in range(29):
            if i != test_idx:
                assert abs(cmd.joint_positions[i] - home[i]) < 1e-6, \
                    f"Joint {G1_29DOF_JOINTS[i]} deviated from home"
