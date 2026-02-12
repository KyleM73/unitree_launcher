"""Tests for src/policy/observations.py — Phase 4, Task 4.2."""
import math

import numpy as np
import pytest

from src.config import (
    Config,
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    Q_HOME_29DOF,
    resolve_joint_name,
    ISAACLAB_G1_29DOF_JOINTS,
)
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from src.robot.base import RobotState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ISAACLAB_CONFIG_ORDER = [resolve_joint_name(n) for n in ISAACLAB_G1_29DOF_JOINTS]


def _make_builder(
    n_observed=29,
    n_controlled=29,
    use_estimator=True,
    variant="g1_29dof",
    observed_joints=None,
    controlled_joints=None,
):
    """Convenience builder factory for tests."""
    if variant == "g1_29dof":
        joints = G1_29DOF_JOINTS
    else:
        joints = G1_23DOF_JOINTS

    if observed_joints is None:
        observed_joints = list(joints[:n_observed])
    if controlled_joints is None:
        controlled_joints = list(joints[:n_controlled])

    mapper = JointMapper(joints, observed_joints=observed_joints, controlled_joints=controlled_joints)
    cfg = Config()
    cfg.robot.variant = variant
    return ObservationBuilder(mapper, cfg, use_estimator=use_estimator), mapper


def _make_state(n_dof=29, quat=None, ang_vel=None, base_vel=None, positions=None):
    """Create a RobotState for testing."""
    s = RobotState.zeros(n_dof)
    if quat is not None:
        s.imu_quaternion = np.asarray(quat, dtype=np.float64)
    if ang_vel is not None:
        s.imu_angular_velocity = np.asarray(ang_vel, dtype=np.float64)
    if base_vel is not None:
        s.base_velocity = np.asarray(base_vel, dtype=np.float64)
    if positions is not None:
        s.joint_positions = np.asarray(positions, dtype=np.float64)
    return s


# ---------------------------------------------------------------------------
# Dimension tests
# ---------------------------------------------------------------------------


class TestObservationDim:
    def test_observation_dim_full_body(self):
        """12 + 29 + 29 + 29 = 99 (with estimator)."""
        builder, _ = _make_builder(29, 29, use_estimator=True)
        assert builder.observation_dim == 99

    def test_observation_dim_full_body_no_est(self):
        """9 + 29 + 29 + 29 = 96 (without estimator)."""
        builder, _ = _make_builder(29, 29, use_estimator=False)
        assert builder.observation_dim == 96

    def test_observation_dim_partial_control(self):
        """12 + 29 + 29 + 12 = 82 (with estimator, observe all, control 12)."""
        builder, _ = _make_builder(29, 12, use_estimator=True)
        assert builder.observation_dim == 82

    def test_observation_dim_partial_control_no_est(self):
        """9 + 29 + 29 + 12 = 79."""
        builder, _ = _make_builder(29, 12, use_estimator=False)
        assert builder.observation_dim == 79

    def test_observation_dim_isolated(self):
        """12 + 12 + 12 + 12 = 48 (observe 12, control 12)."""
        legs = list(G1_29DOF_JOINTS[:12])
        builder, _ = _make_builder(
            observed_joints=legs, controlled_joints=legs, use_estimator=True
        )
        assert builder.observation_dim == 48

    def test_observation_dim_isolated_no_est(self):
        """9 + 12 + 12 + 12 = 45."""
        legs = list(G1_29DOF_JOINTS[:12])
        builder, _ = _make_builder(
            observed_joints=legs, controlled_joints=legs, use_estimator=False
        )
        assert builder.observation_dim == 45


# ---------------------------------------------------------------------------
# Gravity and velocity transform tests
# ---------------------------------------------------------------------------


class TestGravityAndVelocity:
    def test_projected_gravity_upright(self):
        """Quaternion [1,0,0,0] -> gravity = [0, 0, -1]."""
        builder, _ = _make_builder()
        g = builder.compute_projected_gravity(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-10)

    def test_projected_gravity_tilted_forward(self):
        """45° pitch forward: gravity should have negative x and z components."""
        angle = math.pi / 4  # 45 degrees
        # Quaternion for rotation about y-axis by +45°
        q = np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])
        builder, _ = _make_builder()
        g = builder.compute_projected_gravity(q)
        # World gravity [0,0,-1] rotated into tilted body frame
        # R^T @ [0,0,-1] where R is rotation about y by 45°
        expected_x = -math.sin(angle) * (-1)  # = sin(45°) ≈ 0.707
        expected_z = -math.cos(angle) * (-1)  # = -cos(45°) ≈ -0.707
        # Actually let's compute properly:
        # R_y(θ) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        # R_y(θ)^T = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        # R^T @ [0,0,-1] = [sin(θ), 0, -cos(θ)]
        np.testing.assert_allclose(
            g, [math.sin(angle), 0.0, -math.cos(angle)], atol=1e-10
        )

    def test_projected_gravity_inverted(self):
        """Upside down -> gravity = [0, 0, 1]."""
        # 180° rotation about x-axis: q = [cos(90°), sin(90°), 0, 0] = [0, 1, 0, 0]
        q = np.array([0.0, 1.0, 0.0, 0.0])
        builder, _ = _make_builder()
        g = builder.compute_projected_gravity(q)
        np.testing.assert_allclose(g, [0.0, 0.0, 1.0], atol=1e-10)

    def test_projected_gravity_90_degree_roll(self):
        """90° roll -> gravity in horizontal plane [0, ±1, 0]."""
        angle = math.pi / 2
        # Quaternion for 90° rotation about x-axis
        q = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        builder, _ = _make_builder()
        g = builder.compute_projected_gravity(q)
        # R_x(90°) = [[1,0,0],[0,0,-1],[0,1,0]]
        # R_x(90°)^T = [[1,0,0],[0,0,1],[0,-1,0]]
        # R^T @ [0,0,-1] = [0, -1, 0]
        np.testing.assert_allclose(g, [0.0, -1.0, 0.0], atol=1e-10)

    def test_body_velocity_transform_identity(self):
        """Upright robot: world vel = body vel."""
        builder, _ = _make_builder()
        v = builder.compute_body_velocity_in_body_frame(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(v, [1.0, 2.0, 3.0], atol=1e-10)

    def test_body_velocity_transform_rotated(self):
        """90° yaw: world [1,0,0] -> body [0,-1,0]."""
        angle = math.pi / 2
        # 90° rotation about z-axis
        q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        builder, _ = _make_builder()
        v = builder.compute_body_velocity_in_body_frame(
            np.array([1.0, 0.0, 0.0]), q
        )
        # R_z(90°) maps body x->world y. R_z(90°)^T maps world x -> body -y? Let's compute:
        # R_z(90°) = [[0,-1,0],[1,0,0],[0,0,1]]
        # R_z(90°)^T = [[0,1,0],[-1,0,0],[0,0,1]]
        # R^T @ [1,0,0] = [0, -1, 0]
        np.testing.assert_allclose(v, [0.0, -1.0, 0.0], atol=1e-10)

    def test_body_velocity_transform_180_yaw(self):
        """180° yaw: world [1,0,0] -> body [-1,0,0]."""
        # 180° about z: q = [cos(90°), 0, 0, sin(90°)] = [0, 0, 0, 1]
        q = np.array([0.0, 0.0, 0.0, 1.0])
        builder, _ = _make_builder()
        v = builder.compute_body_velocity_in_body_frame(
            np.array([1.0, 0.0, 0.0]), q
        )
        np.testing.assert_allclose(v, [-1.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Build output tests
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def test_build_output_shape(self):
        """Output shape matches observation_dim."""
        builder, mapper = _make_builder(29, 29, use_estimator=True)
        state = _make_state()
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        assert obs.shape == (builder.observation_dim,)

    def test_build_output_shape_no_est(self):
        """Output shape matches observation_dim when use_estimator=False."""
        builder, mapper = _make_builder(29, 29, use_estimator=False)
        state = _make_state()
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        assert obs.shape == (builder.observation_dim,)

    def test_build_joint_positions_are_relative(self):
        """joint_pos segment equals q - q_home (not raw positions)."""
        builder, mapper = _make_builder(29, 29, use_estimator=True)
        # Set positions to home -> relative should be zero
        q_home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        state = _make_state(positions=q_home)
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        # joint_pos starts at offset 12 (3 lin_vel + 3 ang_vel + 3 grav + 3 cmd)
        joint_pos_segment = obs[12 : 12 + 29]
        np.testing.assert_allclose(joint_pos_segment, np.zeros(29), atol=1e-10)

    def test_build_joint_velocities_correct(self):
        """Joint velocities are raw (not relative)."""
        builder, mapper = _make_builder(29, 29, use_estimator=True)
        state = _make_state()
        state.joint_velocities = np.ones(29) * 0.5
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        # joint_vel at offset 12 + 29 = 41
        joint_vel_segment = obs[41 : 41 + 29]
        np.testing.assert_allclose(joint_vel_segment, np.full(29, 0.5))

    def test_build_actions_correct(self):
        """Actions segment matches last_action."""
        builder, mapper = _make_builder(29, 29, use_estimator=True)
        state = _make_state()
        last_action = np.arange(29, dtype=np.float64) * 0.1
        obs = builder.build(state, last_action, np.zeros(3))
        # actions at offset 12 + 29 + 29 = 70
        action_segment = obs[70 : 70 + 29]
        np.testing.assert_allclose(action_segment, last_action)

    def test_build_velocity_command_correct(self):
        """Velocity commands segment matches input."""
        builder, mapper = _make_builder(29, 29, use_estimator=True)
        state = _make_state()
        vel_cmd = np.array([0.5, -0.3, 0.1])
        obs = builder.build(state, np.zeros(29), vel_cmd)
        # vel_cmd at offset 3 + 3 + 3 = 9
        np.testing.assert_allclose(obs[9:12], vel_cmd)


# ---------------------------------------------------------------------------
# Segment position / ordering tests
# ---------------------------------------------------------------------------


class TestSegmentPositions:
    def test_build_lin_vel_segment_position(self):
        """base_lin_vel at offset 0 (first 3 elements)."""
        builder, _ = _make_builder(use_estimator=True)
        state = _make_state(base_vel=[1.0, 2.0, 3.0])
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        # Identity quaternion: body vel = world vel
        np.testing.assert_allclose(obs[0:3], [1.0, 2.0, 3.0], atol=1e-10)

    def test_build_ang_vel_segment_position(self):
        """base_ang_vel at offset 3."""
        builder, _ = _make_builder(use_estimator=True)
        state = _make_state(ang_vel=[0.1, 0.2, 0.3])
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        np.testing.assert_allclose(obs[3:6], [0.1, 0.2, 0.3])

    def test_build_gravity_segment_position(self):
        """projected_gravity at offset 6."""
        builder, _ = _make_builder(use_estimator=True)
        state = _make_state()  # upright quaternion
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        np.testing.assert_allclose(obs[6:9], [0.0, 0.0, -1.0], atol=1e-10)

    def test_build_vel_cmd_segment_position(self):
        """velocity_commands at offset 9."""
        builder, _ = _make_builder(use_estimator=True)
        state = _make_state()
        obs = builder.build(state, np.zeros(29), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(obs[9:12], [1.0, 2.0, 3.0])

    def test_build_joint_pos_segment_position(self):
        """joint_pos at offset 12."""
        builder, _ = _make_builder(use_estimator=True)
        state = _make_state()
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        # At home with zero positions -> relative = -q_home
        q_home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        np.testing.assert_allclose(obs[12 : 12 + 29], -q_home, atol=1e-10)

    def test_build_no_est_ang_vel_at_offset_0(self):
        """Without estimator, base_ang_vel starts at offset 0."""
        builder, _ = _make_builder(use_estimator=False)
        state = _make_state(ang_vel=[0.4, 0.5, 0.6])
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        np.testing.assert_allclose(obs[0:3], [0.4, 0.5, 0.6])

    def test_build_no_est_joint_pos_at_offset_9(self):
        """Without estimator, joint_pos starts at offset 9."""
        builder, _ = _make_builder(use_estimator=False)
        q_home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        state = _make_state(positions=q_home)
        obs = builder.build(state, np.zeros(29), np.zeros(3))
        np.testing.assert_allclose(obs[9 : 9 + 29], np.zeros(29), atol=1e-10)


# ---------------------------------------------------------------------------
# Multi-config tests
# ---------------------------------------------------------------------------


class TestMultiConfig:
    def test_23dof_observation_builder(self):
        """ObservationBuilder with 23-DOF: correct dim and output shape."""
        builder, _ = _make_builder(23, 23, variant="g1_23dof")
        assert builder.observation_dim == 12 + 23 + 23 + 23
        state = _make_state(n_dof=23)
        obs = builder.build(state, np.zeros(23), np.zeros(3))
        assert obs.shape == (builder.observation_dim,)

    def test_build_with_mismatched_n_observed_n_controlled(self):
        """n_observed=29, n_controlled=12: actions segment is size 12."""
        all_joints = list(G1_29DOF_JOINTS)
        legs = list(G1_29DOF_JOINTS[:12])
        builder, mapper = _make_builder(
            observed_joints=all_joints, controlled_joints=legs, use_estimator=True
        )
        assert builder.observation_dim == 12 + 29 + 29 + 12  # = 82
        state = _make_state()
        last_action = np.arange(12, dtype=np.float64)
        obs = builder.build(state, last_action, np.zeros(3))
        assert obs.shape == (82,)
        # Actions segment: offset = 12 + 29 + 29 = 70, length = 12
        np.testing.assert_allclose(obs[70:82], last_action)
