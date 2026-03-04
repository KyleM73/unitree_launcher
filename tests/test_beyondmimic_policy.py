"""Tests for BeyondMimicPolicy (policy/beyondmimic_policy.py).

Includes value-level geometry checks for 6D rotation, body-relative
transforms, and quaternion operations.
"""
import os

import numpy as np
import pytest

from unitree_launcher.config import G1_29DOF_JOINTS, ISAACLAB_G1_29DOF_JOINTS, load_config
from unitree_launcher.policy.beyondmimic_policy import (
    BeyondMimicPolicy,
    compute_body_relative_orientation,
    compute_body_relative_position,
    quat_inverse,
    quat_multiply,
    quat_to_6d,
    quat_to_rotation_matrix,
)
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotState
from tests.conftest import PROJECT_ROOT, create_beyondmimic_onnx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ISAACLAB_ORDER_CONFIG_NAMES = [
    # Map MuJoCo names in ISAACLAB_G1_29DOF_JOINTS to config names
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch",
    "left_knee", "right_knee",
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]


@pytest.fixture
def joint_mapper_isaaclab_order():
    """JointMapper with observed/controlled in IsaacLab order."""
    return JointMapper(
        G1_29DOF_JOINTS,
        policy_joints=_ISAACLAB_ORDER_CONFIG_NAMES,
    )


@pytest.fixture
def bm_obs_dim():
    """BeyondMimic observation dim: command(58) + anchor_pos(3) + anchor_ori(6)
    + lin_vel(3) + ang_vel(3) + joint_pos(29) + joint_vel(29) + actions(29) = 160
    """
    return 160


@pytest.fixture
def bm_metadata():
    """Minimal valid metadata dict for a BeyondMimic model."""
    return {
        "joint_names": ",".join(ISAACLAB_G1_29DOF_JOINTS),
        "joint_stiffness": ",".join(["40.0"] * 29),
        "joint_damping": ",".join(["2.5"] * 29),
        "action_scale": ",".join(["0.5"] * 29),
        "default_joint_pos": ",".join(["0.0"] * 29),
        "anchor_body_name": "torso_link",
        "body_names": "pelvis,torso_link,left_knee_link",
        "observation_names": "command,motion_anchor_pos_b,motion_anchor_ori_b,"
                             "base_lin_vel,base_ang_vel,joint_pos,joint_vel,actions",
        "observation_history_lengths": ",".join(["1.0"] * 8),
    }


@pytest.fixture
def bm_model_path(tmp_path, bm_obs_dim, bm_metadata):
    """Path to a test BeyondMimic ONNX model with metadata."""
    path = str(tmp_path / "bm_policy.onnx")
    create_beyondmimic_onnx(bm_obs_dim, 29, 29, path, metadata=bm_metadata)
    return path


@pytest.fixture
def bm_policy(joint_mapper_isaaclab_order, bm_obs_dim, bm_model_path):
    """Loaded BeyondMimicPolicy."""
    policy = BeyondMimicPolicy(
        joint_mapper_isaaclab_order, bm_obs_dim, use_onnx_metadata=True
    )
    policy.load(bm_model_path)
    return policy


@pytest.fixture
def standing_robot_state():
    """Robot state at home position, upright."""
    from unitree_launcher.config import Q_HOME_29DOF
    home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
    return RobotState(
        timestamp=0.0,
        joint_positions=home.copy(),
        joint_velocities=np.zeros(29),
        joint_torques=np.zeros(29),
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_angular_velocity=np.zeros(3),
        imu_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        base_position=np.array([0.0, 0.0, 0.793]),
        base_velocity=np.zeros(3),
    )


# ======================================================================
# Geometry helpers — value-level tests
# ======================================================================

class TestQuatToRotationMatrix:

    def test_identity(self):
        R = quat_to_rotation_matrix(np.array([1, 0, 0, 0], dtype=float))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90_yaw(self):
        """90° rotation about Z axis."""
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        R = quat_to_rotation_matrix(q)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_180_pitch(self):
        """180° rotation about Y axis."""
        q = np.array([0, 0, 1, 0], dtype=float)  # cos(90)=0, sin(90)=1
        R = quat_to_rotation_matrix(q)
        expected = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)


class TestQuatTo6D:

    def test_identity_6d(self):
        """Identity quaternion -> first 2 columns of I3 in row-major order."""
        result = quat_to_6d(np.array([1, 0, 0, 0], dtype=float))
        # R[:,:2].flatten() for I3 = [1,0, 0,1, 0,0]
        expected = np.array([1, 0, 0, 1, 0, 0], dtype=float)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_90_yaw_6d(self):
        """90° yaw -> first 2 columns [[0,-1],[1,0],[0,0]] row-major."""
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        result = quat_to_6d(q)
        # R_z(90°) = [[0,-1,0],[1,0,0],[0,0,1]], R[:,:2].flatten() = [0,-1, 1,0, 0,0]
        expected = np.array([0, -1, 1, 0, 0, 0], dtype=float)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_shape(self):
        result = quat_to_6d(np.array([1, 0, 0, 0], dtype=float))
        assert result.shape == (6,)


class TestQuatInverse:

    def test_conjugate(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        qi = quat_inverse(q)
        np.testing.assert_allclose(qi, [0.5, -0.5, -0.5, -0.5])

    def test_identity_inverse(self):
        q = np.array([1, 0, 0, 0], dtype=float)
        qi = quat_inverse(q)
        np.testing.assert_allclose(qi, q)


class TestQuatMultiply:

    def test_identity_product(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        identity = np.array([1, 0, 0, 0], dtype=float)
        result = quat_multiply(identity, q)
        np.testing.assert_allclose(result, q)

    def test_inverse_product_gives_identity(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        qi = quat_inverse(q)
        result = quat_multiply(q, qi)
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-12)

    def test_two_90_yaws_give_180(self):
        """Two 90° yaw rotations = 180° yaw."""
        angle = np.pi / 2
        q90 = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        q180 = quat_multiply(q90, q90)
        expected = np.array([np.cos(np.pi/2), 0, 0, np.sin(np.pi/2)])
        np.testing.assert_allclose(q180, expected, atol=1e-12)


class TestBodyRelativePosition:

    def test_same_position_is_zero(self):
        pos = np.array([1.0, 2.0, 3.0])
        q = np.array([1, 0, 0, 0], dtype=float)
        result = compute_body_relative_position(pos, q, pos)
        np.testing.assert_allclose(result, [0, 0, 0], atol=1e-12)

    def test_offset_no_rotation(self):
        """Body 1m ahead in X, identity rotation."""
        anchor_pos = np.array([0, 0, 0], dtype=float)
        anchor_q = np.array([1, 0, 0, 0], dtype=float)
        body_pos = np.array([1, 0, 0], dtype=float)
        result = compute_body_relative_position(anchor_pos, anchor_q, body_pos)
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-12)

    def test_offset_with_90_yaw(self):
        """Body at world [1,0,0], anchor at origin rotated 90° yaw.

        After 90° yaw, the body's X=1 in world becomes Y=-1 in body frame.
        Wait, R^T = R_z(-90°). World [1,0,0] -> body frame:
        R_z(90°)^T @ [1,0,0] = [0, -1, 0]... no wait.

        R_z(90°) maps body->world. So body x-axis points to world y.
        R_z(90°)^T maps world->body. So world x = body (-y)?
        R_z(90°) = [[0,-1,0],[1,0,0],[0,0,1]]
        R_z(90°)^T = [[0,1,0],[-1,0,0],[0,0,1]]
        R_z(90°)^T @ [1,0,0] = [0,-1,0]
        """
        angle = np.pi / 2
        anchor_q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        anchor_pos = np.zeros(3)
        body_pos = np.array([1, 0, 0], dtype=float)
        result = compute_body_relative_position(anchor_pos, anchor_q, body_pos)
        np.testing.assert_allclose(result, [0, -1, 0], atol=1e-12)

    def test_known_values(self):
        """Hand-computed: anchor at [1,2,0], body at [3,2,0], identity rotation.
        Relative: [2,0,0]."""
        anchor_pos = np.array([1, 2, 0], dtype=float)
        anchor_q = np.array([1, 0, 0, 0], dtype=float)
        body_pos = np.array([3, 2, 0], dtype=float)
        result = compute_body_relative_position(anchor_pos, anchor_q, body_pos)
        np.testing.assert_allclose(result, [2, 0, 0], atol=1e-12)


class TestBodyRelativeOrientation:

    def test_same_orientation_gives_identity_6d(self):
        q = np.array([1, 0, 0, 0], dtype=float)
        result = compute_body_relative_orientation(q, q)
        expected = quat_to_6d(np.array([1, 0, 0, 0], dtype=float))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_90_yaw_difference(self):
        """Anchor identity, body 90° yaw -> relative = 90° yaw."""
        anchor_q = np.array([1, 0, 0, 0], dtype=float)
        angle = np.pi / 2
        body_q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        result = compute_body_relative_orientation(anchor_q, body_q)
        expected = quat_to_6d(body_q)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_opposite_rotation_gives_inverse(self):
        """Both rotated same amount -> relative is identity."""
        angle = np.pi / 4
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        result = compute_body_relative_orientation(q, q)
        expected = quat_to_6d(np.array([1, 0, 0, 0], dtype=float))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_shape(self):
        q = np.array([1, 0, 0, 0], dtype=float)
        result = compute_body_relative_orientation(q, q)
        assert result.shape == (6,)


# ======================================================================
# BeyondMimicPolicy — loading and metadata
# ======================================================================

class TestBeyondMimicLoad:

    def test_load_valid_policy(self, bm_policy, bm_obs_dim):
        assert bm_policy.observation_dim == bm_obs_dim
        assert bm_policy.action_dim == 29

    def test_metadata_extraction(self, bm_policy):
        md = bm_policy.metadata
        assert "joint_names" in md
        assert bm_policy.anchor_body_name == "torso_link"
        assert len(bm_policy.body_names) == 3
        assert len(bm_policy.obs_terms) == 8

    def test_metadata_overrides_config(self, bm_policy):
        """When use_onnx_metadata=True, stiffness/damping/action_scale are loaded."""
        assert bm_policy.stiffness is not None
        assert bm_policy.damping is not None
        assert bm_policy.action_scale is not None
        np.testing.assert_allclose(bm_policy.stiffness, np.full(29, 40.0))
        np.testing.assert_allclose(bm_policy.damping, np.full(29, 2.5))

    def test_metadata_not_used_when_disabled(
        self, joint_mapper_isaaclab_order, bm_obs_dim, bm_model_path
    ):
        policy = BeyondMimicPolicy(
            joint_mapper_isaaclab_order, bm_obs_dim, use_onnx_metadata=False
        )
        policy.load(bm_model_path)
        # stiffness/damping/action_scale return defaults (never None)
        assert policy.stiffness is not None
        assert policy.damping is not None
        assert policy.action_scale is not None

    def test_metadata_missing_required_field(
        self, joint_mapper_isaaclab_order, bm_obs_dim, tmp_path
    ):
        path = str(tmp_path / "no_joints.onnx")
        bad_metadata = {"anchor_body_name": "torso_link"}
        create_beyondmimic_onnx(bm_obs_dim, 29, 29, path, metadata=bad_metadata)
        policy = BeyondMimicPolicy(joint_mapper_isaaclab_order, bm_obs_dim)
        with pytest.raises(ValueError, match="joint_names"):
            policy.load(path)

    def test_load_invalid_path_raises(
        self, joint_mapper_isaaclab_order, bm_obs_dim
    ):
        policy = BeyondMimicPolicy(joint_mapper_isaaclab_order, bm_obs_dim)
        with pytest.raises(ValueError):
            policy.load("/nonexistent/path.onnx")


# ======================================================================
# BeyondMimicPolicy — inference
# ======================================================================

class TestBeyondMimicInference:

    def test_get_action_with_time_step(self, bm_policy, bm_obs_dim):
        obs = np.zeros(bm_obs_dim)
        action = bm_policy.get_action(obs, time_step=0.0)
        assert action.shape == (29,)

    def test_get_action_stores_targets(self, bm_policy, bm_obs_dim):
        obs = np.zeros(bm_obs_dim)
        bm_policy.get_action(obs, time_step=0.0)
        assert bm_policy.target_q.shape == (29,)
        assert bm_policy.target_dq.shape == (29,)

    def test_get_action_output_shape(self, bm_policy, bm_obs_dim):
        obs = np.zeros(bm_obs_dim)
        action = bm_policy.get_action(obs, time_step=1.0)
        assert action.shape == (29,)

    def test_get_action_missing_time_step(self, bm_policy, bm_obs_dim):
        obs = np.zeros(bm_obs_dim)
        with pytest.raises(TypeError, match="time_step"):
            bm_policy.get_action(obs)

    def test_get_action_without_load_raises(
        self, joint_mapper_isaaclab_order, bm_obs_dim
    ):
        policy = BeyondMimicPolicy(joint_mapper_isaaclab_order, bm_obs_dim)
        with pytest.raises(RuntimeError, match="No policy loaded"):
            policy.get_action(np.zeros(bm_obs_dim), time_step=0.0)

    def test_reset_clears_targets(self, bm_policy, bm_obs_dim):
        obs = np.zeros(bm_obs_dim)
        bm_policy.get_action(obs, time_step=0.0)
        bm_policy.reset()
        np.testing.assert_array_equal(bm_policy.target_q, np.zeros(29))
        np.testing.assert_array_equal(bm_policy.target_dq, np.zeros(29))

    def test_deterministic(self, bm_policy, bm_obs_dim):
        obs = np.random.randn(bm_obs_dim).astype(np.float32)
        a1 = bm_policy.get_action(obs, time_step=0.0)
        bm_policy.reset()
        a2 = bm_policy.get_action(obs, time_step=0.0)
        np.testing.assert_array_equal(a1, a2)


# ======================================================================
# BeyondMimicPolicy — observation building
# ======================================================================

class TestBeyondMimicObservation:

    def test_build_observation_shape(self, bm_policy, bm_obs_dim, standing_robot_state):
        anchor_pos = np.array([0, 0, 0.793])
        anchor_quat = np.array([1, 0, 0, 0], dtype=float)
        obs = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        assert obs.shape == (bm_obs_dim,)

    def test_build_observation_command_is_prev_targets(
        self, bm_policy, bm_obs_dim, standing_robot_state
    ):
        """Before any inference, command (first 58 dims) should be zeros."""
        anchor_pos = np.zeros(3)
        anchor_quat = np.array([1, 0, 0, 0], dtype=float)
        obs = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        # command = prev_target_q(29) + prev_target_dq(29) = first 58 dims
        np.testing.assert_array_equal(obs[:58], np.zeros(58))

    def test_build_observation_base_velocities(
        self, bm_policy, standing_robot_state
    ):
        """base_lin_vel and base_ang_vel come from robot state."""
        standing_robot_state.base_velocity = np.array([0.1, 0.2, 0.3])
        standing_robot_state.imu_angular_velocity = np.array([0.4, 0.5, 0.6])
        anchor_pos = np.zeros(3)
        anchor_quat = np.array([1, 0, 0, 0], dtype=float)
        obs = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        # command(58) + anchor_pos(3) + anchor_ori(6) = 67
        # Then base_lin_vel starts at index 67
        np.testing.assert_allclose(obs[67:70], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(obs[70:73], [0.4, 0.5, 0.6])

    def test_build_observation_joint_pos_relative_to_default(
        self, bm_policy, standing_robot_state
    ):
        """joint_pos in observation should be relative to default_joint_pos."""
        anchor_pos = np.zeros(3)
        anchor_quat = np.array([1, 0, 0, 0], dtype=float)
        obs = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        # command(58) + anchor_pos(3) + anchor_ori(6) + lin_vel(3) + ang_vel(3) = 73
        # joint_pos starts at 73, length 29
        default_jp = bm_policy.default_joint_pos
        # The mapper reorders robot-native to IsaacLab order
        expected_jp = bm_policy._mapper.robot_to_policy(
            standing_robot_state.joint_positions
        ) - default_jp
        np.testing.assert_allclose(obs[73:102], expected_jp, atol=1e-12)

    def test_build_observation_actions_follow_inference(
        self, bm_policy, bm_obs_dim, standing_robot_state
    ):
        """After inference, the 'actions' term should contain the last action."""
        anchor_pos = np.zeros(3)
        anchor_quat = np.array([1, 0, 0, 0], dtype=float)

        # Run inference once
        obs1 = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        action = bm_policy.get_action(obs1, time_step=0.0)

        # Build next observation — actions term should have the last action
        obs2 = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        # actions is the last 29 dims
        np.testing.assert_allclose(obs2[-29:], action, atol=1e-12)

    def test_build_observation_motion_anchor_pos_zeros_initially(
        self, bm_policy, standing_robot_state
    ):
        """Before any inference, motion_anchor_pos_b should be zeros."""
        anchor_pos = np.zeros(3)
        anchor_quat = np.array([1, 0, 0, 0], dtype=float)
        obs = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        # motion_anchor_pos_b starts at index 58, length 3
        np.testing.assert_allclose(obs[58:61], [0, 0, 0], atol=1e-12)

    def test_build_observation_motion_anchor_ori_identity_initially(
        self, bm_policy, standing_robot_state
    ):
        """Before any inference, motion_anchor_ori_b should be identity 6D."""
        anchor_pos = np.zeros(3)
        anchor_quat = np.array([1, 0, 0, 0], dtype=float)
        obs = bm_policy.build_observation(standing_robot_state, anchor_pos, anchor_quat)
        # motion_anchor_ori_b starts at index 61, length 6
        expected = quat_to_6d(np.array([1, 0, 0, 0], dtype=float))
        np.testing.assert_allclose(obs[61:67], expected, atol=1e-12)

