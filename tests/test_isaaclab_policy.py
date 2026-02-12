"""Tests for IsaacLab policy backend (Phase 6, Task 6.3)."""
import os
import tempfile

import numpy as np
import pytest

from src.config import G1_29DOF_JOINTS, load_config
from src.policy.base import detect_policy_format
from src.policy.isaaclab_policy import IsaacLabPolicy
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from tests.conftest import PROJECT_ROOT, create_isaaclab_onnx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def joint_mapper_29dof():
    """JointMapper with all 29 joints in robot-native order."""
    return JointMapper(G1_29DOF_JOINTS)


@pytest.fixture
def obs_builder_29dof(joint_mapper_29dof):
    """ObservationBuilder for 29-DOF with estimator."""
    config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
    return ObservationBuilder(joint_mapper_29dof, config, use_estimator=True)


@pytest.fixture
def obs_dim_29dof(obs_builder_29dof):
    return obs_builder_29dof.observation_dim


@pytest.fixture
def policy_with_model(joint_mapper_29dof, obs_dim_29dof, tmp_path):
    """IsaacLabPolicy with a loaded test ONNX model."""
    model_path = str(tmp_path / "test_policy.onnx")
    create_isaaclab_onnx(obs_dim_29dof, 29, model_path)
    policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim_29dof)
    policy.load(model_path)
    return policy


# ---------------------------------------------------------------------------
# detect_policy_format tests (Task 6.2)
# ---------------------------------------------------------------------------

class TestDetectPolicyFormat:

    def test_detect_isaaclab(self, tmp_path):
        path = str(tmp_path / "il.onnx")
        create_isaaclab_onnx(10, 5, path)
        assert detect_policy_format(path) == "isaaclab"

    def test_detect_beyondmimic(self, tmp_path):
        from tests.conftest import create_beyondmimic_onnx
        path = str(tmp_path / "bm.onnx")
        create_beyondmimic_onnx(10, 5, 5, path)
        assert detect_policy_format(path) == "beyondmimic"

    def test_detect_corrupt_file(self, tmp_path):
        path = str(tmp_path / "bad.onnx")
        with open(path, "wb") as f:
            f.write(b"not a valid onnx model")
        with pytest.raises(ValueError, match="Failed to load"):
            detect_policy_format(path)

    def test_detect_missing_file(self, tmp_path):
        with pytest.raises(ValueError):
            detect_policy_format(str(tmp_path / "missing.onnx"))


# ---------------------------------------------------------------------------
# IsaacLabPolicy tests (Task 6.3)
# ---------------------------------------------------------------------------

class TestIsaacLabPolicyLoad:

    def test_load_valid_policy(self, joint_mapper_29dof, obs_dim_29dof, tmp_path):
        model_path = str(tmp_path / "policy.onnx")
        create_isaaclab_onnx(obs_dim_29dof, 29, model_path)
        policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim_29dof)
        policy.load(model_path)
        assert policy.observation_dim == obs_dim_29dof
        assert policy.action_dim == 29

    def test_load_invalid_path_raises(self, joint_mapper_29dof, obs_dim_29dof):
        policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim_29dof)
        with pytest.raises(ValueError, match="Failed to load"):
            policy.load("/nonexistent/path/policy.onnx")

    def test_load_dimension_mismatch_raises(self, joint_mapper_29dof, tmp_path):
        model_path = str(tmp_path / "bad_dim.onnx")
        create_isaaclab_onnx(50, 29, model_path)  # wrong obs_dim
        policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim=99)
        with pytest.raises(ValueError, match="observation dim"):
            policy.load(model_path)

    def test_load_action_dim_mismatch_raises(
        self, joint_mapper_29dof, obs_dim_29dof, tmp_path
    ):
        model_path = str(tmp_path / "bad_action.onnx")
        create_isaaclab_onnx(obs_dim_29dof, 10, model_path)  # wrong action_dim
        policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim_29dof)
        with pytest.raises(ValueError, match="action dim"):
            policy.load(model_path)

    def test_load_corrupt_onnx_raises(self, joint_mapper_29dof, obs_dim_29dof, tmp_path):
        bad_path = str(tmp_path / "corrupt.onnx")
        with open(bad_path, "wb") as f:
            f.write(b"\x00\x01\x02corrupt")
        policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim_29dof)
        with pytest.raises(ValueError, match="Failed to load"):
            policy.load(bad_path)

    def test_load_twice_replaces_session(
        self, joint_mapper_29dof, obs_dim_29dof, tmp_path
    ):
        path1 = str(tmp_path / "p1.onnx")
        path2 = str(tmp_path / "p2.onnx")
        create_isaaclab_onnx(obs_dim_29dof, 29, path1)
        create_isaaclab_onnx(obs_dim_29dof, 29, path2)
        policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim_29dof)
        policy.load(path1)
        policy.load(path2)
        # Should work without errors — second load replaces first
        obs = np.zeros(obs_dim_29dof)
        action = policy.get_action(obs)
        assert action.shape == (29,)


class TestIsaacLabPolicyInference:

    def test_get_action_output_shape(self, policy_with_model, obs_dim_29dof):
        obs = np.zeros(obs_dim_29dof)
        action = policy_with_model.get_action(obs)
        assert action.shape == (29,)

    def test_get_action_output_dtype(self, policy_with_model, obs_dim_29dof):
        obs = np.zeros(obs_dim_29dof)
        action = policy_with_model.get_action(obs)
        assert action.dtype == np.float64

    def test_get_action_deterministic(self, policy_with_model, obs_dim_29dof):
        obs = np.random.randn(obs_dim_29dof)
        a1 = policy_with_model.get_action(obs)
        a2 = policy_with_model.get_action(obs)
        np.testing.assert_array_equal(a1, a2)

    def test_get_action_without_load_raises(self, joint_mapper_29dof, obs_dim_29dof):
        policy = IsaacLabPolicy(joint_mapper_29dof, obs_dim_29dof)
        with pytest.raises(RuntimeError, match="No policy loaded"):
            policy.get_action(np.zeros(obs_dim_29dof))

    def test_reset_clears_state(self, policy_with_model, obs_dim_29dof):
        obs = np.ones(obs_dim_29dof)
        policy_with_model.get_action(obs)
        assert policy_with_model.last_action is not None
        policy_with_model.reset()
        np.testing.assert_array_equal(
            policy_with_model.last_action,
            np.zeros(29),
        )

    def test_observation_dim_matches_builder(
        self, policy_with_model, obs_builder_29dof
    ):
        assert policy_with_model.observation_dim == obs_builder_29dof.observation_dim

    def test_last_action_updated_after_inference(
        self, policy_with_model, obs_dim_29dof
    ):
        obs = np.zeros(obs_dim_29dof)
        action = policy_with_model.get_action(obs)
        np.testing.assert_array_equal(policy_with_model.last_action, action)

    def test_last_action_is_copy(self, policy_with_model, obs_dim_29dof):
        """last_action should be a copy, not a reference to internal state."""
        obs = np.zeros(obs_dim_29dof)
        action = policy_with_model.get_action(obs)
        la = policy_with_model.last_action
        la[:] = 999.0
        np.testing.assert_array_equal(policy_with_model.last_action, action)
