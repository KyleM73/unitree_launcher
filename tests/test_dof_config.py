"""Tests for DoFConfig dataclass and pre-built instances."""
import numpy as np
import pytest

from unitree_launcher.config import (
    BM_ACTION_SCALE_29DOF,
    DOF_CONFIGS,
    DOF_ISAACLAB_29,
    DOF_STANDBY_29,
    DOF_UNITREE_29,
    DoFConfig,
    G1_29DOF_JOINTS,
    ISAACLAB_KD_29DOF,
    ISAACLAB_KP_29DOF,
    JOINT_LIMITS_29DOF,
    Q_HOME_29DOF,
    STANDBY_KD_29DOF,
    STANDBY_KP_29DOF,
    TORQUE_LIMITS_29DOF,
    UNITREE_KD_29DOF,
    UNITREE_KP_29DOF,
    VELOCITY_LIMITS_29DOF,
)


class TestDoFConfigShape:
    def test_isaaclab_29_has_29_joints(self):
        assert DOF_ISAACLAB_29.n_dof == 29

    def test_standby_29_has_29_joints(self):
        assert DOF_STANDBY_29.n_dof == 29

    def test_unitree_29_has_29_joints(self):
        assert DOF_UNITREE_29.n_dof == 29

    def test_joint_names_match(self):
        assert DOF_ISAACLAB_29.joint_names == G1_29DOF_JOINTS

    def test_array_shapes(self):
        cfg = DOF_ISAACLAB_29
        assert cfg.default_pos.shape == (29,)
        assert cfg.stiffness.shape == (29,)
        assert cfg.damping.shape == (29,)
        assert cfg.action_scale.shape == (29,)
        assert cfg.torque_limits.shape == (29,)
        assert cfg.position_limits.shape == (29, 2)
        assert cfg.velocity_limits.shape == (29,)


class TestDoFConfigValues:
    def test_isaaclab_stiffness_matches_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_ISAACLAB_29.stiffness[i] == pytest.approx(ISAACLAB_KP_29DOF[j])

    def test_isaaclab_damping_matches_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_ISAACLAB_29.damping[i] == pytest.approx(ISAACLAB_KD_29DOF[j])

    def test_isaaclab_action_scale_matches_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_ISAACLAB_29.action_scale[i] == pytest.approx(BM_ACTION_SCALE_29DOF[j])

    def test_isaaclab_default_pos_matches_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_ISAACLAB_29.default_pos[i] == pytest.approx(Q_HOME_29DOF[j])

    def test_standby_stiffness_matches_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_STANDBY_29.stiffness[i] == pytest.approx(STANDBY_KP_29DOF[j])

    def test_unitree_stiffness_matches_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_UNITREE_29.stiffness[i] == pytest.approx(UNITREE_KP_29DOF[j])

    def test_position_limits_match_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            lo, hi = JOINT_LIMITS_29DOF[j]
            assert DOF_ISAACLAB_29.position_limits[i, 0] == pytest.approx(lo)
            assert DOF_ISAACLAB_29.position_limits[i, 1] == pytest.approx(hi)

    def test_torque_limits_match_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_ISAACLAB_29.torque_limits[i] == pytest.approx(TORQUE_LIMITS_29DOF[j])

    def test_velocity_limits_match_dict(self):
        for i, j in enumerate(G1_29DOF_JOINTS):
            assert DOF_ISAACLAB_29.velocity_limits[i] == pytest.approx(VELOCITY_LIMITS_29DOF[j])


class TestDoFConfigForJoints:
    def test_for_joints_slices_correctly(self):
        subset = ["left_hip_pitch", "right_knee", "left_elbow"]
        sliced = DOF_ISAACLAB_29.for_joints(subset)

        assert sliced.n_dof == 3
        assert sliced.joint_names == subset

        for i, j in enumerate(subset):
            orig_idx = G1_29DOF_JOINTS.index(j)
            assert sliced.default_pos[i] == pytest.approx(DOF_ISAACLAB_29.default_pos[orig_idx])
            assert sliced.stiffness[i] == pytest.approx(DOF_ISAACLAB_29.stiffness[orig_idx])
            assert sliced.damping[i] == pytest.approx(DOF_ISAACLAB_29.damping[orig_idx])
            assert sliced.action_scale[i] == pytest.approx(DOF_ISAACLAB_29.action_scale[orig_idx])
            assert sliced.torque_limits[i] == pytest.approx(DOF_ISAACLAB_29.torque_limits[orig_idx])
            np.testing.assert_array_almost_equal(
                sliced.position_limits[i], DOF_ISAACLAB_29.position_limits[orig_idx]
            )
            assert sliced.velocity_limits[i] == pytest.approx(DOF_ISAACLAB_29.velocity_limits[orig_idx])

    def test_for_joints_unknown_raises(self):
        with pytest.raises(ValueError):
            DOF_ISAACLAB_29.for_joints(["nonexistent_joint"])


class TestDoFConfigRegistry:
    def test_registry_has_expected_keys(self):
        assert "isaaclab_29" in DOF_CONFIGS
        assert "standby_29" in DOF_CONFIGS
        assert "unitree_29" in DOF_CONFIGS

    def test_registry_values_are_same_objects(self):
        assert DOF_CONFIGS["isaaclab_29"] is DOF_ISAACLAB_29
        assert DOF_CONFIGS["standby_29"] is DOF_STANDBY_29
        assert DOF_CONFIGS["unitree_29"] is DOF_UNITREE_29
