"""Tests for src/policy/joint_mapper.py — Phase 4, Task 4.1."""
import numpy as np
import pytest

from unitree_launcher.config import (
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    ISAACLAB_TO_NATIVE_INDICES,
    resolve_joint_name,
)
from unitree_launcher.policy.joint_mapper import JointMapper


# ---------------------------------------------------------------------------
# Helper: IsaacLab joint names in config-name form (policy order)
# ---------------------------------------------------------------------------
ISAACLAB_CONFIG_ORDER = [resolve_joint_name(n, "g1_29dof") for n in ISAACLAB_G1_29DOF_JOINTS]


# ---------------------------------------------------------------------------
# Default resolution tests
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_all_joints(self):
        """Both None -> observe and control all 29 in native order."""
        m = JointMapper(G1_29DOF_JOINTS)
        assert m.n_observed == 29
        assert m.n_controlled == 29
        assert m.n_total == 29
        np.testing.assert_array_equal(m.observed_indices, np.arange(29))
        np.testing.assert_array_equal(m.controlled_indices, np.arange(29))

    def test_controlled_only(self):
        """Only controlled_joints specified -> observed defaults to controlled."""
        legs = G1_29DOF_JOINTS[:12]
        m = JointMapper(G1_29DOF_JOINTS, controlled_joints=legs)
        assert m.n_observed == 12
        assert m.n_controlled == 12
        assert m.observed_joints == legs
        assert m.controlled_joints == legs

    def test_observed_only(self):
        """Only observed_joints specified -> controlled defaults to all."""
        legs = G1_29DOF_JOINTS[:12]
        m = JointMapper(G1_29DOF_JOINTS, observed_joints=legs)
        assert m.n_observed == 12
        assert m.n_controlled == 29
        assert m.controlled_joints == list(G1_29DOF_JOINTS)

    def test_both_specified(self):
        """Both specified with different sets."""
        obs = G1_29DOF_JOINTS[:6]
        ctrl = G1_29DOF_JOINTS[6:12]
        m = JointMapper(G1_29DOF_JOINTS, observed_joints=obs, controlled_joints=ctrl)
        assert m.n_observed == 6
        assert m.n_controlled == 6
        assert m.observed_joints == list(obs)
        assert m.controlled_joints == list(ctrl)


# ---------------------------------------------------------------------------
# Identity and reordering tests
# ---------------------------------------------------------------------------


class TestMapping:
    def test_identity_mapping(self):
        """When policy order == robot order, indices are identity."""
        m = JointMapper(G1_29DOF_JOINTS)
        np.testing.assert_array_equal(m.observed_indices, np.arange(29))

    def test_isaaclab_reordering(self):
        """IsaacLab 29-DOF order: indices match ISAACLAB_TO_NATIVE_INDICES."""
        m = JointMapper(
            G1_29DOF_JOINTS,
            observed_joints=ISAACLAB_CONFIG_ORDER,
            controlled_joints=ISAACLAB_CONFIG_ORDER,
        )
        np.testing.assert_array_equal(
            m.observed_indices, np.array(ISAACLAB_TO_NATIVE_INDICES)
        )

    def test_robot_to_observation_values(self):
        """Known robot state -> verify observation matches expected reordering."""
        m = JointMapper(
            G1_29DOF_JOINTS,
            observed_joints=ISAACLAB_CONFIG_ORDER,
        )
        robot_vals = np.arange(29, dtype=np.float64)
        obs = m.robot_to_observation(robot_vals)
        # First few expected values: native[0]=0, native[6]=6, native[12]=12, native[1]=1
        expected_first4 = [0.0, 6.0, 12.0, 1.0]
        np.testing.assert_array_equal(obs[:4], expected_first4)

    def test_robot_to_action_values(self):
        """Known robot state -> verify action subset/reorder."""
        legs = G1_29DOF_JOINTS[:12]  # first 12 joints
        m = JointMapper(G1_29DOF_JOINTS, controlled_joints=legs)
        robot_vals = np.arange(29, dtype=np.float64)
        action = m.robot_to_action(robot_vals)
        np.testing.assert_array_equal(action, np.arange(12, dtype=np.float64))

    def test_action_to_robot_values(self):
        """Known action -> correct indices filled, default elsewhere."""
        legs = G1_29DOF_JOINTS[:12]
        m = JointMapper(G1_29DOF_JOINTS, controlled_joints=legs)
        action = np.ones(12) * 5.0
        result = m.action_to_robot(action, default_value=-1.0)
        np.testing.assert_array_equal(result[:12], np.full(12, 5.0))
        np.testing.assert_array_equal(result[12:], np.full(17, -1.0))

    def test_action_to_robot_roundtrip(self):
        """action_to_robot(robot_to_action(full)) reconstructs controlled joints."""
        m = JointMapper(G1_29DOF_JOINTS)
        full = np.random.randn(29)
        roundtrip = m.action_to_robot(m.robot_to_action(full))
        np.testing.assert_allclose(roundtrip, full)

    def test_action_to_robot_default_value(self):
        """Uncontrolled joints filled with the specified default."""
        legs = G1_29DOF_JOINTS[:6]
        m = JointMapper(G1_29DOF_JOINTS, controlled_joints=legs)
        result = m.action_to_robot(np.zeros(6), default_value=99.0)
        assert result[6] == 99.0
        assert result[28] == 99.0

    def test_isaaclab_reordering_known_state(self):
        """Numerical check: state [0.0, 0.1, ..., 2.8], IsaacLab reordering."""
        m = JointMapper(
            G1_29DOF_JOINTS,
            observed_joints=ISAACLAB_CONFIG_ORDER,
        )
        robot_vals = np.arange(29) * 0.1
        obs = m.robot_to_observation(robot_vals)
        # IsaacLab index 0 -> native 0 -> 0.0
        # IsaacLab index 1 -> native 6 -> 0.6
        # IsaacLab index 2 -> native 12 -> 1.2
        # IsaacLab index 3 -> native 1 -> 0.1
        # IsaacLab index 12 -> native 22 -> 2.2
        assert obs[0] == pytest.approx(0.0)
        assert obs[1] == pytest.approx(0.6)
        assert obs[2] == pytest.approx(1.2)
        assert obs[3] == pytest.approx(0.1)
        assert obs[12] == pytest.approx(2.2)
        # Full verification
        expected = np.array(ISAACLAB_TO_NATIVE_INDICES, dtype=np.float64) * 0.1
        np.testing.assert_allclose(obs, expected)


# ---------------------------------------------------------------------------
# Partial control tests
# ---------------------------------------------------------------------------


class TestPartialControl:
    def test_partial_control_12_legs(self):
        """12 leg joints controlled, 17 non-controlled."""
        legs = G1_29DOF_JOINTS[:12]
        m = JointMapper(G1_29DOF_JOINTS, controlled_joints=legs)
        assert m.n_controlled == 12
        assert len(m.non_controlled_indices) == 17

    def test_partial_control_7_arm(self):
        """7 right arm joints controlled."""
        right_arm = G1_29DOF_JOINTS[22:29]
        m = JointMapper(G1_29DOF_JOINTS, controlled_joints=right_arm)
        assert m.n_controlled == 7
        assert len(m.non_controlled_indices) == 22


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_joint_raises(self):
        """Unknown joint name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            JointMapper(G1_29DOF_JOINTS, controlled_joints=["fake_joint"])

    def test_duplicate_joint_raises(self):
        """Duplicate joint in controlled list raises ValueError."""
        with pytest.raises(ValueError, match="Duplicate"):
            JointMapper(
                G1_29DOF_JOINTS,
                controlled_joints=["left_hip_pitch", "left_hip_pitch"],
            )

    def test_empty_controlled_raises(self):
        """Empty controlled list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            JointMapper(G1_29DOF_JOINTS, controlled_joints=[])


# ---------------------------------------------------------------------------
# 23-DOF tests
# ---------------------------------------------------------------------------


class Test23DOF:
    def test_23dof_joint_mapper_basic(self):
        """JointMapper with G1_23DOF_JOINTS: n_total == 23, basic ops work."""
        m = JointMapper(G1_23DOF_JOINTS)
        assert m.n_total == 23
        assert m.n_observed == 23
        assert m.n_controlled == 23
        robot_vals = np.arange(23, dtype=np.float64)
        obs = m.robot_to_observation(robot_vals)
        np.testing.assert_array_equal(obs, robot_vals)
        action = m.robot_to_action(robot_vals)
        np.testing.assert_array_equal(action, robot_vals)
