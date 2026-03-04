"""Tests for config.py — constants, dataclasses, and config loading."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    G1_23DOF_JOINTS,
    G1_29DOF_MUJOCO_JOINTS,
    G1_23DOF_MUJOCO_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    ISAACLAB_TO_NATIVE_INDICES,
    Q_HOME_29DOF,
    Q_HOME_23DOF,
    JOINT_LIMITS_29DOF,
    JOINT_LIMITS_23DOF,
    TORQUE_LIMITS_29DOF,
    TORQUE_LIMITS_23DOF,
    Config,
    ControlConfig,
    LoggingConfig,
    PolicyConfig,
    RobotConfig,
    ViewerConfig,
    load_config,
    merge_configs,
    resolve_joint_name,
)
from unitree_launcher.robot.base import RobotCommand, RobotState

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Part 1: Robot Constants
# ============================================================================

class TestJointCounts:
    def test_29dof_joint_count(self):
        assert len(G1_29DOF_JOINTS) == 29

    def test_23dof_joint_count(self):
        assert len(G1_23DOF_JOINTS) == 23

    def test_29dof_names_no_joint_suffix(self):
        for name in G1_29DOF_JOINTS:
            assert not name.endswith("_joint"), f"{name} has unexpected _joint suffix"


class TestHomePositionKeys:
    def test_29dof_home_position_keys_match(self):
        assert set(Q_HOME_29DOF.keys()) == set(G1_29DOF_JOINTS)

    def test_23dof_home_position_keys_match(self):
        assert set(Q_HOME_23DOF.keys()) == set(G1_23DOF_JOINTS)


class TestJointLimitsKeys:
    def test_joint_limits_keys_match(self):
        assert set(JOINT_LIMITS_29DOF.keys()) == set(G1_29DOF_JOINTS)

    def test_23dof_joint_limits_keys_match(self):
        assert set(JOINT_LIMITS_23DOF.keys()) == set(G1_23DOF_JOINTS)

    def test_torque_limits_keys_match(self):
        assert set(TORQUE_LIMITS_29DOF.keys()) == set(G1_29DOF_JOINTS)


class TestHomeWithinLimits:
    def test_home_position_within_limits(self):
        for joint in G1_29DOF_JOINTS:
            home = Q_HOME_29DOF[joint]
            lo, hi = JOINT_LIMITS_29DOF[joint]
            assert lo <= home <= hi, (
                f"{joint}: home={home} not in [{lo}, {hi}]"
            )

    def test_23dof_home_within_limits(self):
        for joint in G1_23DOF_JOINTS:
            home = Q_HOME_23DOF[joint]
            lo, hi = JOINT_LIMITS_23DOF[joint]
            assert lo <= home <= hi, (
                f"{joint}: home={home} not in [{lo}, {hi}]"
            )


class TestIsaacLabMapping:
    def test_isaaclab_index_mapping_length(self):
        assert len(ISAACLAB_TO_NATIVE_INDICES) == 29

    def test_isaaclab_index_mapping_bijective(self):
        assert sorted(ISAACLAB_TO_NATIVE_INDICES) == list(range(29))

    def test_isaaclab_index_mapping_known_values(self):
        # Verify specific mappings for known joint indices
        assert ISAACLAB_TO_NATIVE_INDICES[0] == 0   # left_hip_pitch
        assert ISAACLAB_TO_NATIVE_INDICES[1] == 6   # right_hip_pitch
        assert ISAACLAB_TO_NATIVE_INDICES[2] == 12  # waist_yaw
        assert ISAACLAB_TO_NATIVE_INDICES[9] == 3   # left_knee
        assert ISAACLAB_TO_NATIVE_INDICES[21] == 18  # left_elbow
        assert ISAACLAB_TO_NATIVE_INDICES[22] == 25  # right_elbow

    def test_isaaclab_joint_names_length(self):
        assert len(ISAACLAB_G1_29DOF_JOINTS) == 29

    def test_isaaclab_names_have_joint_suffix(self):
        for name in ISAACLAB_G1_29DOF_JOINTS:
            assert name.endswith("_joint"), f"{name} missing _joint suffix"


class TestMujocoJointNames:
    def test_mujoco_joint_name_mapping_complete(self):
        assert set(G1_29DOF_MUJOCO_JOINTS.keys()) == set(G1_29DOF_JOINTS)

    def test_23dof_mujoco_joint_name_mapping_complete(self):
        assert set(G1_23DOF_MUJOCO_JOINTS.keys()) == set(G1_23DOF_JOINTS)


# ============================================================================
# Part 2: Dataclasses
# ============================================================================

class TestRobotState:
    def test_robot_state_zeros(self):
        s = RobotState.zeros(29)
        assert s.joint_positions.shape == (29,)
        assert s.joint_velocities.shape == (29,)
        assert s.joint_torques.shape == (29,)
        assert s.imu_quaternion.shape == (4,)
        assert s.imu_angular_velocity.shape == (3,)
        assert s.imu_linear_acceleration.shape == (3,)
        assert s.base_position.shape == (3,)
        assert s.base_velocity.shape == (3,)
        assert s.timestamp == 0.0
        np.testing.assert_array_equal(s.joint_positions, 0.0)
        np.testing.assert_array_equal(s.imu_quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_robot_state_copy(self):
        s = RobotState.zeros(29)
        s.joint_positions[0] = 1.23
        c = s.copy()
        c.joint_positions[0] = 9.99
        assert s.joint_positions[0] == 1.23, "Copy should not affect original"

    def test_robot_state_nan_base(self):
        s = RobotState.zeros(29)
        s.base_position = np.full(3, np.nan)
        s.base_velocity = np.full(3, np.nan)
        # Should not raise
        _ = s.copy()
        _ = s.base_position
        _ = s.base_velocity


class TestRobotCommand:
    def test_robot_command_damping(self):
        c = RobotCommand.damping(29, kd=5.0)
        assert c.joint_positions.shape == (29,)
        np.testing.assert_array_equal(c.kp, 0.0)
        np.testing.assert_array_equal(c.kd, 5.0)
        np.testing.assert_array_equal(c.joint_positions, 0.0)
        np.testing.assert_array_equal(c.joint_velocities, 0.0)
        np.testing.assert_array_equal(c.joint_torques, 0.0)


# ============================================================================
# Part 3: Joint Name Resolution
# ============================================================================

class TestJointNameResolution:
    def test_joint_name_resolution_config_name(self):
        assert resolve_joint_name("left_hip_pitch", "g1_29dof") == "left_hip_pitch"

    def test_joint_name_resolution_mujoco_name(self):
        assert resolve_joint_name("left_hip_pitch_joint", "g1_29dof") == "left_hip_pitch"

    def test_joint_name_resolution_dds_name(self):
        assert resolve_joint_name("L_LEG_HIP_PITCH", "g1_29dof") == "left_hip_pitch"

    def test_joint_name_resolution_unknown_raises(self):
        with pytest.raises(ValueError, match="nonexistent_joint"):
            resolve_joint_name("nonexistent_joint", "g1_29dof")

    def test_joint_name_resolution_23dof(self):
        assert resolve_joint_name("torso", "g1_23dof") == "torso"
        assert resolve_joint_name("TORSO", "g1_23dof") == "torso"
        assert resolve_joint_name("waist_yaw_joint", "g1_23dof") == "torso"


# ============================================================================
# Part 4: Config Loading
# ============================================================================

class TestConfigLoading:
    def test_load_default_config(self):
        cfg = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert cfg.robot.variant == "g1_29dof"
        assert cfg.control.policy_frequency == 50
        assert cfg.control.sim_frequency == 500
        assert cfg.control.kp is None
        assert cfg.control.kd is None
        assert cfg.control.ka is None
        assert cfg.safety.joint_position_limits is True
        assert cfg.logging.format == "hdf5"

    def test_load_robot_specific_config(self):
        base = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        override = load_config(str(PROJECT_ROOT / "configs" / "g1_29dof.yaml"))
        merged = merge_configs(base, override)
        assert merged.robot.variant == "g1_29dof"
        # Base values preserved where override doesn't set them
        assert merged.control.kp is None

    def test_load_23dof_config(self):
        base = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        override = load_config(str(PROJECT_ROOT / "configs" / "g1_23dof.yaml"))
        merged = merge_configs(base, override)
        assert merged.robot.variant == "g1_23dof"


class TestConfigValidation:
    def _write_yaml(self, data: dict) -> str:
        """Write a dict to a temp YAML file and return the path."""
        import yaml
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        return path

    def test_invalid_variant_rejected(self):
        path = self._write_yaml({"robot": {"variant": "g1_99dof"}})
        with pytest.raises(ValueError, match="variant"):
            load_config(path)

    def test_invalid_joint_name_rejected(self):
        path = self._write_yaml({
            "robot": {"variant": "g1_29dof"},
            "policy": {"controlled_joints": ["nonexistent_joint"]},
        })
        with pytest.raises(ValueError, match="nonexistent_joint"):
            load_config(path)

    def test_kp_list_wrong_length_rejected(self):
        path = self._write_yaml({
            "control": {"kp": [1.0, 2.0, 3.0]},  # 3 != 29
        })
        with pytest.raises(ValueError, match="kp"):
            load_config(path)

    def test_ka_list_wrong_length_rejected(self):
        path = self._write_yaml({
            "control": {"ka": [0.1, 0.2]},  # 2 != 29
        })
        with pytest.raises(ValueError, match="ka"):
            load_config(path)

    def test_frequency_divisibility(self):
        # OK: 200 / 50 = 4
        path = self._write_yaml({
            "control": {"sim_frequency": 200, "policy_frequency": 50},
        })
        cfg = load_config(path)
        assert cfg.control.sim_frequency == 200

        # Bad: 200 / 60 != integer
        path2 = self._write_yaml({
            "control": {"sim_frequency": 200, "policy_frequency": 60},
        })
        with pytest.raises(ValueError, match="divisible"):
            load_config(path2)

    def test_idl_mode_validation(self):
        path = self._write_yaml({"robot": {"idl_mode": 2}})
        with pytest.raises(ValueError, match="idl_mode"):
            load_config(path)

    def test_logging_format_validation(self):
        path = self._write_yaml({"logging": {"format": "invalid"}})
        with pytest.raises(ValueError, match="format"):
            load_config(path)

    def test_default_gains_are_none(self):
        cfg = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        assert cfg.control.kp is None
        assert cfg.control.kd is None
        assert cfg.control.ka is None

    def test_viewer_config_defaults(self):
        cfg = Config()
        assert cfg.viewer.enabled is True
        assert cfg.viewer.sync is True


class TestMergeConfigs:
    def test_merge_configs_none_preserves_base(self):
        base = Config(control=ControlConfig(kp=100.0))
        override = Config()
        # ControlConfig kp defaults to None, PolicyConfig format defaults to None.
        override.policy = PolicyConfig(format=None, controlled_joints=None)
        merged = merge_configs(base, override)
        # base policy.format was None, override is None -> stays None
        assert merged.policy.format is None
        # base control.kp was 100.0, override is None -> stays 100.0
        assert merged.control.kp == 100.0

    def test_cli_override_merges(self):
        base = Config(control=ControlConfig(kp=100.0, kd=10.0))
        override = Config(control=ControlConfig(kp=200.0))
        merged = merge_configs(base, override)
        assert merged.control.kp == 200.0
        assert merged.control.kd == 10.0  # preserved from base
