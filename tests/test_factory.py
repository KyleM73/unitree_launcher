"""Tests for policy factory (load_policy, load_default_policy, preload_policy_dir)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import create_beyondmimic_onnx, create_isaaclab_onnx

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    ISAACLAB_KP_29DOF,
    load_config,
)
from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy
from unitree_launcher.policy.factory import (
    _apply_gain_overrides,
    _expand_gain,
    load_default_policy,
    load_policy,
    preload_policy_dir,
)
from unitree_launcher.policy.hold_policy import HoldPolicy
from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
from unitree_launcher.policy.joint_mapper import JointMapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs" / "default.yaml")


# ============================================================================
# Helpers
# ============================================================================

def _make_isaaclab_onnx(path: str, obs_dim: int = 99, n_actions: int = 29):
    create_isaaclab_onnx(obs_dim, n_actions, path)


def _make_bm_onnx(path: str, obs_dim: int = 160, n_actions: int = 29):
    bm_metadata = {
        "joint_names": ",".join(ISAACLAB_G1_29DOF_JOINTS),
        "joint_stiffness": ",".join(["40.0"] * n_actions),
        "joint_damping": ",".join(["2.5"] * n_actions),
        "action_scale": ",".join(["0.5"] * n_actions),
        "default_joint_pos": ",".join(["0.0"] * n_actions),
        "anchor_body_name": "torso_link",
        "body_names": "pelvis,torso_link,left_knee_link",
        "observation_names": "command,motion_anchor_pos_b,motion_anchor_ori_b,"
                             "base_lin_vel,base_ang_vel,joint_pos,joint_vel,actions",
    }
    create_beyondmimic_onnx(obs_dim, n_actions, n_actions, path, metadata=bm_metadata)


# ============================================================================
# load_policy — IsaacLab
# ============================================================================

class TestLoadPolicyIsaacLab:
    def test_loads_isaaclab_onnx(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        onnx_path = str(tmp_path / "test_il.onnx")
        _make_isaaclab_onnx(onnx_path)

        policy, mapper = load_policy(onnx_path, config)
        assert isinstance(policy, IsaacLabPolicy)
        assert mapper.n_policy == 29
        assert mapper.n_robot == 29

    def test_isaaclab_preserves_gains_when_config_none(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        assert config.control.kp is None
        onnx_path = str(tmp_path / "test_il.onnx")
        _make_isaaclab_onnx(onnx_path)

        policy, _ = load_policy(onnx_path, config)
        # Should have IsaacLab training gains, not overridden
        assert policy._kp[0] == pytest.approx(ISAACLAB_KP_29DOF["left_hip_pitch"])


# ============================================================================
# load_policy — BeyondMimic
# ============================================================================

class TestLoadPolicyBeyondMimic:
    def test_loads_beyondmimic_onnx(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        onnx_path = str(tmp_path / "test_bm.onnx")
        _make_bm_onnx(onnx_path)

        policy, mapper = load_policy(onnx_path, config)
        assert isinstance(policy, BeyondMimicPolicy)
        assert mapper.n_policy == 29


# ============================================================================
# Gain overrides
# ============================================================================

class TestGainOverrides:
    def test_scalar_kp_override(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        config.control.kp = 5.0
        onnx_path = str(tmp_path / "test_il.onnx")
        _make_isaaclab_onnx(onnx_path)

        policy, _ = load_policy(onnx_path, config)
        np.testing.assert_array_equal(policy._kp, 5.0)

    def test_list_kp_override(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        config.control.kp = [float(i) for i in range(29)]
        onnx_path = str(tmp_path / "test_il.onnx")
        _make_isaaclab_onnx(onnx_path)

        policy, _ = load_policy(onnx_path, config)
        np.testing.assert_array_equal(policy._kp, np.arange(29, dtype=np.float64))

    def test_kd_override(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        config.control.kd = 3.0
        onnx_path = str(tmp_path / "test_il.onnx")
        _make_isaaclab_onnx(onnx_path)

        policy, _ = load_policy(onnx_path, config)
        np.testing.assert_array_equal(policy._kd, 3.0)

    def test_ka_override(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        config.control.ka = 0.1
        onnx_path = str(tmp_path / "test_il.onnx")
        _make_isaaclab_onnx(onnx_path)

        policy, _ = load_policy(onnx_path, config)
        np.testing.assert_array_equal(policy._action_scale, 0.1)

    def test_none_preserves_onnx_metadata(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        assert config.control.kp is None
        onnx_path = str(tmp_path / "test_bm.onnx")
        _make_bm_onnx(onnx_path)

        policy, _ = load_policy(onnx_path, config)
        # BM ONNX metadata sets stiffness=40.0 for all joints
        np.testing.assert_array_equal(policy._kp, 40.0)


# ============================================================================
# load_default_policy
# ============================================================================

class TestLoadDefaultPolicy:
    def test_falls_back_to_hold_policy(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        config.policy.default_policy = str(tmp_path / "nonexistent.onnx")

        policy, mapper = load_default_policy(config)
        assert isinstance(policy, HoldPolicy)
        assert mapper.n_robot == 29

    def test_no_default_path_returns_hold(self):
        config = load_config(DEFAULT_CONFIG)
        config.policy.default_policy = None

        policy, mapper = load_default_policy(config)
        assert isinstance(policy, HoldPolicy)

    def test_loads_valid_default_policy(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        onnx_path = str(tmp_path / "default.onnx")
        _make_isaaclab_onnx(onnx_path)
        config.policy.default_policy = onnx_path

        policy, mapper = load_default_policy(config)
        assert isinstance(policy, IsaacLabPolicy)


# ============================================================================
# preload_policy_dir
# ============================================================================

class TestPreloadPolicyDir:
    def test_loads_multiple_onnx_files(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        _make_isaaclab_onnx(str(tmp_path / "a.onnx"))
        _make_isaaclab_onnx(str(tmp_path / "b.onnx"))

        result = preload_policy_dir(config, str(tmp_path))
        assert len(result) == 2

    def test_excludes_specified_paths(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _make_isaaclab_onnx(path_a)
        _make_isaaclab_onnx(path_b)

        result = preload_policy_dir(config, str(tmp_path), exclude={path_a})
        assert len(result) == 1
        assert path_a not in result
        assert path_b in result

    def test_handles_bad_onnx_gracefully(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        _make_isaaclab_onnx(str(tmp_path / "good.onnx"))
        # Write garbage to bad.onnx
        (tmp_path / "bad.onnx").write_text("not an onnx file")

        result = preload_policy_dir(config, str(tmp_path))
        assert len(result) == 1  # Only the good one loaded

    def test_empty_dir_returns_empty(self, tmp_path):
        config = load_config(DEFAULT_CONFIG)
        result = preload_policy_dir(config, str(tmp_path))
        assert result == {}


# ============================================================================
# _expand_gain helper
# ============================================================================

class TestExpandGain:
    def test_scalar_expands(self):
        result = _expand_gain(5.0, 3)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_list_passes_through(self):
        result = _expand_gain([1.0, 2.0, 3.0], 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_int_scalar(self):
        result = _expand_gain(10, 2)
        np.testing.assert_array_equal(result, [10.0, 10.0])
