"""Shared pytest fixtures for the unitree_launcher test suite."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Phase 3 will add: from src.compat import patch_unitree_threading
# patch_unitree_threading()  # Must be called before any SDK imports

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Joint name fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def g1_29dof_joint_names():
    """29 config-style joint names in robot-native order."""
    return [
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_roll",
        "left_wrist_pitch",
        "left_wrist_yaw",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
    ]


@pytest.fixture
def g1_23dof_joint_names():
    """23 config-style joint names in robot-native order."""
    return [
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "torso",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow_pitch",
        "left_elbow_roll",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow_pitch",
        "right_elbow_roll",
    ]


@pytest.fixture
def isaaclab_29dof_joint_names():
    """29 MuJoCo-style joint names in IsaacLab order."""
    return [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures (will be expanded in Phase 2 when dataclasses exist)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_robot_state_dict():
    """Plausible standing robot state as raw arrays (pre-dataclass).

    Will be replaced by a proper RobotState fixture in Phase 2.
    """
    n_dof = 29
    return {
        "timestamp": 0.0,
        "joint_positions": np.array([
            -0.312, 0.0, 0.0, 0.669, -0.33, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.33, 0.0,   # right leg
            0.0, 0.0, 0.0,                           # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,    # right arm
        ]),
        "joint_velocities": np.zeros(n_dof),
        "joint_torques": np.zeros(n_dof),
        "imu_quaternion": np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz, upright
        "imu_angular_velocity": np.zeros(3),
        "imu_linear_acceleration": np.array([0.0, 0.0, 9.81]),
        "base_position": np.array([0.0, 0.0, 0.793]),
        "base_velocity": np.zeros(3),
    }


# ---------------------------------------------------------------------------
# Model path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mujoco_model_path_29dof():
    """Path to the 29-DOF MJCF file."""
    return str(PROJECT_ROOT / "assets" / "robots" / "g1" / "g1_29dof.xml")


@pytest.fixture
def mujoco_model_path_23dof():
    """Path to the 23-DOF MJCF file."""
    return str(PROJECT_ROOT / "assets" / "robots" / "g1" / "g1_23dof.xml")


# ---------------------------------------------------------------------------
# Temporary directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_log_dir(tmp_path):
    """Temporary directory for log output, cleaned up after test."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)


# ---------------------------------------------------------------------------
# ONNX model creation helpers (used by Phase 6 policy tests)
# ---------------------------------------------------------------------------

def create_isaaclab_onnx(obs_dim, action_dim, path):
    """Create a minimal ONNX model: Input 'obs' [1, obs_dim] -> Output 'action' [1, action_dim].

    The model outputs zeros regardless of input (identity-like for testing).
    """
    import onnx
    from onnx import TensorProto, helper

    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, action_dim])

    # Constant zero output
    zero_init = onnx.numpy_helper.from_array(
        np.zeros((1, action_dim), dtype=np.float32), name="zero_action"
    )
    identity_node = helper.make_node("Identity", inputs=["zero_action"], outputs=["action"])

    graph = helper.make_graph(
        [identity_node],
        "isaaclab_test",
        [obs],
        [action],
        initializer=[zero_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def create_beyondmimic_onnx(obs_dim, action_dim, n_joints, path, metadata=None):
    """Create a minimal BeyondMimic-style ONNX model.

    Inputs: 'obs' [1, obs_dim], 'time_step' [1, 1]
    Outputs: 'action' [1, action_dim], 'target_q' [1, n_joints], 'target_dq' [1, n_joints]
    Metadata embedded as model properties.
    """
    import onnx
    from onnx import TensorProto, helper

    obs_input = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])
    time_input = helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])

    action_output = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, action_dim])
    target_q_output = helper.make_tensor_value_info("target_q", TensorProto.FLOAT, [1, n_joints])
    target_dq_output = helper.make_tensor_value_info("target_dq", TensorProto.FLOAT, [1, n_joints])

    # Constant zero outputs
    zero_action = onnx.numpy_helper.from_array(
        np.zeros((1, action_dim), dtype=np.float32), name="zero_action"
    )
    zero_q = onnx.numpy_helper.from_array(
        np.zeros((1, n_joints), dtype=np.float32), name="zero_q"
    )
    zero_dq = onnx.numpy_helper.from_array(
        np.zeros((1, n_joints), dtype=np.float32), name="zero_dq"
    )

    nodes = [
        helper.make_node("Identity", inputs=["zero_action"], outputs=["action"]),
        helper.make_node("Identity", inputs=["zero_q"], outputs=["target_q"]),
        helper.make_node("Identity", inputs=["zero_dq"], outputs=["target_dq"]),
    ]

    graph = helper.make_graph(
        nodes,
        "beyondmimic_test",
        [obs_input, time_input],
        [action_output, target_q_output, target_dq_output],
        initializer=[zero_action, zero_q, zero_dq],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    if metadata:
        for key, value in metadata.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = str(value)

    onnx.save(model, path)
