"""Shared pytest fixtures for the unitree_launcher test suite."""
from pathlib import Path

import numpy as np
import pytest

from unitree_launcher.compat import patch_unitree_threading
patch_unitree_threading()  # Must be called before any SDK imports

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
