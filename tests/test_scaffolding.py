"""Phase 1 scaffolding tests — verify fixtures and model files work."""
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest


class TestFixtures:
    """Verify conftest fixtures produce valid data."""

    def test_29dof_joint_count(self, g1_29dof_joint_names):
        assert len(g1_29dof_joint_names) == 29

    def test_23dof_joint_count(self, g1_23dof_joint_names):
        assert len(g1_23dof_joint_names) == 23

    def test_isaaclab_joint_count(self, isaaclab_29dof_joint_names):
        assert len(isaaclab_29dof_joint_names) == 29

    def test_isaaclab_names_have_joint_suffix(self, isaaclab_29dof_joint_names):
        for name in isaaclab_29dof_joint_names:
            assert name.endswith("_joint"), f"{name} missing _joint suffix"

    def test_29dof_names_no_suffix(self, g1_29dof_joint_names):
        for name in g1_29dof_joint_names:
            assert not name.endswith("_joint"), f"{name} has unexpected _joint suffix"

    def test_sample_robot_state_shapes(self, sample_robot_state_dict):
        s = sample_robot_state_dict
        assert s["joint_positions"].shape == (29,)
        assert s["joint_velocities"].shape == (29,)
        assert s["joint_torques"].shape == (29,)
        assert s["imu_quaternion"].shape == (4,)
        assert s["imu_angular_velocity"].shape == (3,)
        assert s["imu_linear_acceleration"].shape == (3,)
        assert s["base_position"].shape == (3,)
        assert s["base_velocity"].shape == (3,)

    def test_sample_robot_state_quaternion_unit(self, sample_robot_state_dict):
        q = sample_robot_state_dict["imu_quaternion"]
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_tmp_log_dir_exists(self, tmp_log_dir):
        assert os.path.isdir(tmp_log_dir)


class TestModelFiles:
    """Verify MuJoCo model files load correctly."""

    def test_29dof_model_exists(self, mujoco_model_path_29dof):
        assert os.path.isfile(mujoco_model_path_29dof)

    def test_23dof_model_exists(self, mujoco_model_path_23dof):
        assert os.path.isfile(mujoco_model_path_23dof)

    def test_29dof_model_loads(self, mujoco_model_path_29dof):
        model = mujoco.MjModel.from_xml_path(mujoco_model_path_29dof)
        assert model.nu == 29, f"Expected 29 actuators, got {model.nu}"

    def test_23dof_model_loads(self, mujoco_model_path_23dof):
        model = mujoco.MjModel.from_xml_path(mujoco_model_path_23dof)
        assert model.nu > 0

    def test_29dof_model_steps(self, mujoco_model_path_29dof):
        model = mujoco.MjModel.from_xml_path(mujoco_model_path_29dof)
        data = mujoco.MjData(model)
        initial_height = data.qpos[2]
        for _ in range(200):
            mujoco.mj_step(model, data)
        # Robot should fall under gravity (no control)
        assert data.qpos[2] < initial_height, "Robot should fall with no control"

    def test_meshes_directory_populated(self):
        meshes = Path(__file__).parent.parent / "assets" / "robots" / "g1" / "meshes"
        stl_files = list(meshes.glob("*.STL"))
        assert len(stl_files) >= 20, f"Expected >=20 mesh files, got {len(stl_files)}"


class TestOnnxHelpers:
    """Verify ONNX model creation helpers work."""

    def test_create_isaaclab_onnx(self, tmp_path):
        from tests.conftest import create_isaaclab_onnx
        path = str(tmp_path / "test_isaaclab.onnx")
        create_isaaclab_onnx(obs_dim=47, action_dim=29, path=path)
        assert os.path.isfile(path)

        import onnxruntime as ort
        sess = ort.InferenceSession(path)
        assert sess.get_inputs()[0].name == "obs"
        assert sess.get_outputs()[0].name == "action"
        result = sess.run(None, {"obs": np.zeros((1, 47), dtype=np.float32)})
        assert result[0].shape == (1, 29)

    def test_create_beyondmimic_onnx(self, tmp_path):
        from tests.conftest import create_beyondmimic_onnx
        path = str(tmp_path / "test_bm.onnx")
        create_beyondmimic_onnx(
            obs_dim=100, action_dim=29, n_joints=29, path=path,
            metadata={"format": "beyondmimic", "version": "1.0"},
        )
        assert os.path.isfile(path)

        import onnxruntime as ort
        sess = ort.InferenceSession(path)
        input_names = [i.name for i in sess.get_inputs()]
        assert "obs" in input_names
        assert "time_step" in input_names
        output_names = [o.name for o in sess.get_outputs()]
        assert "action" in output_names
        assert "target_q" in output_names
        assert "target_dq" in output_names
