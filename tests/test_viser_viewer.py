"""Tests for the viser web viewer and MuJoCo-to-trimesh conversions."""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest
import trimesh

from unitree_launcher.viz.conversions import (
    build_body_meshes,
    create_primitive_mesh,
    mat_to_quat_wxyz,
    mujoco_mesh_to_trimesh,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = str(PROJECT_ROOT / "assets" / "robots" / "g1" / "scene_29dof.xml")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def mj_model():
    """Load the G1 29-DOF MuJoCo model once for all conversion tests."""
    return mujoco.MjModel.from_xml_path(SCENE_XML)


# ============================================================================
# Conversion Tests (no viser server needed)
# ============================================================================

class TestMujocoMeshToTrimesh:
    def test_pelvis_mesh_valid(self, mj_model):
        """Pelvis visual geom -> trimesh with correct vertex/face counts."""
        # geom[1] is the first pelvis mesh geom (type=7, group=2)
        mesh = mujoco_mesh_to_trimesh(mj_model, 1)
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_mesh_has_vertex_colors(self, mj_model):
        """Converted mesh should have RGBA vertex colors from geom_rgba."""
        mesh = mujoco_mesh_to_trimesh(mj_model, 1)
        colors = mesh.visual.vertex_colors
        assert colors is not None
        assert colors.shape == (len(mesh.vertices), 4)


class TestCreatePrimitiveMesh:
    def test_capsule(self, mj_model):
        """A capsule collision geom creates a valid trimesh."""
        # Find first capsule geom (type 3)
        capsule_id = None
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == 3 and mj_model.geom_bodyid[i] > 0:
                capsule_id = i
                break
        assert capsule_id is not None, "No capsule geom found in model"
        mesh = create_primitive_mesh(mj_model, capsule_id)
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0

    def test_cylinder(self, mj_model):
        """A cylinder collision geom creates a valid trimesh."""
        cyl_id = None
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == 5 and mj_model.geom_bodyid[i] > 0:
                cyl_id = i
                break
        assert cyl_id is not None, "No cylinder geom found in model"
        mesh = create_primitive_mesh(mj_model, cyl_id)
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0


class TestBuildBodyMeshes:
    def test_all_dynamic_bodies_get_meshes(self, mj_model):
        """Every dynamic body with visual geoms should have a mesh."""
        body_meshes = build_body_meshes(mj_model)
        assert len(body_meshes) > 0
        # World body (id=0) should not be included.
        assert 0 not in body_meshes
        # Pelvis (id=1) should be included.
        assert 1 in body_meshes

    def test_mesh_count_matches_bodies_with_visual_geoms(self, mj_model):
        """The number of meshes should match bodies that have visual geoms."""
        body_meshes = build_body_meshes(mj_model)
        # G1 has 30 dynamic bodies (1..30), all should have visual meshes.
        assert len(body_meshes) == 30

    def test_merge_applies_local_transform(self, mj_model):
        """Geom offsets should be baked into the merged mesh vertices."""
        body_meshes = build_body_meshes(mj_model)
        # Pelvis has multiple geoms with non-zero local positions.
        pelvis_mesh = body_meshes[1]
        # Vertices should not all be at origin.
        assert not np.allclose(pelvis_mesh.vertices, 0.0)

    def test_visual_only_excludes_collision(self, mj_model):
        """With visual_only=True, collision geoms (group 3) are excluded."""
        visual_meshes = build_body_meshes(mj_model, visual_only=True)
        all_meshes = build_body_meshes(mj_model, visual_only=False)
        # With collision included, total vertex count should be larger.
        visual_verts = sum(len(m.vertices) for m in visual_meshes.values())
        all_verts = sum(len(m.vertices) for m in all_meshes.values())
        assert all_verts > visual_verts


class TestMatToQuatWxyz:
    def test_identity(self):
        """Identity matrix -> [1, 0, 0, 0]."""
        q = mat_to_quat_wxyz(np.eye(3))
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_90deg_z_rotation(self):
        """90-degree rotation about Z -> expected quaternion."""
        Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        q = mat_to_quat_wxyz(Rz90)
        expected = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        np.testing.assert_allclose(q, expected, atol=1e-10)

    def test_180deg_x_rotation(self):
        """180-degree rotation about X."""
        Rx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        q = mat_to_quat_wxyz(Rx180)
        # w should be ~0, x should be ~1
        assert abs(q[0]) < 1e-10
        assert abs(abs(q[1]) - 1.0) < 1e-10


# ============================================================================
# ViserViewer Class Tests (mocked viser server)
# ============================================================================

class TestViserViewerKeyQueue:
    def test_drain_key_queue_empty(self):
        """drain_key_queue on empty queue returns empty list."""
        from unitree_launcher.viz.viser_viewer import ViserViewer

        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        viewer = ViserViewer(model, port=0)
        assert viewer.drain_key_queue() == []

    def test_drain_key_queue_ordered(self):
        """Keys are returned in FIFO order."""
        from unitree_launcher.viz.viser_viewer import ViserViewer

        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        viewer = ViserViewer(model, port=0)
        viewer._key_queue.put("space")
        viewer._key_queue.put("backspace")
        viewer._key_queue.put("enter")
        keys = viewer.drain_key_queue()
        assert keys == ["space", "backspace", "enter"]
        assert viewer.drain_key_queue() == []

    def test_drain_key_queue_thread_safe(self):
        """Keys put from another thread are drained correctly."""
        from unitree_launcher.viz.viser_viewer import ViserViewer

        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        viewer = ViserViewer(model, port=0)

        def put_keys():
            for k in ["up", "down", "left"]:
                viewer._key_queue.put(k)

        t = threading.Thread(target=put_keys)
        t.start()
        t.join()
        keys = viewer.drain_key_queue()
        assert keys == ["up", "down", "left"]


class TestViserViewerUpdate:
    def test_update_snapshots_under_lock(self):
        """Lock should be acquired during xpos/xmat copy."""
        from unitree_launcher.viz.viser_viewer import ViserViewer

        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        viewer = ViserViewer(model, port=0)
        # Mock server and body handles so update() doesn't fail.
        mock_server = MagicMock()
        mock_server.atomic.return_value.__enter__ = MagicMock()
        mock_server.atomic.return_value.__exit__ = MagicMock(return_value=False)
        viewer._server = mock_server
        viewer._body_handles = {}  # No handles = skip inner loop

        # Use a tracking lock that records whether it was acquired.
        real_lock = threading.Lock()
        lock_acquired = []

        class TrackingLock:
            """Wraps a real lock, recording when __enter__/__exit__ are called."""
            def __enter__(self):
                real_lock.acquire()
                lock_acquired.append(True)
                return self

            def __exit__(self, *args):
                real_lock.release()

        viewer.update(data, TrackingLock())

        assert any(lock_acquired), "Lock must be acquired during update()"

    def test_update_releases_lock_before_viser_send(self):
        """Lock should NOT be held during viser handle updates."""
        from unitree_launcher.viz.viser_viewer import ViserViewer

        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        viewer = ViserViewer(model, port=0)
        lock = threading.Lock()

        # Create a mock handle that checks lock state.
        lock_held_during_update = []

        class MockHandle:
            @property
            def position(self):
                return np.zeros(3)

            @position.setter
            def position(self, val):
                lock_held_during_update.append(lock.locked())

            @property
            def wxyz(self):
                return np.array([1, 0, 0, 0])

            @wxyz.setter
            def wxyz(self, val):
                pass

        mock_server = MagicMock()
        mock_server.atomic.return_value.__enter__ = MagicMock()
        mock_server.atomic.return_value.__exit__ = MagicMock(return_value=False)
        viewer._server = mock_server
        viewer._body_handles = {1: MockHandle()}

        viewer.update(data, lock)

        assert lock_held_during_update, "Handle position should have been set"
        assert not any(lock_held_during_update), (
            "Lock must NOT be held during viser handle updates"
        )


# ============================================================================
# CLI Argument Parsing Tests
# ============================================================================

class TestViserCLIArgs:
    def test_parse_viser_default_port(self):
        """--viser uses default port 8080."""
        from unitree_launcher.main import build_parser
        args = build_parser().parse_args(
            ["sim", "--policy", "p.onnx", "--viser"]
        )
        assert args.viser is True
        assert args.port == 8080

    def test_parse_viser_custom_port(self):
        """--viser --port 9090 sets port to 9090."""
        from unitree_launcher.main import build_parser
        args = build_parser().parse_args(
            ["sim", "--policy", "p.onnx", "--viser", "--port", "9090"]
        )
        assert args.viser is True
        assert args.port == 9090

    def test_parse_headless_default(self):
        """Sim defaults to headless (no --gui, no --viser)."""
        from unitree_launcher.main import build_parser
        args = build_parser().parse_args(["sim", "--policy", "p.onnx"])
        assert args.gui is False
        assert args.viser is False

    def test_parse_gui_and_viser_together(self):
        """--gui and --viser can both be set."""
        from unitree_launcher.main import build_parser
        args = build_parser().parse_args(
            ["sim", "--policy", "p.onnx", "--gui", "--viser"]
        )
        assert args.gui is True
        assert args.viser is True

    def test_parse_viser_real_mode(self):
        """--viser works in real mode."""
        from unitree_launcher.main import build_parser
        args = build_parser().parse_args(
            ["real", "--policy", "p.onnx", "--interface", "eth0", "--viser"]
        )
        assert args.viser is True
        assert args.port == 8080


# ============================================================================
# Runner Integration Tests (mocked ViserViewer)
# ============================================================================

class TestRunWithViser:
    def test_run_with_viser_starts_stops_controller(self):
        """run_with_viser starts the controller and stops it on exit."""
        from unitree_launcher.main import run_with_viser

        robot = MagicMock()
        robot.mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
        robot.mj_data = mujoco.MjData(robot.mj_model)
        robot.lock = threading.Lock()

        ctrl = MagicMock()
        ctrl.is_running = False  # Exit immediately
        ctrl.safety = MagicMock()
        ctrl.safety.velocity_command = np.zeros(3)

        mock_viewer = MagicMock()
        mock_viewer.is_running.return_value = True
        mock_viewer.drain_key_queue.return_value = []
        mock_viewer.get_velocity_commands.return_value = None

        with patch(
            "unitree_launcher.viz.viser_viewer.ViserViewer",
            return_value=mock_viewer,
        ):
            run_with_viser(robot, ctrl, port=0)

        ctrl.start.assert_called_once()
        ctrl.stop.assert_called_once()
        mock_viewer.setup.assert_called_once()
        mock_viewer.close.assert_called_once()

    def test_run_with_viser_drains_keys_to_handle_key(self):
        """Keys from viser GUI buttons are forwarded to controller.handle_key."""
        from unitree_launcher.main import run_with_viser

        robot = MagicMock()
        robot.mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
        robot.mj_data = mujoco.MjData(robot.mj_model)
        robot.lock = threading.Lock()

        ctrl = MagicMock()
        ctrl.safety = MagicMock()
        ctrl.safety.velocity_command = np.zeros(3)

        call_count = [0]

        def fake_is_running():
            call_count[0] += 1
            return call_count[0] <= 1  # One iteration

        type(ctrl).is_running = property(lambda self: fake_is_running())

        mock_viewer = MagicMock()
        mock_viewer.is_running.return_value = True
        mock_viewer.drain_key_queue.return_value = ["space", "up"]
        mock_viewer.get_velocity_commands.return_value = None

        with patch(
            "unitree_launcher.viz.viser_viewer.ViserViewer",
            return_value=mock_viewer,
        ):
            run_with_viser(robot, ctrl, port=0)

        ctrl.handle_key.assert_any_call("space")
        ctrl.handle_key.assert_any_call("up")

    def test_run_with_viser_duration_termination(self):
        """With duration=0.1, run exits quickly."""
        from unitree_launcher.main import run_with_viser
        import time

        robot = MagicMock()
        robot.mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
        robot.mj_data = mujoco.MjData(robot.mj_model)
        robot.lock = threading.Lock()

        ctrl = MagicMock()
        ctrl.safety = MagicMock()
        ctrl.safety.velocity_command = np.zeros(3)
        type(ctrl).is_running = property(lambda self: True)

        mock_viewer = MagicMock()
        mock_viewer.is_running.return_value = True
        mock_viewer.drain_key_queue.return_value = []
        mock_viewer.get_velocity_commands.return_value = None

        with patch(
            "unitree_launcher.viz.viser_viewer.ViserViewer",
            return_value=mock_viewer,
        ):
            start = time.time()
            run_with_viser(robot, ctrl, port=0, duration=0.1)
            elapsed = time.time() - start

        assert elapsed < 2.0, f"Should have exited quickly, took {elapsed:.1f}s"
