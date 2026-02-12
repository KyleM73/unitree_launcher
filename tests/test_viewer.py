"""Tests for src/main.py — Phase 9.

Covers GLFW key map, key callback dispatch (queue-based), run_with_viewer
(mocked viewer with lock + queue), and run_headless (duration, step, and
trajectory-end termination).
"""
from __future__ import annotations

import os
import queue
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.config import (
    Config,
    G1_29DOF_JOINTS,
    Q_HOME_29DOF,
    load_config,
)
from src.control.controller import Controller
from src.control.safety import SafetyController, SystemState
from src.main import GLFW_KEY_MAP, run_headless, run_with_viewer
from src.policy.base import PolicyInterface
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from src.robot.base import RobotCommand, RobotInterface, RobotState

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Helpers
# ============================================================================

def _make_config() -> Config:
    return load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))


def _make_mock_robot(n_dof: int = 29) -> MagicMock:
    robot = MagicMock(spec=RobotInterface)
    robot.n_dof = n_dof
    home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
    robot.get_state.return_value = RobotState(
        timestamp=0.0,
        joint_positions=home.copy(),
        joint_velocities=np.zeros(n_dof),
        joint_torques=np.zeros(n_dof),
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_angular_velocity=np.zeros(3),
        imu_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        base_position=np.array([0.0, 0.0, 0.793]),
        base_velocity=np.zeros(3),
    )
    return robot


def _make_mock_policy(n_ctrl: int = 29) -> MagicMock:
    policy = MagicMock(spec=PolicyInterface)
    policy.get_action.return_value = np.zeros(n_ctrl)
    policy.observation_dim = 70
    policy.action_dim = n_ctrl
    return policy


def _make_controller(**kwargs) -> Controller:
    config = kwargs.pop("config", _make_config())
    robot = kwargs.pop("robot", _make_mock_robot())
    policy = kwargs.pop("policy", _make_mock_policy())
    mapper = JointMapper(G1_29DOF_JOINTS)
    safety = SafetyController(config, n_dof=29)
    obs_builder = ObservationBuilder(mapper, config)
    return Controller(
        robot=robot,
        policy=policy,
        safety=safety,
        joint_mapper=mapper,
        obs_builder=obs_builder,
        config=config,
        **kwargs,
    )


# ============================================================================
# GLFW Key Map
# ============================================================================

class TestGLFWKeyMap:
    """Verify GLFW_KEY_MAP maps keycodes to the correct action names.

    Keys are chosen to avoid conflicts with MuJoCo viewer built-in
    shortcuts (which bind most letter keys for rendering toggles).
    """

    def test_glfw_key_map_space(self):
        assert GLFW_KEY_MAP[32] == "space"

    def test_glfw_key_map_up(self):
        assert GLFW_KEY_MAP[265] == "up"

    def test_glfw_key_map_down(self):
        assert GLFW_KEY_MAP[264] == "down"

    def test_glfw_key_map_left(self):
        assert GLFW_KEY_MAP[263] == "left"

    def test_glfw_key_map_right(self):
        assert GLFW_KEY_MAP[262] == "right"

    def test_glfw_key_map_comma(self):
        assert GLFW_KEY_MAP[44] == "comma"

    def test_glfw_key_map_period(self):
        assert GLFW_KEY_MAP[46] == "period"

    def test_glfw_key_map_slash(self):
        assert GLFW_KEY_MAP[47] == "slash"

    def test_glfw_key_map_backspace(self):
        assert GLFW_KEY_MAP[259] == "backspace"

    def test_glfw_key_map_enter(self):
        assert GLFW_KEY_MAP[257] == "enter"

    def test_glfw_key_map_minus(self):
        assert GLFW_KEY_MAP[45] == "minus"

    def test_glfw_key_map_equal(self):
        assert GLFW_KEY_MAP[61] == "equal"

    def test_glfw_key_map_delete(self):
        assert GLFW_KEY_MAP[261] == "delete"

    def test_glfw_key_map_all_values_are_strings(self):
        for keycode, name in GLFW_KEY_MAP.items():
            assert isinstance(keycode, int)
            assert isinstance(name, str)

    def test_glfw_key_map_no_letter_keys(self):
        """No single ASCII letter keys (65-90) — they conflict with MuJoCo viewer."""
        for keycode in GLFW_KEY_MAP:
            assert not (65 <= keycode <= 90), (
                f"Letter key {chr(keycode)} (code {keycode}) conflicts with MuJoCo viewer"
            )


# ============================================================================
# Key Callback Dispatch (queue-based)
# ============================================================================

class TestKeyCallback:
    """The key callback fires on MuJoCo's viewer thread.  To avoid
    cross-thread deadlocks, it enqueues key names into a SimpleQueue.
    The main loop drains the queue outside the lock.
    """

    def test_key_callback_enqueues_mapped_key(self):
        """Pressing space (keycode 32) enqueues 'space'."""
        key_queue = queue.SimpleQueue()

        def key_callback(keycode: int) -> None:
            key = GLFW_KEY_MAP.get(keycode)
            if key:
                key_queue.put(key)

        key_callback(32)
        assert key_queue.get_nowait() == "space"

    def test_key_callback_unmapped_ignored(self):
        """An unmapped keycode should NOT enqueue anything."""
        key_queue = queue.SimpleQueue()

        def key_callback(keycode: int) -> None:
            key = GLFW_KEY_MAP.get(keycode)
            if key:
                key_queue.put(key)

        key_callback(999)
        assert key_queue.empty()

    def test_key_callback_all_mapped_keys(self):
        """Every mapped keycode enqueues the correct name."""
        key_queue = queue.SimpleQueue()

        def key_callback(keycode: int) -> None:
            key = GLFW_KEY_MAP.get(keycode)
            if key:
                key_queue.put(key)

        for keycode, expected_name in GLFW_KEY_MAP.items():
            key_callback(keycode)
            assert key_queue.get_nowait() == expected_name

    def test_main_loop_drains_queue_to_handle_key(self):
        """Simulates the main loop: queue is drained, handle_key called."""
        ctrl = _make_controller()
        ctrl.handle_key = MagicMock()

        key_queue = queue.SimpleQueue()
        key_queue.put("space")
        key_queue.put("up")

        # Simulate main loop drain
        while not key_queue.empty():
            ctrl.handle_key(key_queue.get_nowait())

        assert ctrl.handle_key.call_count == 2
        ctrl.handle_key.assert_any_call("space")
        ctrl.handle_key.assert_any_call("up")


# ============================================================================
# run_with_viewer (mocked)
# ============================================================================

class TestRunWithViewer:
    def test_run_with_viewer_starts_and_stops_controller(self):
        """Viewer loop starts the controller and stops it on exit."""
        ctrl = _make_controller()
        robot = MagicMock()
        robot.mj_model = MagicMock()
        robot.mj_data = MagicMock()
        robot.lock = threading.Lock()

        call_count = 0

        class FakeViewer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self_inner):
                nonlocal call_count
                call_count += 1
                # Run for 2 iterations then stop
                return call_count <= 2

            def sync(self):
                pass

        with patch("src.main.mujoco.viewer.launch_passive", return_value=FakeViewer()):
            with patch.object(ctrl, "start") as mock_start, \
                 patch.object(ctrl, "stop") as mock_stop:
                run_with_viewer(robot, ctrl)
                mock_start.assert_called_once()
                mock_stop.assert_called_once()

    def test_run_with_viewer_key_callback_enqueues(self):
        """Key callback from viewer enqueues to queue (not direct handle_key)."""
        ctrl = _make_controller()
        robot = MagicMock()
        robot.mj_model = MagicMock()
        robot.mj_data = MagicMock()
        robot.lock = threading.Lock()

        captured_callback = None

        class FakeViewer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self):
                return False  # Exit immediately

            def sync(self):
                pass

        def fake_launch_passive(model, data, key_callback=None):
            nonlocal captured_callback
            captured_callback = key_callback
            return FakeViewer()

        with patch("src.main.mujoco.viewer.launch_passive", side_effect=fake_launch_passive):
            with patch.object(ctrl, "start"), patch.object(ctrl, "stop"):
                run_with_viewer(robot, ctrl)

        # The captured callback should exist and enqueue (not call handle_key directly)
        assert captured_callback is not None
        # Calling it should not raise — it just pushes to a queue
        captured_callback(32)  # space
        captured_callback(999)  # unmapped — should be silently ignored

    def test_run_with_viewer_key_callback_ignores_unmapped(self):
        """Unmapped keycodes do not reach handle_key."""
        ctrl = _make_controller()
        robot = MagicMock()
        robot.mj_model = MagicMock()
        robot.mj_data = MagicMock()
        robot.lock = threading.Lock()

        captured_callback = None

        class FakeViewer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self):
                return False

            def sync(self):
                pass

        def fake_launch_passive(model, data, key_callback=None):
            nonlocal captured_callback
            captured_callback = key_callback
            return FakeViewer()

        with patch("src.main.mujoco.viewer.launch_passive", side_effect=fake_launch_passive):
            with patch.object(ctrl, "start"), patch.object(ctrl, "stop"):
                ctrl.handle_key = MagicMock()
                run_with_viewer(robot, ctrl)

        captured_callback(12345)  # unmapped — goes into queue but is discarded
        # handle_key should not have been called (viewer exited before drain)
        ctrl.handle_key.assert_not_called()

    def test_run_with_viewer_sync_under_lock(self):
        """viewer.sync() must be called while holding sim_robot.lock."""
        ctrl = _make_controller()
        robot = MagicMock()
        robot.mj_model = MagicMock()
        robot.mj_data = MagicMock()
        lock = threading.Lock()
        robot.lock = lock

        sync_held_lock = None
        call_count = 0

        class FakeViewer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self_inner):
                nonlocal call_count
                call_count += 1
                return call_count <= 1

            def sync(self_inner):
                nonlocal sync_held_lock
                # Check if lock is held (locked() returns True if acquired)
                sync_held_lock = lock.locked()

        with patch("src.main.mujoco.viewer.launch_passive", return_value=FakeViewer()):
            with patch.object(ctrl, "start"), patch.object(ctrl, "stop"):
                run_with_viewer(robot, ctrl)

        assert sync_held_lock is True, "viewer.sync() must run under sim_robot.lock"


# ============================================================================
# run_headless
# ============================================================================

class TestRunHeadless:
    def test_run_headless_starts_policy(self):
        """run_headless calls controller.start() and safety.start()."""
        ctrl = _make_controller()

        with patch.object(ctrl, "start") as mock_start, \
             patch.object(ctrl, "stop") as mock_stop, \
             patch.object(ctrl.safety, "start") as mock_safety_start:
            # Make is_running return False immediately so headless exits
            ctrl._running = False
            run_headless(MagicMock(), ctrl)

            mock_start.assert_called_once()
            mock_safety_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_run_headless_duration_termination(self):
        """With duration=0.3, run completes within a reasonable time window."""
        ctrl = _make_controller()

        # Prevent the real control loop from running
        with patch.object(ctrl, "start"):
            with patch.object(ctrl, "stop"):
                with patch.object(ctrl.safety, "start"):
                    # is_running stays True so only duration terminates
                    ctrl._running = True
                    start = time.time()
                    run_headless(MagicMock(), ctrl, duration=0.3)
                    elapsed = time.time() - start

        assert 0.2 <= elapsed <= 1.0

    def test_run_headless_step_termination(self):
        """With max_steps, controller._max_steps is set."""
        ctrl = _make_controller()

        with patch.object(ctrl, "start"):
            with patch.object(ctrl, "stop"):
                with patch.object(ctrl.safety, "start"):
                    ctrl._running = False  # Exit immediately
                    run_headless(MagicMock(), ctrl, max_steps=42)

        assert ctrl._max_steps == 42

    def test_run_headless_trajectory_end(self):
        """When controller.is_running becomes False, headless exits."""
        ctrl = _make_controller()
        call_count = 0

        original_is_running = type(ctrl).is_running

        def fake_is_running(self):
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # Stop after 2 checks

        with patch.object(ctrl, "start"):
            with patch.object(ctrl, "stop"):
                with patch.object(ctrl.safety, "start"):
                    ctrl._running = True
                    with patch.object(type(ctrl), "is_running", new_callable=lambda: property(fake_is_running)):
                        start = time.time()
                        run_headless(MagicMock(), ctrl)
                        elapsed = time.time() - start

        # Should exit within a second (2 iterations × 0.1s sleep)
        assert elapsed < 2.0

    def test_run_headless_stop_called_on_exit(self):
        """controller.stop() is always called (finally block)."""
        ctrl = _make_controller()

        with patch.object(ctrl, "start"):
            with patch.object(ctrl, "stop") as mock_stop:
                with patch.object(ctrl.safety, "start"):
                    ctrl._running = False
                    run_headless(MagicMock(), ctrl)

        mock_stop.assert_called_once()
