"""Tests for src/main.py — Phase 9.

Covers GLFW key map, key callback dispatch, run_with_viewer (mocked viewer),
and run_headless (duration, step, and trajectory-end termination).
"""
from __future__ import annotations

import os
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
    def test_glfw_key_map_space(self):
        assert GLFW_KEY_MAP[32] == "space"

    def test_glfw_key_map_e(self):
        assert GLFW_KEY_MAP[69] == "e"

    def test_glfw_key_map_a(self):
        assert GLFW_KEY_MAP[65] == "a"

    def test_glfw_key_map_w(self):
        assert GLFW_KEY_MAP[87] == "w"

    def test_glfw_key_map_s(self):
        assert GLFW_KEY_MAP[83] == "s"

    def test_glfw_key_map_d(self):
        assert GLFW_KEY_MAP[68] == "d"

    def test_glfw_key_map_q(self):
        assert GLFW_KEY_MAP[81] == "q"

    def test_glfw_key_map_z(self):
        assert GLFW_KEY_MAP[90] == "z"

    def test_glfw_key_map_x(self):
        assert GLFW_KEY_MAP[88] == "x"

    def test_glfw_key_map_r(self):
        assert GLFW_KEY_MAP[82] == "r"

    def test_glfw_key_map_c(self):
        assert GLFW_KEY_MAP[67] == "c"

    def test_glfw_key_map_n(self):
        assert GLFW_KEY_MAP[78] == "n"

    def test_glfw_key_map_p(self):
        assert GLFW_KEY_MAP[80] == "p"

    def test_glfw_key_map_all_values_are_strings(self):
        for keycode, name in GLFW_KEY_MAP.items():
            assert isinstance(keycode, int)
            assert isinstance(name, str)


# ============================================================================
# Key Callback Dispatch
# ============================================================================

class TestKeyCallback:
    def test_key_callback_dispatches_to_controller(self):
        """Pressing space (keycode 32) dispatches handle_key('space')."""
        ctrl = _make_controller()
        ctrl.handle_key = MagicMock()

        # Simulate what run_with_viewer does internally
        key = GLFW_KEY_MAP.get(32)
        if key:
            ctrl.handle_key(key)

        ctrl.handle_key.assert_called_once_with("space")

    def test_key_callback_unmapped_ignored(self):
        """An unmapped keycode should NOT call handle_key."""
        ctrl = _make_controller()
        ctrl.handle_key = MagicMock()

        key = GLFW_KEY_MAP.get(999)
        if key:
            ctrl.handle_key(key)

        ctrl.handle_key.assert_not_called()

    def test_key_callback_all_mapped_keys(self):
        """Every mapped keycode dispatches correctly."""
        ctrl = _make_controller()
        ctrl.handle_key = MagicMock()

        for keycode, expected_name in GLFW_KEY_MAP.items():
            ctrl.handle_key.reset_mock()
            key = GLFW_KEY_MAP.get(keycode)
            if key:
                ctrl.handle_key(key)
            ctrl.handle_key.assert_called_once_with(expected_name)


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

    def test_run_with_viewer_key_callback_wired(self):
        """Key callback from viewer dispatches to controller.handle_key."""
        ctrl = _make_controller()
        robot = MagicMock()
        robot.mj_model = MagicMock()
        robot.mj_data = MagicMock()

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
                ctrl.handle_key = MagicMock()
                run_with_viewer(robot, ctrl)

        # Now test the captured callback
        assert captured_callback is not None
        captured_callback(32)  # space
        ctrl.handle_key.assert_called_once_with("space")

    def test_run_with_viewer_key_callback_ignores_unmapped(self):
        """Unmapped keycodes do not reach handle_key."""
        ctrl = _make_controller()
        robot = MagicMock()
        robot.mj_model = MagicMock()
        robot.mj_data = MagicMock()

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

        captured_callback(12345)  # unmapped
        ctrl.handle_key.assert_not_called()


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
