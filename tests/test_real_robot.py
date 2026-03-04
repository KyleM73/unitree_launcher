"""Tests for RealRobot (onboard C++ unitree_cpp).

All tests mock ``unitree_cpp`` so they run without the binding installed.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np
import pytest

from unitree_launcher.config import Config
from unitree_launcher.robot.base import RobotCommand, RobotState


# ---------------------------------------------------------------------------
# Helpers: build a fake ``unitree_cpp`` module
# ---------------------------------------------------------------------------

def _make_fake_unitree_cpp():
    """Create a mock unitree_cpp module matching the real API."""
    mod = types.ModuleType("unitree_cpp")

    # RobotState returned by get_robot_state()
    motor_state = mock.MagicMock()
    motor_state.q = [float(i) * 0.01 for i in range(29)]
    motor_state.dq = [float(i) * 0.001 for i in range(29)]
    motor_state.tau_est = [float(i) * 0.1 for i in range(29)]

    imu_state = mock.MagicMock()
    imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    imu_state.gyroscope = [0.1, 0.2, 0.3]
    imu_state.accelerometer = [0.0, 0.0, 9.81]

    robot_state = mock.MagicMock()
    robot_state.motor_state = motor_state
    robot_state.imu_state = imu_state
    robot_state.wireless_remote = b"\x00" * 40
    robot_state.tick = 1

    # UnitreeController instance
    controller = mock.MagicMock()
    controller.get_robot_state.return_value = robot_state
    controller.self_check.return_value = True
    controller.step = mock.MagicMock()
    controller.set_gains = mock.MagicMock()
    controller.shutdown = mock.MagicMock()

    mod.UnitreeController = mock.MagicMock(return_value=controller)

    return mod, controller, robot_state


@pytest.fixture
def fake_unitree():
    """Install fake unitree_cpp in sys.modules for the test."""
    mod, controller, robot_state = _make_fake_unitree_cpp()
    with mock.patch.dict(sys.modules, {"unitree_cpp": mod}):
        yield {
            "module": mod,
            "controller": controller,
            "robot_state": robot_state,
        }


@pytest.fixture
def config():
    """Minimal Config for RealRobot."""
    cfg = Config()
    cfg.robot.variant = "g1_29dof"
    cfg.network.interface = "eth0"
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportError:
    def test_import_error_message(self, config):
        """Verify clear error when unitree_cpp is missing."""
        with mock.patch.dict(sys.modules, {"unitree_cpp": None}):
            from unitree_launcher.robot.real_robot import RealRobot
            robot = RealRobot(config)
            with pytest.raises(ImportError, match="unitree_cpp"):
                robot.connect()


class TestConnect:
    def test_connect(self, config, fake_unitree):
        """Verify UnitreeController constructed with correct config."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()

        mod = fake_unitree["module"]
        mod.UnitreeController.assert_called_once()
        call_args = mod.UnitreeController.call_args[0][0]
        assert call_args["net_if"] == "eth0"
        assert call_args["msg_type"] == "hg"
        assert call_args["num_dofs"] == 29
        assert robot._connected is True

    def test_connect_idempotent(self, config, fake_unitree):
        """Calling connect() twice only creates one controller."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()
        robot.connect()
        assert fake_unitree["module"].UnitreeController.call_count == 1


class TestGetState:
    def test_get_state_mapping(self, config, fake_unitree):
        """Verify RobotState fields match mock state."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()
        state = robot.get_state()

        assert isinstance(state, RobotState)
        assert state.joint_positions.shape == (29,)
        assert state.joint_velocities.shape == (29,)
        assert state.joint_torques.shape == (29,)
        assert state.imu_quaternion.shape == (4,)
        assert state.imu_angular_velocity.shape == (3,)
        assert state.imu_linear_acceleration.shape == (3,)

        np.testing.assert_allclose(state.imu_quaternion, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(state.imu_angular_velocity, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state.imu_linear_acceleration, [0.0, 0.0, 9.81])
        assert state.joint_positions[0] == pytest.approx(0.0)
        assert state.joint_positions[1] == pytest.approx(0.01)

        assert np.all(np.isnan(state.base_position))
        assert np.all(np.isnan(state.base_velocity))

    def test_get_state_disconnected(self, config, fake_unitree):
        """get_state before connect returns zeros."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        state = robot.get_state()
        assert isinstance(state, RobotState)
        np.testing.assert_array_equal(state.joint_positions, np.zeros(29))


class TestSendCommand:
    def test_send_command_calls_set_gains_and_step(self, config, fake_unitree):
        """Verify set_gains then step called with correct values."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()

        cmd = RobotCommand(
            joint_positions=np.ones(29) * 0.5,
            joint_velocities=np.ones(29) * 0.1,
            joint_torques=np.ones(29) * 0.2,
            kp=np.ones(29) * 100.0,
            kd=np.ones(29) * 10.0,
        )
        robot.send_command(cmd)

        ctrl = fake_unitree["controller"]
        ctrl.set_gains.assert_called_once_with([100.0] * 29, [10.0] * 29)
        ctrl.step.assert_called_once_with([0.5] * 29)

    def test_send_command_disconnected(self, config, fake_unitree):
        """send_command before connect is a no-op."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        cmd = RobotCommand.damping(29)
        robot.send_command(cmd)  # should not raise


class TestGracefulShutdown:
    def test_graceful_shutdown(self, config, fake_unitree):
        """Verify shutdown called, sleep called, disconnected."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()

        with mock.patch("unitree_launcher.robot.real_robot.time.sleep") as mock_sleep:
            robot.graceful_shutdown(damping_duration=0.3)

        fake_unitree["controller"].shutdown.assert_called_once()
        mock_sleep.assert_called_once_with(0.3)
        assert robot._connected is False

    def test_graceful_shutdown_idempotent(self, config, fake_unitree):
        """Calling graceful_shutdown when not connected is a no-op."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.graceful_shutdown()  # should not raise


class TestStep:
    def test_step_is_noop(self, config, fake_unitree):
        """Verify step() does nothing."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()
        robot.step()


class TestNDof:
    def test_n_dof_29(self, config, fake_unitree):
        from unitree_launcher.robot.real_robot import RealRobot
        robot = RealRobot(config)
        assert robot.n_dof == 29

    def test_n_dof_23(self, fake_unitree):
        from unitree_launcher.robot.real_robot import RealRobot
        cfg = Config()
        cfg.robot.variant = "g1_23dof"
        cfg.network.interface = "eth0"
        robot = RealRobot(cfg)
        assert robot.n_dof == 23


class TestReset:
    def test_reset_warns(self, config, fake_unitree):
        from unitree_launcher.robot.real_robot import RealRobot
        robot = RealRobot(config)
        robot.reset()


class TestSafety:
    def test_set_safety(self, config, fake_unitree):
        from unitree_launcher.robot.real_robot import RealRobot
        robot = RealRobot(config)
        safety = mock.MagicMock()
        robot.set_safety(safety)
        assert robot._safety is safety
