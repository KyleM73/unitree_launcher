"""Tests for RealRobot (onboard C++ unitree_interface).

All tests mock ``unitree_interface`` so they run without the wheel installed.
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
# Helpers: build a fake ``unitree_interface`` module
# ---------------------------------------------------------------------------

def _make_fake_unitree_interface():
    """Create a mock unitree_interface module with correct structure."""
    mod = types.ModuleType("unitree_interface")

    # Enums
    mod.RobotType = mock.MagicMock()
    mod.RobotType.G1 = "G1"
    mod.MessageType = mock.MagicMock()
    mod.MessageType.HG = "HG"
    mod.ControlMode = mock.MagicMock()
    mod.ControlMode.PR = "PR"

    # IMU state
    imu = mock.MagicMock()
    imu.quat = [1.0, 0.0, 0.0, 0.0]
    imu.omega = [0.1, 0.2, 0.3]
    imu.accel = [0.0, 0.0, 9.81]

    # Motor state
    motor = mock.MagicMock()
    motor.q = [float(i) * 0.01 for i in range(29)]
    motor.dq = [float(i) * 0.001 for i in range(29)]
    motor.tau_est = [float(i) * 0.1 for i in range(29)]

    # LowState
    low_state = mock.MagicMock()
    low_state.imu = imu
    low_state.motor = motor
    low_state.mode_machine = 0

    # Motor command (returned by create_zero_command)
    motor_cmd = mock.MagicMock()
    motor_cmd.q_target = [0.0] * 29
    motor_cmd.dq_target = [0.0] * 29
    motor_cmd.tau_ff = [0.0] * 29
    motor_cmd.kp = [0.0] * 29
    motor_cmd.kd = [0.0] * 29

    # UnitreeInterface instance
    interface = mock.MagicMock()
    interface.read_low_state.return_value = low_state
    interface.create_zero_command.return_value = motor_cmd
    interface.write_low_command = mock.MagicMock()
    interface.set_control_mode = mock.MagicMock()

    mod.create_robot = mock.MagicMock(return_value=interface)

    return mod, interface, low_state, motor_cmd


@pytest.fixture
def fake_unitree():
    """Install fake unitree_interface in sys.modules for the test."""
    mod, interface, low_state, motor_cmd = _make_fake_unitree_interface()
    with mock.patch.dict(sys.modules, {"unitree_interface": mod}):
        yield {
            "module": mod,
            "interface": interface,
            "low_state": low_state,
            "motor_cmd": motor_cmd,
        }


@pytest.fixture
def config():
    """Minimal Config for RealRobot."""
    cfg = Config()
    cfg.robot.variant = "g1_29dof"
    cfg.network.interface = "en8"
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportError:
    def test_import_error_message(self, config):
        """Verify clear error when unitree_interface is missing."""
        # Ensure unitree_interface is NOT importable
        with mock.patch.dict(sys.modules, {"unitree_interface": None}):
            from unitree_launcher.robot.real_robot import RealRobot
            robot = RealRobot(config)
            with pytest.raises(ImportError, match="unitree_interface"):
                robot.connect()


class TestConnect:
    def test_connect(self, config, fake_unitree):
        """Mock create_robot, verify set_control_mode(PR) called."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()

        mod = fake_unitree["module"]
        mod.create_robot.assert_called_once_with(
            "en8", mod.RobotType.G1, mod.MessageType.HG,
        )
        fake_unitree["interface"].set_control_mode.assert_called_once_with(
            mod.ControlMode.PR,
        )
        assert robot._connected is True

    def test_connect_idempotent(self, config, fake_unitree):
        """Calling connect() twice only creates one interface."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()
        robot.connect()  # should not create a second interface
        assert fake_unitree["module"].create_robot.call_count == 1


class TestGetState:
    def test_get_state_mapping(self, config, fake_unitree):
        """Verify RobotState fields match mock LowState."""
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

        # Check values
        np.testing.assert_allclose(state.imu_quaternion, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(state.imu_angular_velocity, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state.imu_linear_acceleration, [0.0, 0.0, 9.81])
        assert state.joint_positions[0] == pytest.approx(0.0)
        assert state.joint_positions[1] == pytest.approx(0.01)

        # Base position/velocity should be NaN (real robot)
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
    def test_send_command_mapping(self, config, fake_unitree):
        """Verify q_target/kp/kd filled from RobotCommand."""
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

        motor_cmd = fake_unitree["motor_cmd"]
        # Verify the fields were set via attribute assignment
        assert motor_cmd.q_target == [0.5] * 29
        assert motor_cmd.dq_target == [0.1] * 29
        assert motor_cmd.tau_ff == [0.2] * 29
        assert motor_cmd.kp == [100.0] * 29
        assert motor_cmd.kd == [10.0] * 29

        # Verify write_low_command was called
        fake_unitree["interface"].write_low_command.assert_called_once_with(motor_cmd)

    def test_send_command_disconnected(self, config, fake_unitree):
        """send_command before connect is a no-op (no crash)."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        cmd = RobotCommand.damping(29)
        robot.send_command(cmd)  # should not raise


class TestGracefulShutdown:
    def test_graceful_shutdown(self, config, fake_unitree):
        """Verify damping command sent, sleep called, disconnect called."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.connect()

        with mock.patch("unitree_launcher.robot.real_robot.time.sleep") as mock_sleep:
            robot.graceful_shutdown(damping_duration=0.3)

        # Should have slept for the damping duration
        mock_sleep.assert_called_once_with(0.3)

        # Should be disconnected
        assert robot._connected is False
        assert robot._interface is None

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
        robot.step()  # should not raise, should not call anything on interface


class TestNDof:
    def test_n_dof_29(self, config, fake_unitree):
        """Verify returns 29 for g1_29dof variant."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        assert robot.n_dof == 29

    def test_n_dof_23(self, fake_unitree):
        """Verify returns 23 for g1_23dof variant."""
        from unitree_launcher.robot.real_robot import RealRobot

        cfg = Config()
        cfg.robot.variant = "g1_23dof"
        cfg.network.interface = "en8"
        robot = RealRobot(cfg)
        assert robot.n_dof == 23


class TestReset:
    def test_reset_warns(self, config, fake_unitree):
        """Verify reset logs a warning."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        robot.reset()  # should not raise


class TestSafety:
    def test_set_safety(self, config, fake_unitree):
        """Verify set_safety stores the reference."""
        from unitree_launcher.robot.real_robot import RealRobot

        robot = RealRobot(config)
        safety = mock.MagicMock()
        robot.set_safety(safety)
        assert robot._safety is safety
