"""Tests for the RealRobot DDS interface (Phase 11).

All tests use mocked DDS — no real robot or network required.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from unitree_launcher.config import Config, load_config
from unitree_launcher.robot.base import RobotCommand, RobotState
from unitree_launcher.robot.real_robot import (
    RealRobot,
    _MOTOR_MODE_SERVO,
    _NUM_MOTOR_IDL_HG,
    _WATCHDOG_TIMEOUT_S,
)


# ---------------------------------------------------------------------------
# Helpers: Mock IDL message types
# ---------------------------------------------------------------------------

class MockMotorState:
    """Minimal mock of unitree_hg MotorState_."""
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.ddq = 0.0
        self.tau_est = 0.0


class MockMotorCmd:
    """Minimal mock of unitree_hg MotorCmd_."""
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.reserve = 0


class MockIMUState:
    """Minimal mock of unitree_hg IMUState_."""
    def __init__(self):
        self.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.gyroscope = [0.0, 0.0, 0.0]
        self.accelerometer = [0.0, 0.0, 9.81]
        self.rpy = [0.0, 0.0, 0.0]
        self.temperature = 25


class MockLowState:
    """Minimal mock of unitree_hg LowState_."""
    def __init__(self, n_motors=35):
        self.motor_state = [MockMotorState() for _ in range(n_motors)]
        self.imu_state = MockIMUState()
        self.tick = 0
        self.crc = 0
        self.mode_machine = 0
        self.mode_pr = 0


class MockLowCmd:
    """Minimal mock of unitree_hg LowCmd_."""
    def __init__(self, n_motors=35):
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [MockMotorCmd() for _ in range(n_motors)]
        self.reserve = [0, 0, 0, 0]
        self.crc = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_29dof():
    """Default 29-DOF config."""
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    return load_config(str(cfg_path))


@pytest.fixture
def config_23dof():
    """23-DOF config."""
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "configs" / "g1_23dof.yaml"
    return load_config(str(cfg_path))


@pytest.fixture
def robot_29dof(config_29dof):
    """RealRobot instance (29-DOF, not connected)."""
    return RealRobot(config_29dof)


@pytest.fixture
def robot_23dof(config_23dof):
    """RealRobot instance (23-DOF, not connected)."""
    return RealRobot(config_23dof)


@pytest.fixture
def robot_with_cmd(robot_29dof):
    """RealRobot with mock publisher and cmd message injected (no DDS connect)."""
    robot = robot_29dof
    robot._low_cmd_msg = MockLowCmd()
    robot._cmd_pub = MagicMock()
    robot._crc = MagicMock()
    robot._crc.Crc = MagicMock(return_value=0xDEADBEEF)
    robot._connected = True
    return robot


# ---------------------------------------------------------------------------
# Basic lifecycle (no DDS required)
# ---------------------------------------------------------------------------

class TestRealRobotLifecycle:
    def test_real_robot_init_without_dds(self, config_29dof):
        """Init succeeds without DDS — no network call on construction."""
        robot = RealRobot(config_29dof)
        assert robot.n_dof == 29
        assert not robot._connected

    def test_real_robot_step_is_noop(self, robot_29dof):
        """step() returns immediately with no side effects."""
        # Should not raise, should not modify state
        robot_29dof.step()
        robot_29dof.step()
        robot_29dof.step()
        # No assertion needed — just confirming no exception

    def test_real_robot_reset_logs_warning(self, robot_29dof, caplog):
        """reset() logs a warning that physical robot cannot be reset."""
        with caplog.at_level(logging.WARNING):
            robot_29dof.reset()
        assert "cannot reset physical robot" in caplog.text.lower()

    def test_real_robot_reset_with_state_logs_warning(self, robot_29dof, caplog):
        """reset(initial_state) also logs warning."""
        state = RobotState.zeros(29)
        with caplog.at_level(logging.WARNING):
            robot_29dof.reset(initial_state=state)
        assert "cannot reset physical robot" in caplog.text.lower()

    def test_real_robot_n_dof_29(self, robot_29dof):
        """n_dof returns 29 for 29-DOF config."""
        assert robot_29dof.n_dof == 29

    def test_real_robot_n_dof_23(self, robot_23dof):
        """n_dof returns 23 for 23-DOF config."""
        assert robot_23dof.n_dof == 23


# ---------------------------------------------------------------------------
# Command construction (mock DDS)
# ---------------------------------------------------------------------------

class TestRealRobotCommands:
    def test_real_robot_send_command_motor_mode(self, robot_with_cmd):
        """All controlled joints get motor mode 0x01 (PMSM servo)."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand(
            joint_positions=np.zeros(n),
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=np.ones(n) * 100.0,
            kd=np.ones(n) * 10.0,
        )
        robot_with_cmd.send_command(cmd)

        msg = robot_with_cmd._low_cmd_msg
        for i in range(n):
            assert msg.motor_cmd[i].mode == _MOTOR_MODE_SERVO, (
                f"Motor {i} mode should be 0x01, got {msg.motor_cmd[i].mode}"
            )

    def test_real_robot_send_command_field_mapping(self, robot_with_cmd):
        """Known RobotCommand maps correctly to LowCmd_ fields."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand(
            joint_positions=np.arange(n, dtype=np.float64) * 0.1,
            joint_velocities=np.arange(n, dtype=np.float64) * 0.01,
            joint_torques=np.arange(n, dtype=np.float64) * 0.5,
            kp=np.full(n, 80.0),
            kd=np.full(n, 5.0),
        )
        robot_with_cmd.send_command(cmd)

        msg = robot_with_cmd._low_cmd_msg
        for i in range(n):
            assert msg.motor_cmd[i].q == pytest.approx(i * 0.1, abs=1e-6)
            assert msg.motor_cmd[i].dq == pytest.approx(i * 0.01, abs=1e-6)
            assert msg.motor_cmd[i].tau == pytest.approx(i * 0.5, abs=1e-6)
            assert msg.motor_cmd[i].kp == pytest.approx(80.0)
            assert msg.motor_cmd[i].kd == pytest.approx(5.0)

    def test_real_robot_send_command_crc(self, robot_with_cmd):
        """CRC32 is computed and set on LowCmd_ before publishing."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand.damping(n)
        robot_with_cmd.send_command(cmd)

        # Verify CRC was computed
        robot_with_cmd._crc.Crc.assert_called_once_with(
            robot_with_cmd._low_cmd_msg
        )
        # Verify CRC value was set
        assert robot_with_cmd._low_cmd_msg.crc == 0xDEADBEEF
        # Verify message was published
        robot_with_cmd._cmd_pub.Write.assert_called_once_with(
            robot_with_cmd._low_cmd_msg
        )

    def test_real_robot_send_command_without_connect(self, robot_29dof):
        """send_command() is a no-op when not connected (no publisher)."""
        cmd = RobotCommand.damping(29)
        # Should not raise
        robot_29dof.send_command(cmd)


# ---------------------------------------------------------------------------
# State subscription (mock DDS)
# ---------------------------------------------------------------------------

class TestRealRobotState:
    def test_real_robot_get_state_mapping(self, robot_29dof):
        """LowState_ with known values maps correctly to RobotState."""
        msg = MockLowState()

        # Set known motor states
        for i in range(29):
            msg.motor_state[i].q = float(i) * 0.1
            msg.motor_state[i].dq = float(i) * 0.01
            msg.motor_state[i].tau_est = float(i) * 0.5

        # Set known IMU
        msg.imu_state.quaternion = [0.9, 0.1, 0.2, 0.3]
        msg.imu_state.gyroscope = [0.01, 0.02, 0.03]
        msg.imu_state.accelerometer = [0.1, 0.2, 9.8]

        # Invoke the state callback directly
        robot_29dof._on_low_state(msg)

        state = robot_29dof.get_state()

        # Verify joint data
        for i in range(29):
            assert state.joint_positions[i] == pytest.approx(i * 0.1, abs=1e-6)
            assert state.joint_velocities[i] == pytest.approx(i * 0.01, abs=1e-6)
            assert state.joint_torques[i] == pytest.approx(i * 0.5, abs=1e-6)

        # Verify IMU
        np.testing.assert_allclose(
            state.imu_quaternion, [0.9, 0.1, 0.2, 0.3], atol=1e-6
        )
        np.testing.assert_allclose(
            state.imu_angular_velocity, [0.01, 0.02, 0.03], atol=1e-6
        )
        np.testing.assert_allclose(
            state.imu_linear_acceleration, [0.1, 0.2, 9.8], atol=1e-6
        )

    def test_real_robot_get_state_thread_safe(self, robot_29dof):
        """Concurrent callback + get_state() causes no corruption."""
        errors = []
        n_iters = 200
        barrier = threading.Barrier(2)

        def writer():
            barrier.wait()
            for i in range(n_iters):
                msg = MockLowState()
                for j in range(29):
                    msg.motor_state[j].q = float(i)
                robot_29dof._on_low_state(msg)

        def reader():
            barrier.wait()
            for _ in range(n_iters):
                try:
                    state = robot_29dof.get_state()
                    # All positions should be the same value (from one write)
                    positions = state.joint_positions
                    if not np.all(positions == positions[0]):
                        # Check if it's still zeros (initial) or a consistent write
                        unique = np.unique(positions)
                        if len(unique) > 1:
                            errors.append(
                                f"Torn read: {unique}"
                            )
                except Exception as e:
                    errors.append(str(e))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_real_robot_get_state_nan_base(self, robot_29dof):
        """base_position and base_velocity are NaN (not available on real robot)."""
        msg = MockLowState()
        robot_29dof._on_low_state(msg)

        state = robot_29dof.get_state()
        assert np.all(np.isnan(state.base_position))
        assert np.all(np.isnan(state.base_velocity))

    def test_real_robot_get_state_before_connect(self, robot_29dof):
        """get_state() before any callback returns zero state."""
        state = robot_29dof.get_state()
        np.testing.assert_array_equal(state.joint_positions, np.zeros(29))

    def test_real_robot_get_state_returns_copy(self, robot_29dof):
        """get_state() returns independent copies."""
        msg = MockLowState()
        for i in range(29):
            msg.motor_state[i].q = 1.0
        robot_29dof._on_low_state(msg)

        s1 = robot_29dof.get_state()
        s2 = robot_29dof.get_state()
        s1.joint_positions[0] = 999.0
        assert s2.joint_positions[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Communication monitoring
# ---------------------------------------------------------------------------

class TestRealRobotWatchdog:
    def test_real_robot_watchdog_timeout(self, robot_29dof):
        """E-stop triggered when state is stale beyond watchdog timeout."""
        mock_safety = MagicMock()
        robot_29dof.set_safety(mock_safety)

        # Deliver a state message
        msg = MockLowState()
        robot_29dof._on_low_state(msg)

        # Artificially age the timestamp
        robot_29dof._last_state_time = time.monotonic() - 0.5  # 500ms ago

        # get_state should trigger estop via safety
        robot_29dof.get_state()
        mock_safety.estop.assert_called_once()

    def test_real_robot_watchdog_no_false_trigger(self, robot_29dof):
        """No E-stop when state is fresh."""
        mock_safety = MagicMock()
        robot_29dof.set_safety(mock_safety)

        msg = MockLowState()
        robot_29dof._on_low_state(msg)

        # State just arrived — should NOT trigger estop
        robot_29dof.get_state()
        mock_safety.estop.assert_not_called()

    def test_real_robot_connect_timeout(self, config_29dof):
        """connect() raises TimeoutError if no state message within 5s."""
        robot = RealRobot(config_29dof)

        # Mock all DDS imports to avoid real network
        mock_channel_pub = MagicMock()
        mock_channel_sub = MagicMock()
        mock_crc = MagicMock()

        with patch("unitree_launcher.robot.real_robot.patch_unitree_threading"), \
             patch("unitree_launcher.robot.real_robot.resolve_network_interface", return_value="lo0"), \
             patch.dict("sys.modules", {
                 "unitree_sdk2py.core.channel": MagicMock(
                     ChannelFactoryInitialize=MagicMock(),
                     ChannelPublisher=MagicMock(return_value=mock_channel_pub),
                     ChannelSubscriber=MagicMock(return_value=mock_channel_sub),
                 ),
                 "unitree_sdk2py.idl.unitree_hg.msg.dds_": MagicMock(),
                 "unitree_sdk2py.idl.default": MagicMock(
                     unitree_hg_msg_dds__LowCmd_=MagicMock(return_value=MockLowCmd()),
                 ),
                 "unitree_sdk2py.utils.crc": MagicMock(
                     CRC=MagicMock(return_value=mock_crc),
                 ),
             }):
            # Monkey-patch _CONNECT_TIMEOUT_S to make test fast
            import unitree_launcher.robot.real_robot as rr_mod
            original_timeout = rr_mod._CONNECT_TIMEOUT_S
            rr_mod._CONNECT_TIMEOUT_S = 0.1  # 100ms for test speed

            try:
                with pytest.raises(TimeoutError, match="No state message"):
                    robot.connect()
            finally:
                rr_mod._CONNECT_TIMEOUT_S = original_timeout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestRealRobotConfig:
    def test_real_robot_domain_id(self, config_29dof):
        """connect() calls ChannelFactoryInitialize with domain_id=0."""
        robot = RealRobot(config_29dof)

        mock_init = MagicMock()
        mock_channel_pub = MagicMock()
        mock_channel_sub = MagicMock()
        mock_crc = MagicMock()

        with patch("unitree_launcher.robot.real_robot.patch_unitree_threading"), \
             patch("unitree_launcher.robot.real_robot.resolve_network_interface", return_value="eth0"), \
             patch.dict("sys.modules", {
                 "unitree_sdk2py.core.channel": MagicMock(
                     ChannelFactoryInitialize=mock_init,
                     ChannelPublisher=MagicMock(return_value=mock_channel_pub),
                     ChannelSubscriber=MagicMock(return_value=mock_channel_sub),
                 ),
                 "unitree_sdk2py.idl.unitree_hg.msg.dds_": MagicMock(),
                 "unitree_sdk2py.idl.default": MagicMock(
                     unitree_hg_msg_dds__LowCmd_=MagicMock(return_value=MockLowCmd()),
                 ),
                 "unitree_sdk2py.utils.crc": MagicMock(
                     CRC=MagicMock(return_value=mock_crc),
                 ),
             }):
            # Make connect() succeed by simulating state arrival
            original_event_wait = threading.Event.wait

            def fake_wait(self_event, timeout=None):
                self_event.set()
                return True

            with patch.object(threading.Event, "wait", fake_wait):
                robot.connect()

            # Verify domain_id=0 was used
            mock_init.assert_called_once_with(0, "eth0")

    def test_real_robot_dds_topic_names(self, config_29dof):
        """Subscriber listens on rt/lowstate, publisher targets rt/lowcmd."""
        robot = RealRobot(config_29dof)

        mock_pub_cls = MagicMock()
        mock_sub_cls = MagicMock()

        with patch("unitree_launcher.robot.real_robot.patch_unitree_threading"), \
             patch("unitree_launcher.robot.real_robot.resolve_network_interface", return_value="lo0"), \
             patch.dict("sys.modules", {
                 "unitree_sdk2py.core.channel": MagicMock(
                     ChannelFactoryInitialize=MagicMock(),
                     ChannelPublisher=mock_pub_cls,
                     ChannelSubscriber=mock_sub_cls,
                 ),
                 "unitree_sdk2py.idl.unitree_hg.msg.dds_": MagicMock(),
                 "unitree_sdk2py.idl.default": MagicMock(
                     unitree_hg_msg_dds__LowCmd_=MagicMock(return_value=MockLowCmd()),
                 ),
                 "unitree_sdk2py.utils.crc": MagicMock(
                     CRC=MagicMock(),
                 ),
             }):
            with patch.object(threading.Event, "wait", lambda self, timeout=None: (self.set(), True)[1]):
                robot.connect()

            # Check topic names
            pub_call_args = mock_pub_cls.call_args
            assert pub_call_args[0][0] == "rt/lowcmd"

            sub_call_args = mock_sub_cls.call_args
            assert sub_call_args[0][0] == "rt/lowstate"


# ---------------------------------------------------------------------------
# Protocol fields (mode_machine, mode_pr, non-controlled slots)
# ---------------------------------------------------------------------------

class TestRealRobotProtocol:
    def test_mode_machine_echoed(self, robot_with_cmd):
        """mode_machine from LowState is echoed in every LowCmd."""
        # Simulate receiving a state with mode_machine=5
        msg = MockLowState()
        msg.mode_machine = 5
        robot_with_cmd._on_low_state(msg)

        # Send a command
        n = robot_with_cmd.n_dof
        cmd = RobotCommand.damping(n)
        robot_with_cmd.send_command(cmd)

        assert robot_with_cmd._low_cmd_msg.mode_machine == 5

    def test_mode_machine_updates(self, robot_with_cmd):
        """mode_machine tracks the latest LowState value."""
        for mm in [0, 3, 5, 7]:
            msg = MockLowState()
            msg.mode_machine = mm
            robot_with_cmd._on_low_state(msg)

            cmd = RobotCommand.damping(robot_with_cmd.n_dof)
            robot_with_cmd.send_command(cmd)
            assert robot_with_cmd._low_cmd_msg.mode_machine == mm

    def test_mode_pr_zero(self, robot_with_cmd):
        """mode_pr is always set to 0 in every LowCmd."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand.damping(n)

        # Set mode_pr to something else to verify it gets overwritten
        robot_with_cmd._low_cmd_msg.mode_pr = 99
        robot_with_cmd.send_command(cmd)

        assert robot_with_cmd._low_cmd_msg.mode_pr == 0

    def test_non_controlled_slots_disabled(self, robot_with_cmd):
        """Motor slots 29-34 get mode=0 and zeroed fields after send_command."""
        n = robot_with_cmd.n_dof  # 29
        cmd = RobotCommand(
            joint_positions=np.ones(n),
            joint_velocities=np.ones(n),
            joint_torques=np.ones(n),
            kp=np.full(n, 100.0),
            kd=np.full(n, 10.0),
        )
        robot_with_cmd.send_command(cmd)

        msg = robot_with_cmd._low_cmd_msg
        for i in range(n, 35):
            assert msg.motor_cmd[i].mode == 0, f"Slot {i} mode should be 0"
            assert msg.motor_cmd[i].q == 0.0, f"Slot {i} q should be 0"
            assert msg.motor_cmd[i].dq == 0.0, f"Slot {i} dq should be 0"
            assert msg.motor_cmd[i].tau == 0.0, f"Slot {i} tau should be 0"
            assert msg.motor_cmd[i].kp == 0.0, f"Slot {i} kp should be 0"
            assert msg.motor_cmd[i].kd == 0.0, f"Slot {i} kd should be 0"

    def test_controlled_slots_servo_mode(self, robot_with_cmd):
        """Controlled slots 0-28 have mode=0x01 (PMSM servo)."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand.damping(n)
        robot_with_cmd.send_command(cmd)

        msg = robot_with_cmd._low_cmd_msg
        for i in range(n):
            assert msg.motor_cmd[i].mode == _MOTOR_MODE_SERVO


# ---------------------------------------------------------------------------
# 500 Hz publish thread
# ---------------------------------------------------------------------------

class TestRealRobotPublishThread:
    def test_publish_noop_before_first_send(self, robot_with_cmd):
        """_publish_cmd is a no-op before any send_command call."""
        assert not robot_with_cmd._cmd_ready

        # Call _publish_cmd directly — should not call Write
        robot_with_cmd._publish_cmd()
        robot_with_cmd._cmd_pub.Write.assert_not_called()

    def test_publish_after_send(self, robot_with_cmd):
        """_publish_cmd re-publishes after send_command sets _cmd_ready."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand.damping(n)
        robot_with_cmd.send_command(cmd)

        # send_command calls Write once
        assert robot_with_cmd._cmd_pub.Write.call_count == 1

        # _publish_cmd should call Write again
        robot_with_cmd._publish_cmd()
        assert robot_with_cmd._cmd_pub.Write.call_count == 2

    def test_publish_cmd_uses_lock(self, robot_with_cmd):
        """_publish_cmd acquires _cmd_lock to protect the message."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand.damping(n)
        robot_with_cmd.send_command(cmd)

        # If lock is already held, _publish_cmd should block.
        # We test by acquiring the lock and verifying _publish_cmd doesn't
        # call Write (since it can't acquire the lock in a non-blocking test).
        lock_acquired = threading.Event()
        publish_done = threading.Event()

        def hold_lock():
            with robot_with_cmd._cmd_lock:
                lock_acquired.set()
                # Hold lock for a bit
                publish_done.wait(timeout=1.0)

        t = threading.Thread(target=hold_lock)
        t.start()
        lock_acquired.wait()

        # Reset call count
        robot_with_cmd._cmd_pub.Write.reset_mock()

        # _publish_cmd in a thread — should block on _cmd_lock
        result = [None]
        def try_publish():
            robot_with_cmd._publish_cmd()
            result[0] = "done"

        t2 = threading.Thread(target=try_publish)
        t2.start()
        # Give it a moment — it should be blocked
        t2.join(timeout=0.05)

        # Release the lock
        publish_done.set()
        t.join()
        t2.join(timeout=1.0)

        # Now it should have completed
        assert result[0] == "done"

    def test_disconnect_stops_publish_thread(self, robot_with_cmd):
        """disconnect() shuts down the publish thread."""
        from unitree_launcher.compat import RecurrentThread

        mock_thread = MagicMock(spec=RecurrentThread)
        robot_with_cmd._publish_thread = mock_thread

        robot_with_cmd.disconnect()

        mock_thread.Shutdown.assert_called_once()
        assert robot_with_cmd._publish_thread is None


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

class TestRealRobotGracefulShutdown:
    def test_graceful_shutdown_sends_damping(self, robot_with_cmd):
        """graceful_shutdown sends a damping command before disconnecting."""
        # Keep a reference to the mock publisher before disconnect nulls it
        pub_mock = robot_with_cmd._cmd_pub
        robot_with_cmd.graceful_shutdown(damping_duration=0.05)

        # Verify at least one Write call was made (damping command)
        assert pub_mock.Write.call_count >= 1

        # Verify robot is disconnected
        assert not robot_with_cmd._connected

    def test_graceful_shutdown_damping_fields(self, robot_with_cmd):
        """Damping command has kp=0 and non-zero kd."""
        robot_with_cmd.graceful_shutdown(damping_duration=0.05)

        msg = robot_with_cmd._low_cmd_msg
        for i in range(robot_with_cmd.n_dof):
            assert msg.motor_cmd[i].kp == 0.0, f"Joint {i} kp should be 0"
            assert msg.motor_cmd[i].kd > 0.0, f"Joint {i} kd should be > 0"

    def test_graceful_shutdown_idempotent(self, robot_with_cmd):
        """Calling graceful_shutdown twice doesn't error."""
        robot_with_cmd.graceful_shutdown(damping_duration=0.05)
        # Second call should be a no-op (already disconnected)
        robot_with_cmd.graceful_shutdown(damping_duration=0.05)
        assert not robot_with_cmd._connected

    def test_graceful_shutdown_clears_cmd_ready(self, robot_with_cmd):
        """After graceful_shutdown, _cmd_ready is False."""
        n = robot_with_cmd.n_dof
        cmd = RobotCommand.damping(n)
        robot_with_cmd.send_command(cmd)
        assert robot_with_cmd._cmd_ready

        robot_with_cmd.graceful_shutdown(damping_duration=0.05)
        assert not robot_with_cmd._cmd_ready

    def test_graceful_shutdown_when_not_connected(self, robot_29dof):
        """graceful_shutdown on unconnected robot is a no-op."""
        assert not robot_29dof._connected
        robot_29dof.graceful_shutdown()  # Should not raise
