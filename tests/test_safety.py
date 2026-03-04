"""Tests for src/control/safety.py — SafetyController state machine, damping, orientation, clamping."""
import threading

import numpy as np
import pytest

from unitree_launcher.config import (
    Config,
    G1_29DOF_JOINTS,
    JOINT_LIMITS_29DOF,
    Q_HOME_29DOF,
    SafetyConfig,
    TORQUE_LIMITS_29DOF,
    VELOCITY_LIMITS_29DOF,
)
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.robot.base import RobotCommand, RobotState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return Config()  # default: g1_29dof, all safety limits on


@pytest.fixture
def safety(config):
    return SafetyController(config, n_dof=29)


@pytest.fixture
def standing_state():
    """RobotState at home position, upright."""
    state = RobotState.zeros(29)
    state.imu_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # identity = upright
    return state


# ---------------------------------------------------------------------------
# State Machine Tests
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_initial_state_idle(self, safety):
        assert safety.state == SystemState.IDLE

    def test_idle_to_running(self, safety):
        assert safety.start() is True
        assert safety.state == SystemState.RUNNING

    def test_running_to_stopped(self, safety):
        safety.start()
        assert safety.stop() is True
        assert safety.state == SystemState.STOPPED

    def test_running_to_estop(self, safety):
        safety.start()
        safety.estop()
        assert safety.state == SystemState.ESTOP

    def test_estop_to_stopped(self, safety):
        safety.start()
        safety.estop()
        assert safety.clear_estop() is True
        assert safety.state == SystemState.STOPPED

    def test_estop_latching(self, safety):
        """After estop(), state remains ESTOP until cleared."""
        safety.start()
        safety.estop()
        # start() should fail from ESTOP
        assert safety.start() is False
        assert safety.state == SystemState.ESTOP

    def test_cannot_start_from_running(self, safety):
        safety.start()
        assert safety.start() is False
        assert safety.state == SystemState.RUNNING

    def test_estop_from_stopped(self, safety):
        safety.start()
        safety.stop()
        safety.estop()
        assert safety.state == SystemState.ESTOP

    def test_estop_from_idle_rejected(self, safety):
        """estop() from IDLE does nothing (IDLE is not E-stoppable)."""
        safety.estop()
        assert safety.state == SystemState.IDLE

    def test_stopped_to_running_transition(self, safety):
        """start() from STOPPED transitions to RUNNING (resume after stop)."""
        safety.start()
        safety.stop()
        assert safety.start() is True
        assert safety.state == SystemState.RUNNING

    def test_stop_from_idle_noop(self, safety):
        """stop() from IDLE returns False (invalid transition)."""
        assert safety.stop() is False
        assert safety.state == SystemState.IDLE

    def test_clear_estop_from_non_estop(self, safety):
        """clear_estop() from RUNNING/IDLE/STOPPED returns False."""
        assert safety.clear_estop() is False  # IDLE

        safety.start()
        assert safety.clear_estop() is False  # RUNNING

        safety.stop()
        assert safety.clear_estop() is False  # STOPPED

    def test_estop_idempotent(self, safety):
        """Calling estop() 10 times from RUNNING stays in ESTOP, no error."""
        safety.start()
        for _ in range(10):
            safety.estop()
        assert safety.state == SystemState.ESTOP


# ---------------------------------------------------------------------------
# Damping Command Tests
# ---------------------------------------------------------------------------

class TestDampingCommand:
    def test_damping_command_values(self, safety, standing_state):
        """Damping command has correct shape and field values."""
        standing_state.joint_positions = np.arange(29, dtype=float) * 0.1
        cmd = safety.get_damping_command(standing_state)
        assert cmd.joint_positions.shape == (29,)
        np.testing.assert_array_equal(cmd.kp, np.zeros(29))
        np.testing.assert_array_equal(cmd.kd, np.full(29, 8.0))
        np.testing.assert_array_equal(cmd.joint_positions, standing_state.joint_positions)
        np.testing.assert_array_equal(cmd.joint_velocities, np.zeros(29))
        np.testing.assert_array_equal(cmd.joint_torques, np.zeros(29))

    def test_damping_command_does_not_alias_state(self, safety, standing_state):
        """Damping command positions should be a copy, not a reference."""
        cmd = safety.get_damping_command(standing_state)
        standing_state.joint_positions[0] = 999.0
        assert cmd.joint_positions[0] != 999.0


# ---------------------------------------------------------------------------
# Orientation Check Tests
# ---------------------------------------------------------------------------

class TestOrientationCheck:
    def test_orientation_check_upright(self, safety):
        """Identity quaternion = upright, should be safe."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        is_safe, msg = safety.check_orientation(quat)
        assert is_safe is True

    def test_orientation_check_tilted_safe(self, safety):
        """Small tilt (~15 deg) should be safe."""
        angle = np.radians(15)
        # Rotation about X axis by 15 degrees
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        is_safe, msg = safety.check_orientation(quat)
        assert is_safe is True

    def test_orientation_check_tilted_unsafe(self, safety):
        """Large tilt (~60 deg) should be unsafe."""
        angle = np.radians(60)
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        is_safe, msg = safety.check_orientation(quat)
        assert is_safe is False

    def test_orientation_check_inverted(self, safety):
        """180 deg rotation = inverted, should be unsafe."""
        # 180 deg about X: quat = [0, 1, 0, 0]
        quat = np.array([0.0, 1.0, 0.0, 0.0])
        is_safe, msg = safety.check_orientation(quat)
        assert is_safe is False

    def test_orientation_boundary_angle(self, safety):
        """Test at boundary: projected gravity Z ≈ -0.8 corresponds to ~36.87 deg.
        Just below threshold should fail, just above should pass."""
        # arccos(0.8) ≈ 36.87 degrees
        # Just inside safe: 35 deg tilt
        angle_safe = np.radians(35)
        quat_safe = np.array([np.cos(angle_safe / 2), np.sin(angle_safe / 2), 0.0, 0.0])
        is_safe, _ = safety.check_orientation(quat_safe)
        assert is_safe is True  # cos(35°) ≈ 0.819 > 0.8

        # Just outside safe: 38 deg tilt
        angle_unsafe = np.radians(38)
        quat_unsafe = np.array([np.cos(angle_unsafe / 2), np.sin(angle_unsafe / 2), 0.0, 0.0])
        is_safe, _ = safety.check_orientation(quat_unsafe)
        assert is_safe is False  # cos(38°) ≈ 0.788 < 0.8


# ---------------------------------------------------------------------------
# Thread Safety Tests
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_safety_thread_safe_concurrent_transitions(self, config):
        """Spawn 10 threads calling estop() and start() concurrently.
        Verify no exceptions, state is always valid."""
        safety = SafetyController(config, n_dof=29)
        safety.start()  # Move to RUNNING first

        barrier = threading.Barrier(10)
        errors = []

        def worker(i):
            try:
                barrier.wait(timeout=5)
                if i % 2 == 0:
                    safety.estop()
                else:
                    safety.start()
                # Verify state is always a valid SystemState
                state = safety.state
                assert isinstance(state, SystemState)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors in threads: {errors}"
        # Final state must be ESTOP (some threads called estop())
        assert safety.state == SystemState.ESTOP


# ---------------------------------------------------------------------------
# Clamp Command Tests
# ---------------------------------------------------------------------------

class TestClampCommand:
    def test_safety_clamp_joint_position(self, config):
        """When joint_position_limits=True, clamp_command() clips target positions."""
        safety = SafetyController(config, n_dof=29)
        joints = G1_29DOF_JOINTS

        # Create command with positions exceeding limits
        cmd = RobotCommand(
            joint_positions=np.full(29, 100.0),  # way above all limits
            joint_velocities=np.zeros(29),
            joint_torques=np.zeros(29),
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result = safety.clamp_command(cmd)

        # Verify each joint is clamped to its max
        for i, name in enumerate(joints):
            lo, hi = JOINT_LIMITS_29DOF[name]
            assert result.joint_positions[i] == pytest.approx(hi), f"Joint {name} not clamped to max"

        # Also test below limits
        cmd_low = RobotCommand(
            joint_positions=np.full(29, -100.0),
            joint_velocities=np.zeros(29),
            joint_torques=np.zeros(29),
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result_low = safety.clamp_command(cmd_low)
        for i, name in enumerate(joints):
            lo, hi = JOINT_LIMITS_29DOF[name]
            assert result_low.joint_positions[i] == pytest.approx(lo), f"Joint {name} not clamped to min"

    def test_safety_clamp_joint_velocity(self, config):
        """When joint_velocity_limits=True, target velocities clipped to per-joint VELOCITY_LIMITS."""
        safety = SafetyController(config, n_dof=29)
        joints = G1_29DOF_JOINTS

        cmd = RobotCommand(
            joint_positions=np.zeros(29),
            joint_velocities=np.full(29, 100.0),  # exceeds all limits
            joint_torques=np.zeros(29),
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result = safety.clamp_command(cmd)
        for i, name in enumerate(joints):
            assert result.joint_velocities[i] == pytest.approx(VELOCITY_LIMITS_29DOF[name]), \
                f"Joint {name} velocity not clamped to max"

        # Negative side
        cmd_neg = RobotCommand(
            joint_positions=np.zeros(29),
            joint_velocities=np.full(29, -100.0),
            joint_torques=np.zeros(29),
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result_neg = safety.clamp_command(cmd_neg)
        for i, name in enumerate(joints):
            assert result_neg.joint_velocities[i] == pytest.approx(-VELOCITY_LIMITS_29DOF[name]), \
                f"Joint {name} negative velocity not clamped"

    def test_safety_clamp_torque(self, config):
        """When torque_limits=True, feedforward torques clipped to TORQUE_LIMITS."""
        safety = SafetyController(config, n_dof=29)
        joints = G1_29DOF_JOINTS

        cmd = RobotCommand(
            joint_positions=np.zeros(29),
            joint_velocities=np.zeros(29),
            joint_torques=np.full(29, 999.0),  # way above all torque limits
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result = safety.clamp_command(cmd)
        for i, name in enumerate(joints):
            assert result.joint_torques[i] == pytest.approx(TORQUE_LIMITS_29DOF[name]), \
                f"Joint {name} torque not clamped"

        # Negative side
        cmd_neg = RobotCommand(
            joint_positions=np.zeros(29),
            joint_velocities=np.zeros(29),
            joint_torques=np.full(29, -999.0),
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result_neg = safety.clamp_command(cmd_neg)
        for i, name in enumerate(joints):
            assert result_neg.joint_torques[i] == pytest.approx(-TORQUE_LIMITS_29DOF[name]), \
                f"Joint {name} negative torque not clamped"

    def test_safety_limits_disabled(self, config):
        """When all limit booleans are False, clamp_command() passes through unchanged."""
        config.safety.joint_position_limits = False
        config.safety.joint_velocity_limits = False
        config.safety.torque_limits = False
        safety = SafetyController(config, n_dof=29)

        original_pos = np.full(29, 100.0)
        original_vel = np.full(29, 100.0)
        original_tau = np.full(29, 999.0)
        cmd = RobotCommand(
            joint_positions=original_pos.copy(),
            joint_velocities=original_vel.copy(),
            joint_torques=original_tau.copy(),
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result = safety.clamp_command(cmd)

        np.testing.assert_array_equal(result.joint_positions, original_pos)
        np.testing.assert_array_equal(result.joint_velocities, original_vel)
        np.testing.assert_array_equal(result.joint_torques, original_tau)

    def test_clamp_does_not_modify_input(self, config):
        """clamp_command() should return a new command, not modify the input."""
        safety = SafetyController(config, n_dof=29)
        original = np.full(29, 100.0)
        cmd = RobotCommand(
            joint_positions=original.copy(),
            joint_velocities=np.zeros(29),
            joint_torques=np.zeros(29),
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        safety.clamp_command(cmd)
        np.testing.assert_array_equal(cmd.joint_positions, original)

    def test_clamp_preserves_gains(self, config):
        """clamp_command() should not alter kp/kd."""
        safety = SafetyController(config, n_dof=29)
        kp = np.full(29, 42.0)
        kd = np.full(29, 7.0)
        cmd = RobotCommand(
            joint_positions=np.full(29, 100.0),
            joint_velocities=np.full(29, 100.0),
            joint_torques=np.full(29, 999.0),
            kp=kp.copy(),
            kd=kd.copy(),
        )
        result = safety.clamp_command(cmd)
        np.testing.assert_array_equal(result.kp, kp)
        np.testing.assert_array_equal(result.kd, kd)

    def test_clamp_within_limits_unchanged(self, config):
        """Values within limits should pass through unchanged."""
        safety = SafetyController(config, n_dof=29)
        cmd = RobotCommand(
            joint_positions=np.zeros(29),  # home-ish, well within limits
            joint_velocities=np.full(29, 1.0),  # well within min vel limit (20 rad/s)
            joint_torques=np.full(29, 1.0),  # well within all torque limits (min 5 Nm)
            kp=np.full(29, 100.0),
            kd=np.full(29, 10.0),
        )
        result = safety.clamp_command(cmd)
        np.testing.assert_array_equal(result.joint_positions, cmd.joint_positions)
        np.testing.assert_array_equal(result.joint_velocities, cmd.joint_velocities)
        np.testing.assert_array_equal(result.joint_torques, cmd.joint_torques)

    def test_clamp_multi_limit_simultaneous(self, config):
        """All three limits (position, velocity, torque) clamped in a single call."""
        safety = SafetyController(config, n_dof=29)
        joints = G1_29DOF_JOINTS

        cmd = RobotCommand(
            joint_positions=np.full(29, 100.0),
            joint_velocities=np.full(29, 100.0),
            joint_torques=np.full(29, 999.0),
            kp=np.full(29, 42.0),
            kd=np.full(29, 7.0),
        )
        result = safety.clamp_command(cmd)

        for i, name in enumerate(joints):
            _, hi = JOINT_LIMITS_29DOF[name]
            assert result.joint_positions[i] == pytest.approx(hi), \
                f"Position not clamped for {name}"
            assert result.joint_velocities[i] == pytest.approx(VELOCITY_LIMITS_29DOF[name]), \
                f"Velocity not clamped for {name}"
            assert result.joint_torques[i] == pytest.approx(TORQUE_LIMITS_29DOF[name]), \
                f"Torque not clamped for {name}"

        # Gains should be untouched
        np.testing.assert_array_equal(result.kp, np.full(29, 42.0))
        np.testing.assert_array_equal(result.kd, np.full(29, 7.0))


# ---------------------------------------------------------------------------
# State Limit Check Tests
# ---------------------------------------------------------------------------

class TestCheckStateLimits:
    """Tests for check_state_limits() — monitors measured robot state."""

    @staticmethod
    def _home_state() -> RobotState:
        """Create a RobotState at home position (well within all limits)."""
        state = RobotState.zeros(29)
        state.joint_positions = np.array(
            [Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS]
        )
        return state

    def test_within_limits_no_fault(self, config):
        """State at home position (well within limits) should not trigger fault."""
        safety = SafetyController(config, n_dof=29)
        safety.start()
        state = self._home_state()
        assert safety.check_state_limits(state) is False
        assert safety.state == SystemState.RUNNING

    @pytest.mark.parametrize("field,value_fn", [
        ("joint_positions", lambda: JOINT_LIMITS_29DOF["left_hip_pitch"][1]),
        ("joint_positions", lambda: JOINT_LIMITS_29DOF["left_hip_pitch"][0]),
        ("joint_velocities", lambda: VELOCITY_LIMITS_29DOF["left_hip_pitch"]),
        ("joint_velocities", lambda: -VELOCITY_LIMITS_29DOF["left_hip_pitch"]),
        ("joint_torques", lambda: TORQUE_LIMITS_29DOF["left_hip_pitch"]),
    ], ids=["pos_high", "pos_low", "vel_high", "vel_neg", "torque"])
    def test_limit_fault_triggers_estop(self, config, field, value_fn):
        """Exceeding any limit on left_hip_pitch (index 0) triggers ESTOP."""
        safety = SafetyController(config, n_dof=29)
        safety.start()
        state = self._home_state()
        getattr(state, field)[0] = value_fn()
        assert safety.check_state_limits(state) is True
        assert safety.state == SystemState.ESTOP

    @pytest.mark.parametrize("flag,field,value_fn", [
        ("joint_position_limits", "joint_positions",
         lambda: JOINT_LIMITS_29DOF["left_hip_pitch"][1]),
        ("joint_velocity_limits", "joint_velocities",
         lambda: VELOCITY_LIMITS_29DOF["left_hip_pitch"]),
        ("torque_limits", "joint_torques",
         lambda: TORQUE_LIMITS_29DOF["left_hip_pitch"]),
    ], ids=["pos_disabled", "vel_disabled", "torque_disabled"])
    def test_disabled_limits_no_fault(self, config, flag, field, value_fn):
        """When a limit check is disabled, exceeding that limit is OK."""
        setattr(config.safety, flag, False)
        safety = SafetyController(config, n_dof=29)
        safety.start()
        state = self._home_state()
        getattr(state, field)[0] = value_fn()
        assert safety.check_state_limits(state) is False
        assert safety.state == SystemState.RUNNING

    def test_custom_threshold_fault(self, config):
        """With threshold=0.5, fault triggers at 60% of velocity limit."""
        config.safety.fault_threshold = 0.5
        safety = SafetyController(config, n_dof=29)
        safety.start()
        state = self._home_state()
        state.joint_velocities[0] = 0.6 * VELOCITY_LIMITS_29DOF["left_hip_pitch"]
        assert safety.check_state_limits(state) is True
        assert safety.state == SystemState.ESTOP

    def test_just_below_threshold_no_fault(self, config):
        """Value just below threshold should not trigger fault."""
        safety = SafetyController(config, n_dof=29)
        safety.start()
        state = self._home_state()
        state.joint_velocities[0] = 0.94 * VELOCITY_LIMITS_29DOF["left_hip_pitch"]
        assert safety.check_state_limits(state) is False
        assert safety.state == SystemState.RUNNING

    def test_already_estop_returns_false(self, config):
        """When already in ESTOP, check_state_limits returns False (no spam)."""
        safety = SafetyController(config, n_dof=29)
        safety.start()
        safety.estop()
        state = self._home_state()
        state.joint_velocities[0] = VELOCITY_LIMITS_29DOF["left_hip_pitch"]
        assert safety.check_state_limits(state) is False

    def test_idle_returns_false(self, config):
        """When in IDLE (pre-operational), check_state_limits skips."""
        safety = SafetyController(config, n_dof=29)
        state = self._home_state()
        state.joint_velocities[0] = VELOCITY_LIMITS_29DOF["left_hip_pitch"]
        assert safety.check_state_limits(state) is False
        assert safety.state == SystemState.IDLE

    def test_fault_logs_joint_name(self, config, caplog):
        """Fault should log an error message containing the joint name."""
        import logging
        safety = SafetyController(config, n_dof=29)
        safety.start()
        state = self._home_state()
        state.joint_velocities[0] = VELOCITY_LIMITS_29DOF["left_hip_pitch"]
        with caplog.at_level(logging.ERROR, logger="unitree_launcher.control.safety"):
            safety.check_state_limits(state)
        assert "left_hip_pitch" in caplog.text
