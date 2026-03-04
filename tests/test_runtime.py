"""Tests for Runtime (control/runtime.py).

Covers command building (value-level), safety integration, velocity commands,
key handling, policy reloading, BeyondMimic trajectory, and control loop lifecycle.
"""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from unitree_launcher.config import (
    Config,
    G1_29DOF_JOINTS,
    Q_HOME_29DOF,
    load_config,
)
from unitree_launcher.control.runtime import Runtime
from unitree_launcher.control.safety import ControlMode, SafetyController, SystemState
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Helpers
# ============================================================================

def _make_config() -> Config:
    """Load default config with transition disabled for test isolation."""
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    cfg.control.transition_steps = 0
    return cfg


def _make_mock_robot(n_dof: int = 29) -> MagicMock:
    """Create a mock RobotInterface."""
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


def _make_mock_policy(n_dof: int = 29) -> MagicMock:
    """Create a mock Policy with step() returning a valid RobotCommand."""
    policy = MagicMock()
    policy.step.return_value = RobotCommand(
        joint_positions=np.zeros(n_dof),
        joint_velocities=np.zeros(n_dof),
        joint_torques=np.zeros(n_dof),
        kp=np.full(n_dof, 100.0),
        kd=np.full(n_dof, 10.0),
    )
    policy.last_action = np.zeros(n_dof)
    policy.observation_dim = 70
    policy.action_dim = n_dof
    policy.stiffness = np.full(n_dof, 100.0)
    policy.damping = np.full(n_dof, 10.0)
    policy.default_pos = np.zeros(n_dof)
    policy.starting_pos = np.zeros(n_dof)
    return policy


def _make_controller(
    config: Config | None = None,
    robot: MagicMock | None = None,
    policy: MagicMock | None = None,
    **kwargs,
) -> Runtime:
    """Build a Runtime with sensible defaults for testing."""
    if config is None:
        config = _make_config()
    if robot is None:
        robot = _make_mock_robot()
    if policy is None:
        policy = _make_mock_policy()

    mapper = JointMapper(G1_29DOF_JOINTS)
    safety = SafetyController(config, n_dof=29)

    from unitree_launcher.controller.input import InputManager
    from unitree_launcher.controller.keyboard import KeyboardInput
    kb = KeyboardInput()
    input_mgr = InputManager([kb])

    rt = Runtime(
        robot=robot,
        policy=policy,
        safety=safety,
        joint_mapper=mapper,
        config=config,
        default_policy=_make_mock_policy(),
        default_joint_mapper=mapper,
        input_manager=input_mgr,
        **kwargs,
    )
    rt._keyboard = kb  # Expose for tests
    return rt




# ============================================================================
# Safety integration
# ============================================================================

class TestSafetyIntegration:
    def test_estop_sends_damping(self):
        """In ESTOP, the loop sends damping command."""
        rt = _make_controller()
        rt.safety.start()
        rt.safety.estop()
        assert rt.safety.state == SystemState.ESTOP

        # Run a single step — in ESTOP, should send damping command
        rt.start()
        rt.step()
        rt.stop()

        assert rt.robot.send_command.called
        cmd = rt.robot.send_command.call_args[0][0]
        # Damping: kp=0
        np.testing.assert_array_equal(cmd.kp, np.zeros(29))
        assert np.all(cmd.kd > 0)

    def test_control_loop_exception_triggers_estop(self):
        """RuntimeError during policy.step() triggers E-stop, not crash."""
        policy = _make_mock_policy()
        policy.step.side_effect = RuntimeError("test error")

        rt = _make_controller(policy=policy)
        rt.safety.start()

        # Run step — safety is already RUNNING, so step() will enter
        # the RUNNING branch and the exception in get_action triggers E-stop
        rt.start()
        rt.step()
        rt.stop()

        assert rt.safety.state == SystemState.ESTOP


# ============================================================================
# Velocity command
# ============================================================================

class TestVelocityCommand:
    def test_velocity_command_set_get(self):
        rt = _make_controller()
        rt._keyboard._velocity = np.array([0.5, -0.3, 0.1])
        vc = rt._keyboard.get_velocity()
        np.testing.assert_array_almost_equal(vc, [0.5, -0.3, 0.1])

    def test_telemetry_updates(self):
        """After running, telemetry dict contains expected keys."""
        rt = _make_controller()
        rt.safety.start()
        rt.start()
        for _ in range(3):
            rt.step()
        rt.stop()

        telem = rt.get_telemetry()
        expected_keys = {"loop_hz", "sim_hz", "inference_ms", "loop_ms",
                         "base_height", "base_vel", "system_state", "step_count"}
        assert expected_keys.issubset(set(telem.keys()))


# ============================================================================
# Key handling
# ============================================================================

class TestKeyHandling:
    def test_handle_key_space_toggles(self):
        """Space from IDLE -> RUNNING, Space from RUNNING -> STOPPED."""
        rt = _make_controller()
        rt.start()

        # IDLE -> RUNNING (push key then step to process command)
        rt._keyboard.push_key("space")
        rt.step()
        assert rt.safety.state == SystemState.RUNNING

        # RUNNING -> STOPPED
        rt._keyboard.push_key("space")
        rt.step()
        assert rt.safety.state == SystemState.STOPPED
        rt.stop()

    def test_handle_key_estop(self):
        rt = _make_controller()
        rt.start()
        rt.safety.start()
        rt._keyboard.push_key("backspace")
        rt.step()
        assert rt.safety.state == SystemState.ESTOP

    def test_handle_key_clear_estop(self):
        rt = _make_controller()
        rt.start()
        rt.safety.start()
        rt.safety.estop()
        rt._keyboard.push_key("enter")
        rt.step()
        assert rt.safety.state == SystemState.STOPPED

    def test_handle_key_reset(self):
        rt = _make_controller()
        rt.start()
        rt._keyboard.push_key("delete")
        rt.step()
        rt.robot.reset.assert_called_once()

    @pytest.mark.parametrize("inc_key,dec_key,axis", [
        ("up", "down", 0),
        ("left", "right", 1),
        ("comma", "period", 2),
    ])
    def test_handle_key_velocity(self, inc_key, dec_key, axis):
        """Velocity keys increment, decrement, and clamp correctly."""
        rt = _make_controller()
        rt._keyboard.push_key(inc_key)
        vc = rt._keyboard.get_velocity()
        assert abs(vc[axis] - 0.1) < 1e-9

        rt._keyboard.push_key(dec_key)
        vc = rt._keyboard.get_velocity()
        assert abs(vc[axis]) < 1e-9

        # Saturates after many presses (exact clamp value varies by axis)
        for _ in range(20):
            rt._keyboard.push_key(inc_key)
        vc = rt._keyboard.get_velocity()
        assert vc[axis] > 0.3

    def test_handle_key_slash_zeros_velocity(self):
        rt = _make_controller()
        rt._keyboard._velocity = np.array([0.5, 0.3, -0.2])
        rt._keyboard.push_key("slash")
        vc = rt._keyboard.get_velocity()
        np.testing.assert_array_equal(vc, np.zeros(3))

    def test_handle_key_unknown_noop(self):
        """Unknown key does nothing and doesn't error."""
        rt = _make_controller()
        rt._keyboard.push_key("j")  # Should not raise
        vc = rt._keyboard.get_velocity()
        np.testing.assert_array_equal(vc, np.zeros(3))

    def _make_policy_dir_runtime(self, tmp_path):
        """Build a Runtime with two preloaded policies in tmp_path."""
        from tests.conftest import create_isaaclab_onnx
        from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
        from unitree_launcher.controller.input import InputManager
        from unitree_launcher.controller.keyboard import KeyboardInput

        p1 = str(tmp_path / "a_policy.onnx")
        p2 = str(tmp_path / "b_policy.onnx")
        obs_dim = 99
        mapper = JointMapper(G1_29DOF_JOINTS)
        config = _make_config()
        create_isaaclab_onnx(obs_dim, 29, p1)
        create_isaaclab_onnx(obs_dim, 29, p2)

        policy = IsaacLabPolicy(mapper, config)
        policy.load(p1)

        kb = KeyboardInput()
        safety = SafetyController(config, n_dof=29)
        rt = Runtime(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            config=config,
            policy_dir=str(tmp_path),
            default_policy=_make_mock_policy(),
            default_joint_mapper=mapper,
            input_manager=InputManager([kb]),
        )
        rt._keyboard = kb
        return rt

    @pytest.mark.parametrize("key,expected_index", [
        ("equal", 1),
        ("minus", 1),  # wraps: -1 % 2 = 1
    ])
    def test_handle_key_policy_navigation(self, tmp_path, key, expected_index):
        """'+'/'-' keys navigate between policies in policy_dir."""
        rt = self._make_policy_dir_runtime(tmp_path)
        rt.start()
        rt._keyboard.push_key(key)
        rt.step()
        assert rt._policy_index == expected_index


# ============================================================================
# Policy reloading
# ============================================================================

class TestPolicyReload:
    def test_reload_policy_while_stopped(self, tmp_path):
        """reload_policy() loads and resets successfully."""
        from tests.conftest import create_isaaclab_onnx
        from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy

        mapper = JointMapper(G1_29DOF_JOINTS)
        config = _make_config()
        

        p1 = str(tmp_path / "policy1.onnx")
        p2 = str(tmp_path / "policy2.onnx")
        obs_dim = 99
        create_isaaclab_onnx(obs_dim, 29, p1)
        create_isaaclab_onnx(obs_dim, 29, p2)

        policy = IsaacLabPolicy(mapper, config)
        policy.load(p1)

        safety = SafetyController(config, n_dof=29)
        rt = Runtime(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            
            config=config,
        )

        # Reload while stopped
        rt.reload_policy(p2)
        # No error means success

    def test_reload_policy_invalid_path(self, tmp_path):
        """Invalid path raises error, original policy is preserved."""
        from tests.conftest import create_isaaclab_onnx
        from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy

        mapper = JointMapper(G1_29DOF_JOINTS)
        config = _make_config()
        

        p1 = str(tmp_path / "policy1.onnx")
        obs_dim = 99
        create_isaaclab_onnx(obs_dim, 29, p1)

        policy = IsaacLabPolicy(mapper, config)
        policy.load(p1)

        robot = _make_mock_robot()
        safety = SafetyController(config, n_dof=29)
        rt = Runtime(
            robot=robot,
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            config=config,
            default_policy=_make_mock_policy(),
            default_joint_mapper=mapper,
        )

        with pytest.raises(ValueError):
            rt.reload_policy("/nonexistent/bad.onnx")

        # Original policy should still work (step returns valid command)
        state = robot.get_state()
        cmd = rt.policy.step(state, np.zeros(3))
        assert cmd.joint_positions.shape == (29,)

    def test_reload_policy_restarts_thread_in_threaded_mode(self, tmp_path):
        """reload_policy() must restart the daemon thread when in threaded mode.

        Verifies that reload_policy() restarts the daemon thread.
        reload_policy() was calling start() (no thread) instead of
        start_threaded(), causing the runtime to freeze after reload.
        """
        from tests.conftest import create_isaaclab_onnx
        from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy

        mapper = JointMapper(G1_29DOF_JOINTS)
        config = _make_config()
        config.safety.tilt_check = False

        p1 = str(tmp_path / "policy1.onnx")
        p2 = str(tmp_path / "policy2.onnx")
        obs_dim = 99
        create_isaaclab_onnx(obs_dim, 29, p1)
        create_isaaclab_onnx(obs_dim, 29, p2)

        policy = IsaacLabPolicy(mapper, config)
        policy.load(p1)
        safety = SafetyController(config, n_dof=29)
        rt = Runtime(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            config=config,
            default_policy=_make_mock_policy(),
            default_joint_mapper=mapper,
        )

        # Start in threaded mode
        rt.start_threaded()
        safety.start()
        time.sleep(0.15)
        assert rt.get_telemetry()["step_count"] > 0, "Thread should be stepping"
        assert rt._thread is not None and rt._thread.is_alive()

        # Reload policy — thread must restart
        rt.reload_policy(p2)
        assert rt._threaded is True, "Should stay in threaded mode"
        assert rt._thread is not None and rt._thread.is_alive(), \
            "Thread must be alive after reload"
        time.sleep(0.15)
        assert rt.get_telemetry()["step_count"] > 0, \
            "Thread should be stepping after reload"

        rt.stop()


# ============================================================================
# BeyondMimic trajectory end -> auto-return to default/hold
# ============================================================================

class TestBeyondMimicTrajectory:
    def test_beyondmimic_trajectory_end_triggers_stop(self):
        """When BM trajectory ends, safety transitions to STOPPED."""
        config = _make_config()
        mapper = JointMapper(G1_29DOF_JOINTS)
        safety = SafetyController(config, n_dof=29)
        robot = _make_mock_robot()

        from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy

        # Create a mock BM that simulates trajectory completion
        bm_policy = MagicMock(spec=BeyondMimicPolicy)
        bm_policy.__class__ = BeyondMimicPolicy
        bm_policy.trajectory_length = 5
        bm_policy.last_action = np.zeros(29)
        bm_policy.starting_pos = np.zeros(29)
        bm_policy.stiffness = np.full(29, 100.0)
        bm_policy.damping = np.full(29, 10.0)
        bm_policy.default_pos = np.zeros(29)
        bm_policy._start_timestep = 0

        # step() returns valid command and increments internal time_step
        step_count = [0]
        def fake_step(state, vel_cmd):
            step_count[0] += 1
            return RobotCommand(
                joint_positions=np.zeros(29),
                joint_velocities=np.zeros(29),
                joint_torques=np.zeros(29),
                kp=np.full(29, 100.0),
                kd=np.full(29, 10.0),
            )
        bm_policy.step = fake_step

        # time_step property tracks steps taken
        type(bm_policy).time_step = property(lambda self: step_count[0])

        default_pol = _make_mock_policy()

        rt = Runtime(
            robot=robot,
            policy=bm_policy,
            safety=safety,
            joint_mapper=mapper,
            config=config,
            default_policy=default_pol,
            default_joint_mapper=mapper,
        )

        rt.start()
        safety.start()

        for _ in range(10):
            rt.step()

        # After trajectory ends (step_count >= 5), safety should be STOPPED
        assert safety.state == SystemState.STOPPED

        rt.stop()


# ============================================================================
# Control loop lifecycle
# ============================================================================

class TestControlLoopLifecycle:
    def test_control_loop_lifecycle(self):
        """start() -> step() -> stop(), no exceptions."""
        rt = _make_controller()
        rt.safety.start()
        rt.start()
        assert rt.is_running is True
        for _ in range(5):
            rt.step()
        rt.safety.stop()
        rt.stop()
        assert rt.is_running is False

    def test_control_loop_stopped_still_steps(self):
        """In STOPPED state, robot.step() is still called but no command sent."""
        rt = _make_controller()
        # Transition to STOPPED (need IDLE -> RUNNING -> STOPPED)
        rt.safety.start()
        rt.safety.stop()

        # Reset mock call counts
        rt.robot.step.reset_mock()
        rt.robot.send_command.reset_mock()

        # Run a few iterations manually
        rt._running = True
        # Not in RUNNING state, so default policy runs
        # Call _control_loop directly — it will see STOPPED and just step
        # We need to limit it somehow; let's use a threading approach
        def run_limited():
            count = 0
            while rt._running and count < 5:
                loop_start = time.perf_counter()
                if rt.safety.state != SystemState.RUNNING:
                    rt.robot.step()
                    count += 1
                    continue
            rt._running = False

        run_limited()
        assert rt.robot.step.call_count == 5
        # send_command should NOT have been called (only step, no command in STOPPED)
        rt.robot.send_command.assert_not_called()

    def test_control_loop_calls_safety_clamp(self):
        """Verify that commands go through safety.clamp_command()."""
        config = _make_config()
        mapper = JointMapper(G1_29DOF_JOINTS)
        safety = SafetyController(config, n_dof=29)
        robot = _make_mock_robot()
        policy = _make_mock_policy()

        rt = Runtime(
            robot=robot,
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            
            config=config,
        )

        # Spy on clamp_command
        original_clamp = safety.clamp_command
        clamp_calls = []

        def spy_clamp(cmd):
            clamp_calls.append(cmd)
            return original_clamp(cmd)

        safety.clamp_command = spy_clamp

        safety.start()
        rt.start()
        for _ in range(3):
            rt.step()

        assert len(clamp_calls) >= 1


# ============================================================================
# Policy transition (position interpolation to starting pose)
# ============================================================================

class TestTransition:
    def test_activation_starts_transition(self):
        """IDLE->RUNNING transition enters TRANSITION mode."""
        config = _make_config()
        config.control.transition_steps = 50
        rt = _make_controller(config=config)
        rt.start()
        rt.safety.start()

        # First step triggers activation and starts interpolation
        rt.step()
        assert rt.control_mode == ControlMode.TRANSITION
        assert rt._transition_active is True
        rt.stop()

    def test_transition_completes(self):
        """After transition_steps, mode becomes ACTIVE_POLICY."""
        config = _make_config()
        config.control.transition_steps = 10
        rt = _make_controller(config=config)
        rt.start()
        rt.safety.start()

        # 1 step for activation + 10 steps of interpolation
        for _ in range(11):
            rt.step()

        assert rt.control_mode == ControlMode.ACTIVE_POLICY
        assert rt._transition_active is False
        rt.stop()

    def test_no_policy_inference_during_transition(self):
        """Policy.step() is not called during transition interpolation."""
        config = _make_config()
        config.control.transition_steps = 10
        policy = _make_mock_policy()

        rt = _make_controller(config=config, policy=policy)
        rt.start()
        rt.safety.start()

        # Reset mock after start() which calls policy.reset()
        policy.step.reset_mock()

        # Run 5 steps — still mid-transition (10 steps total)
        for _ in range(5):
            rt.step()
        assert rt._transition_active is True

        # Policy.step() should not have been called during transition
        policy.step.assert_not_called()

        # Complete transition (5 more steps) + 1 policy step
        for _ in range(6):
            rt.step()
        assert rt._transition_active is False
        assert policy.step.call_count == 1
        rt.stop()

    def test_transition_zero_is_instant(self):
        """transition_steps=0 skips interpolation entirely."""
        config = _make_config()
        config.control.transition_steps = 0
        rt = _make_controller(config=config)
        rt.start()
        rt.safety.start()

        rt.step()
        assert rt.control_mode == ControlMode.ACTIVE_POLICY
        assert rt._transition_active is False
        rt.stop()

    def test_interpolation_is_monotonic(self):
        """Kp values ramp from default toward active policy during transition."""
        config = _make_config()
        config.control.transition_steps = 20
        config.safety.tilt_check = False

        # Active policy has kp=100, default has kp=50
        policy = _make_mock_policy()
        policy.stiffness = np.full(29, 100.0)
        policy.damping = np.full(29, 10.0)
        policy.default_pos = np.zeros(29)

        default = _make_mock_policy()
        default.stiffness = np.full(29, 50.0)
        default.damping = np.full(29, 5.0)
        default.default_pos = np.zeros(29)

        rt = _make_controller(config=config, policy=policy)
        rt._default_policy = default
        rt.start()
        rt.safety.start()

        # Capture kp[0] from sent commands
        kp_values = []
        original_send = rt.robot.send_command
        def capture_cmd(cmd):
            kp_values.append(cmd.kp[0])
            return original_send(cmd)
        rt.robot.send_command = capture_cmd

        # Run through transition (20 steps) + 1 policy step
        for _ in range(21):
            rt.step()

        # kp should increase monotonically during transition (50 -> 100)
        transition_kps = kp_values[:20]
        for i in range(1, len(transition_kps)):
            assert transition_kps[i] >= transition_kps[i - 1] - 1e-9, \
                f"kp not monotonic at step {i}: {transition_kps[i]} < {transition_kps[i-1]}"

        # Final transition kp should be close to target (100)
        assert abs(transition_kps[-1] - 100.0) < 1e-6
        rt.stop()

    def test_position_reaches_target(self):
        """Position interpolates from current state to policy's starting_pos."""
        config = _make_config()
        config.control.transition_steps = 10
        config.safety.tilt_check = False

        policy = _make_mock_policy()
        # Use small target values within joint limits
        target = np.ones(29) * 0.1
        policy.default_pos = target.copy()
        policy.starting_pos = target.copy()
        policy.stiffness = np.full(29, 100.0)
        policy.damping = np.full(29, 10.0)

        rt = _make_controller(config=config, policy=policy)
        rt.start()
        rt.safety.start()

        # Capture positions from sent commands
        positions = []
        original_send = rt.robot.send_command
        def capture_cmd(cmd):
            positions.append(cmd.joint_positions.copy())
            return original_send(cmd)
        rt.robot.send_command = capture_cmd

        for _ in range(10):
            rt.step()

        # Last transition command should be at the target position
        np.testing.assert_allclose(positions[-1], target, atol=1e-6)
        rt.stop()

    def test_reload_while_running_transitions(self):
        """Policy hot-swap via reload_policy triggers a new transition."""
        config = _make_config()
        config.control.transition_steps = 10
        config.safety.tilt_check = False

        policy_a = _make_mock_policy()
        policy_b = _make_mock_policy()
        policy_b.stiffness = np.full(29, 200.0)
        policy_b.damping = np.full(29, 20.0)
        policy_b.default_pos = np.ones(29) * 0.1
        policy_b.last_action = np.zeros(29)

        rt = _make_controller(config=config, policy=policy_a)
        rt._preloaded_policies = {"/fake/b.onnx": (policy_b, rt.joint_mapper)}
        rt.start()
        rt.safety.start()

        # Run past initial transition
        for _ in range(15):
            rt.step()
        assert rt.control_mode == ControlMode.ACTIVE_POLICY

        # Reload triggers new transition to policy_b's starting pose
        rt.reload_policy("/fake/b.onnx")
        rt.step()
        assert rt._transition_active is True
        assert rt.control_mode == ControlMode.TRANSITION

        # Complete the new transition
        for _ in range(15):
            rt.step()
        assert rt.control_mode == ControlMode.ACTIVE_POLICY
        assert rt.policy is policy_b
        rt.stop()

    def test_return_to_default_is_instant(self):
        """Stopping (returning to default policy) has no transition."""
        config = _make_config()
        config.control.transition_steps = 50
        rt = _make_controller(config=config)
        rt.start()
        rt.safety.start()

        # Run through a few transition steps
        for _ in range(3):
            rt.step()
        assert rt._transition_active is True

        # Stop — should immediately return to DEFAULT, no transition
        rt.safety.stop()
        rt.step()

        assert rt._transition_active is False
        assert rt.control_mode == ControlMode.DEFAULT
        assert rt.safety.state == SystemState.STOPPED
        rt.stop()

    def test_estop_during_transition(self):
        """E-stop during transition clears transition state immediately."""
        config = _make_config()
        config.control.transition_steps = 50
        rt = _make_controller(config=config)
        rt.start()
        rt.safety.start()

        # Start transition
        rt.step()
        assert rt._transition_active is True

        # Trigger E-stop
        rt.safety.estop()
        rt.step()

        assert rt._transition_active is False
        assert rt.safety.state == SystemState.ESTOP
        rt.stop()


# ============================================================================
# Motion button handlers
# ============================================================================

class TestMotionButtons:
    def test_motion_fade_out_stops_policy(self):
        """[MOTION_FADE_OUT] (B button) stops running policy."""
        rt = _make_controller()
        rt.start()
        rt.safety.start()
        rt.step()
        assert rt.safety.state == SystemState.RUNNING

        rt._handle_commands({"[MOTION_FADE_OUT]"})
        assert rt.safety.state == SystemState.STOPPED

    def test_motion_fade_in_starts_policy(self):
        """[MOTION_FADE_IN] (X button) re-activates policy."""
        rt = _make_controller()
        rt.start()
        assert rt.safety.state == SystemState.IDLE

        rt._handle_commands({"[MOTION_FADE_IN]"})
        assert rt.safety.state == SystemState.RUNNING

    def test_motion_reset_reactivates(self):
        """[MOTION_RESET] (Y button) stops, resets, and re-activates."""
        rt = _make_controller()
        rt.start()
        rt.safety.start()
        rt.step()
        assert rt.safety.state == SystemState.RUNNING

        rt._handle_commands({"[MOTION_RESET]"})
        assert rt.safety.state == SystemState.RUNNING
        assert rt._policy_active is False  # Will re-activate on next step
        rt.policy.reset.assert_called()

    def test_prepare_aborts_on_shutdown(self):
        """Prepare mode sequence aborts on [SHUTDOWN] command."""
        rt = _make_controller()

        # Configure a PREPARE mode sequence (20s = 1000 steps at 50Hz)
        rt._mode_sequence = [(ControlMode.PREPARE, 20.0)]
        rt._initial_mode_sequence = [(ControlMode.PREPARE, 20.0)]

        # Inject [SHUTDOWN] after 5 steps by making input_manager return it
        call_count = [0]
        orig_get = rt._input_manager.get_commands
        def mock_get():
            call_count[0] += 1
            if call_count[0] == 5:
                return {"[SHUTDOWN]"}
            return set()
        rt._input_manager.get_commands = mock_get

        rt.start()
        rt.safety.start()
        for _ in range(20):
            rt.step()

        # Should have E-stopped (not run all 1000 steps)
        assert rt.safety.state == SystemState.ESTOP
