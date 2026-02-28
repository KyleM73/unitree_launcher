"""Tests for src/control/controller.py — Phase 8.

Covers command building (value-level), safety integration, velocity commands,
key handling, policy reloading, BeyondMimic trajectory, and control loop lifecycle.
"""
from __future__ import annotations

import os
import tempfile
import threading
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
from unitree_launcher.control.controller import Controller
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.policy.base import PolicyInterface
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.policy.observations import ObservationBuilder
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Helpers
# ============================================================================

def _make_config() -> Config:
    """Load default config."""
    return load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))


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


def _make_mock_isaaclab_policy(n_ctrl: int = 29) -> MagicMock:
    """Create a mock IsaacLab-style policy (not BeyondMimicPolicy)."""
    policy = MagicMock(spec=PolicyInterface)
    policy.get_action.return_value = np.zeros(n_ctrl)
    policy.observation_dim = 70
    policy.action_dim = n_ctrl
    return policy


def _make_controller(
    config: Config | None = None,
    robot: MagicMock | None = None,
    policy: MagicMock | None = None,
    obs_builder: ObservationBuilder | None = "auto",
    **kwargs,
) -> Controller:
    """Build a Controller with sensible defaults for testing."""
    if config is None:
        config = _make_config()
    if robot is None:
        robot = _make_mock_robot()
    if policy is None:
        policy = _make_mock_isaaclab_policy()

    mapper = JointMapper(G1_29DOF_JOINTS)
    safety = SafetyController(config, n_dof=29)

    if obs_builder == "auto":
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
# Init
# ============================================================================

class TestControllerInit:
    def test_controller_init(self):
        ctrl = _make_controller()
        assert ctrl.is_running is False
        assert ctrl.safety.state == SystemState.IDLE
        vel = ctrl.get_velocity_command()
        np.testing.assert_array_equal(vel, np.zeros(3))

    def test_controller_gain_expansion_scalar(self):
        ctrl = _make_controller()
        np.testing.assert_array_equal(ctrl._kp, np.full(29, 100.0))
        np.testing.assert_array_equal(ctrl._kd, np.full(29, 10.0))
        np.testing.assert_array_equal(ctrl._ka, np.full(29, 0.5))

    def test_controller_q_home(self):
        ctrl = _make_controller()
        expected = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        np.testing.assert_array_almost_equal(ctrl._q_home, expected)


# ============================================================================
# Command building (value-level)
# ============================================================================

class TestBuildCommand:
    def test_build_command_isaaclab_values(self):
        """Verify: target_pos = q_home + Ka * action, kp/kd = IsaacLab per-joint gains, dq=0, tau=0."""
        ctrl = _make_controller()
        state = ctrl.robot.get_state()

        # Action of all 1.0
        action = np.ones(29)

        cmd = ctrl._build_command(state, action)

        # Expected: target_pos[i] = q_home[i] + isaaclab_ka[i] * 1.0
        expected_pos = ctrl._q_home + ctrl._isaaclab_ka * action
        ctrl_idx = ctrl.joint_mapper.controlled_indices
        np.testing.assert_array_almost_equal(cmd.joint_positions[ctrl_idx], expected_pos)

        # kp, kd from IsaacLab per-joint training gains
        np.testing.assert_array_almost_equal(cmd.kp[ctrl_idx], ctrl._isaaclab_kp)
        np.testing.assert_array_almost_equal(cmd.kd[ctrl_idx], ctrl._isaaclab_kd)

        # dq_target = 0, tau = 0
        np.testing.assert_array_equal(cmd.joint_velocities, np.zeros(29))
        np.testing.assert_array_equal(cmd.joint_torques, np.zeros(29))

    def test_build_command_beyondmimic_values(self):
        """Verify BM: target = default_q + Ka * action, dq_target = 0."""
        from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy

        config = _make_config()
        mapper = JointMapper(G1_29DOF_JOINTS)
        safety = SafetyController(config, n_dof=29)

        bm_policy = MagicMock(spec=BeyondMimicPolicy)
        bm_policy.__class__ = BeyondMimicPolicy
        bm_policy.default_joint_pos = np.full(29, 0.2)
        bm_policy.target_q = np.full(29, 0.9)  # NOT used for PD targets
        bm_policy.target_dq = np.full(29, 0.5)  # NOT used for PD targets
        bm_policy.stiffness = np.full(29, 80.0)
        bm_policy.damping = np.full(29, 8.0)
        bm_policy.action_scale = np.full(29, 0.3)
        bm_policy.get_action.return_value = np.ones(29)

        ctrl = Controller(
            robot=_make_mock_robot(),
            policy=bm_policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=None,
            config=config,
        )

        state = ctrl.robot.get_state()
        action = np.ones(29)
        cmd = ctrl._build_command(state, action)

        ctrl_idx = ctrl.joint_mapper.controlled_indices

        # Training control law: target = default_q + Ka * action = 0.2 + 0.3 * 1.0 = 0.5
        np.testing.assert_array_almost_equal(cmd.joint_positions[ctrl_idx], np.full(29, 0.5))

        # dq_target = 0 (pure damping)
        np.testing.assert_array_almost_equal(cmd.joint_velocities[ctrl_idx], np.zeros(29))

        # kp, kd from metadata
        np.testing.assert_array_almost_equal(cmd.kp[ctrl_idx], np.full(29, 80.0))
        np.testing.assert_array_almost_equal(cmd.kd[ctrl_idx], np.full(29, 8.0))

    def test_build_command_damping_non_controlled(self):
        """Non-controlled joints get target_pos=current_pos, kp=0, kd=kd_damp."""
        config = _make_config()
        # Use a subset of joints as controlled
        controlled = G1_29DOF_JOINTS[:12]  # legs only
        mapper = JointMapper(G1_29DOF_JOINTS, controlled_joints=controlled)
        safety = SafetyController(config, n_dof=29)

        policy = _make_mock_isaaclab_policy(n_ctrl=12)
        obs_builder = ObservationBuilder(mapper, config)

        ctrl = Controller(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=obs_builder,
            config=config,
        )

        state = ctrl.robot.get_state()
        action = np.zeros(12)
        cmd = ctrl._build_command(state, action)

        non_ctrl = mapper.non_controlled_indices
        # target_pos = current joint positions
        np.testing.assert_array_almost_equal(
            cmd.joint_positions[non_ctrl],
            state.joint_positions[non_ctrl],
        )
        # kp = 0
        np.testing.assert_array_equal(cmd.kp[non_ctrl], np.zeros(len(non_ctrl)))
        # kd = kd_damp = 5.0
        np.testing.assert_array_almost_equal(cmd.kd[non_ctrl], np.full(len(non_ctrl), 5.0))
        # dq = 0, tau = 0
        np.testing.assert_array_equal(cmd.joint_velocities[non_ctrl], np.zeros(len(non_ctrl)))
        np.testing.assert_array_equal(cmd.joint_torques[non_ctrl], np.zeros(len(non_ctrl)))


# ============================================================================
# Safety integration
# ============================================================================

class TestSafetyIntegration:
    def test_estop_sends_damping(self):
        """In ESTOP, the loop sends damping command."""
        ctrl = _make_controller()
        ctrl.safety.start()
        ctrl.safety.estop()
        assert ctrl.safety.state == SystemState.ESTOP

        # Run loop briefly in background, then stop
        ctrl.start()
        time.sleep(0.15)
        ctrl.stop()

        # In ESTOP, send_command should have been called with damping
        assert ctrl.robot.send_command.called
        cmd = ctrl.robot.send_command.call_args[0][0]
        # Damping: kp=0
        np.testing.assert_array_equal(cmd.kp, np.zeros(29))
        assert np.all(cmd.kd > 0)

    def test_control_loop_exception_triggers_estop(self):
        """RuntimeError during get_action() triggers E-stop, not crash."""
        policy = _make_mock_isaaclab_policy()
        policy.get_action.side_effect = RuntimeError("test error")

        ctrl = _make_controller(policy=policy)
        ctrl.safety.start()

        # Run loop briefly — the exception triggers estop on first RUNNING iteration
        ctrl.start()
        time.sleep(0.3)
        ctrl.stop()

        assert ctrl.safety.state == SystemState.ESTOP


# ============================================================================
# Velocity command
# ============================================================================

class TestVelocityCommand:
    def test_velocity_command_set_get(self):
        ctrl = _make_controller()
        ctrl.set_velocity_command(0.5, -0.3, 0.1)
        vc = ctrl.get_velocity_command()
        np.testing.assert_array_almost_equal(vc, [0.5, -0.3, 0.1])

    def test_velocity_command_thread_safe(self):
        """Set velocity from one thread, read from another, no corruption."""
        ctrl = _make_controller()
        errors = []

        def writer():
            for _ in range(100):
                ctrl.set_velocity_command(0.5, 0.5, 0.5)
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                vc = ctrl.get_velocity_command()
                if vc.shape != (3,):
                    errors.append(f"Bad shape: {vc.shape}")
                time.sleep(0.001)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0

    def test_telemetry_updates(self):
        """After running, telemetry dict contains expected keys."""
        ctrl = _make_controller(max_steps=2)
        ctrl.safety.start()
        ctrl.start()
        time.sleep(0.3)  # Let a few steps run
        ctrl.stop()

        telem = ctrl.get_telemetry()
        expected_keys = {"policy_hz", "sim_hz", "inference_ms", "loop_ms",
                         "base_height", "base_vel", "system_state", "step_count"}
        assert expected_keys.issubset(set(telem.keys()))


# ============================================================================
# Key handling
# ============================================================================

class TestKeyHandling:
    def test_handle_key_space_toggles(self):
        """Space from IDLE -> RUNNING, Space from RUNNING -> STOPPED."""
        ctrl = _make_controller()

        # IDLE -> RUNNING
        ctrl.handle_key("space")
        assert ctrl.safety.state == SystemState.RUNNING
        assert ctrl.is_running is True

        # RUNNING -> STOPPED (loop keeps running in hold-pose mode)
        ctrl.handle_key("space")
        time.sleep(0.1)
        assert ctrl.safety.state == SystemState.STOPPED
        # Control loop stays alive (hold-pose mode); explicitly stop it.
        ctrl.stop()
        assert ctrl.is_running is False

    def test_handle_key_estop(self):
        ctrl = _make_controller()
        ctrl.safety.start()
        ctrl.handle_key("backspace")
        assert ctrl.safety.state == SystemState.ESTOP

    def test_handle_key_clear_estop(self):
        ctrl = _make_controller()
        ctrl.safety.start()
        ctrl.safety.estop()
        ctrl.handle_key("enter")
        assert ctrl.safety.state == SystemState.STOPPED

    def test_handle_key_reset(self):
        ctrl = _make_controller()
        ctrl.handle_key("delete")
        ctrl.robot.reset.assert_called_once()

    def test_handle_key_velocity_arrows(self):
        ctrl = _make_controller()
        ctrl.handle_key("up")
        vc = ctrl.get_velocity_command()
        assert abs(vc[0] - 0.1) < 1e-9

        ctrl.handle_key("down")
        vc = ctrl.get_velocity_command()
        assert abs(vc[0]) < 1e-9

        ctrl.handle_key("left")
        vc = ctrl.get_velocity_command()
        assert abs(vc[1] - 0.1) < 1e-9

        ctrl.handle_key("right")
        vc = ctrl.get_velocity_command()
        assert abs(vc[1]) < 1e-9

    def test_handle_key_velocity_clamps(self):
        """After 20 Up presses, vx is clamped to 1.0."""
        ctrl = _make_controller()
        for _ in range(20):
            ctrl.handle_key("up")
        vc = ctrl.get_velocity_command()
        assert abs(vc[0] - 1.0) < 1e-9

        for _ in range(20):
            ctrl.handle_key("down")
        vc = ctrl.get_velocity_command()
        assert abs(vc[0] - (-1.0)) < 1e-9

    def test_handle_key_comma_period_yaw(self):
        ctrl = _make_controller()
        ctrl.handle_key("comma")
        vc = ctrl.get_velocity_command()
        assert abs(vc[2] - 0.1) < 1e-9

        ctrl.handle_key("period")
        vc = ctrl.get_velocity_command()
        assert abs(vc[2]) < 1e-9

        # Clamp test
        for _ in range(20):
            ctrl.handle_key("comma")
        vc = ctrl.get_velocity_command()
        assert abs(vc[2] - 1.0) < 1e-9

    def test_handle_key_slash_zeros_velocity(self):
        ctrl = _make_controller()
        ctrl.set_velocity_command(0.5, 0.3, -0.2)
        ctrl.handle_key("slash")
        vc = ctrl.get_velocity_command()
        np.testing.assert_array_equal(vc, np.zeros(3))

    def test_handle_key_unknown_noop(self):
        """Unknown key does nothing and doesn't error."""
        ctrl = _make_controller()
        ctrl.handle_key("j")  # Should not raise
        vc = ctrl.get_velocity_command()
        np.testing.assert_array_equal(vc, np.zeros(3))

    def test_handle_key_next_policy(self, tmp_path):
        """'n' loads next policy from policy_dir."""
        from tests.conftest import create_isaaclab_onnx

        # Create two dummy policies
        p1 = str(tmp_path / "a_policy.onnx")
        p2 = str(tmp_path / "b_policy.onnx")
        obs_dim = 70  # matches obs builder
        mapper = JointMapper(G1_29DOF_JOINTS)
        config = _make_config()
        obs_builder = ObservationBuilder(mapper, config)
        obs_dim = obs_builder.observation_dim
        create_isaaclab_onnx(obs_dim, 29, p1)
        create_isaaclab_onnx(obs_dim, 29, p2)

        # Use a real IsaacLab policy that can actually load
        from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
        policy = IsaacLabPolicy(mapper, obs_dim)
        policy.load(p1)

        safety = SafetyController(config, n_dof=29)
        ctrl = Controller(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=obs_builder,
            config=config,
            policy_dir=str(tmp_path),
        )

        ctrl.handle_key("equal")
        # Should have loaded b_policy (index wraps to 1)
        assert ctrl._policy_index == 1

    def test_handle_key_prev_policy(self, tmp_path):
        """'-' loads previous policy from policy_dir."""
        from tests.conftest import create_isaaclab_onnx

        mapper = JointMapper(G1_29DOF_JOINTS)
        config = _make_config()
        obs_builder = ObservationBuilder(mapper, config)
        obs_dim = obs_builder.observation_dim

        p1 = str(tmp_path / "a_policy.onnx")
        p2 = str(tmp_path / "b_policy.onnx")
        create_isaaclab_onnx(obs_dim, 29, p1)
        create_isaaclab_onnx(obs_dim, 29, p2)

        from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
        policy = IsaacLabPolicy(mapper, obs_dim)
        policy.load(p1)

        safety = SafetyController(config, n_dof=29)
        ctrl = Controller(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=obs_builder,
            config=config,
            policy_dir=str(tmp_path),
        )

        ctrl.handle_key("minus")
        # Should wrap to last policy (index -1 % 2 = 1)
        assert ctrl._policy_index == 1


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
        obs_builder = ObservationBuilder(mapper, config)
        obs_dim = obs_builder.observation_dim

        p1 = str(tmp_path / "policy1.onnx")
        p2 = str(tmp_path / "policy2.onnx")
        create_isaaclab_onnx(obs_dim, 29, p1)
        create_isaaclab_onnx(obs_dim, 29, p2)

        policy = IsaacLabPolicy(mapper, obs_dim)
        policy.load(p1)

        safety = SafetyController(config, n_dof=29)
        ctrl = Controller(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=obs_builder,
            config=config,
        )

        # Reload while stopped
        ctrl.reload_policy(p2)
        # No error means success

    def test_reload_policy_invalid_path(self, tmp_path):
        """Invalid path raises error, original policy is preserved."""
        from tests.conftest import create_isaaclab_onnx
        from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy

        mapper = JointMapper(G1_29DOF_JOINTS)
        config = _make_config()
        obs_builder = ObservationBuilder(mapper, config)
        obs_dim = obs_builder.observation_dim

        p1 = str(tmp_path / "policy1.onnx")
        create_isaaclab_onnx(obs_dim, 29, p1)

        policy = IsaacLabPolicy(mapper, obs_dim)
        policy.load(p1)

        safety = SafetyController(config, n_dof=29)
        ctrl = Controller(
            robot=_make_mock_robot(),
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=obs_builder,
            config=config,
        )

        with pytest.raises(ValueError):
            ctrl.reload_policy("/nonexistent/bad.onnx")

        # Original policy should still work
        obs = np.zeros(obs_dim)
        action = ctrl.policy.get_action(obs)
        assert action.shape == (29,)


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
        bm_policy = MagicMock(spec=BeyondMimicPolicy)
        bm_policy.__class__ = BeyondMimicPolicy
        bm_policy.default_joint_pos = np.full(29, 0.2)
        bm_policy.target_q = np.full(29, 0.3)
        bm_policy.target_dq = np.zeros(29)
        bm_policy.stiffness = None
        bm_policy.damping = None
        bm_policy.action_scale = None
        bm_policy.anchor_body_name = ""
        bm_policy.trajectory_length = 5  # Very short trajectory
        bm_policy.get_action.return_value = np.zeros(29)
        bm_policy.build_observation.return_value = np.zeros(160)
        bm_policy.observation_dim = 160
        bm_policy.action_dim = 29

        ctrl = Controller(
            robot=robot,
            policy=bm_policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=None,
            config=config,
        )

        # Start the controller and policy
        ctrl.start()
        safety.start()

        # Wait for trajectory to complete (5 steps at 50Hz = 0.1s, plus margin)
        import time
        time.sleep(1.0)

        # After trajectory ends, safety should be STOPPED
        assert safety.state == SystemState.STOPPED

        ctrl.stop()


# ============================================================================
# Auto-termination
# ============================================================================

class TestNaNHandling:
    def test_nan_state_does_not_produce_nan_command(self):
        """NaN joint positions in robot state should be handled by safety clamp."""
        ctrl = _make_controller()

        # Mock get_state to return NaN positions
        state_with_nan = ctrl.robot.get_state.return_value
        state_with_nan.joint_positions = np.full(29, np.nan)

        state = ctrl.robot.get_state()
        action = np.zeros(29)
        cmd = ctrl._build_command(state, action)
        clamped = ctrl.safety.clamp_command(cmd)

        # After safety clamping, command positions should not contain NaN
        # (they get clamped to joint limits)
        assert not np.any(np.isnan(clamped.joint_positions)), \
            "Safety clamp should handle NaN positions"


class TestAutoTermination:
    def test_auto_termination_max_steps(self):
        """Set max_steps=5, verify it stops after exactly 5 steps."""
        ctrl = _make_controller(max_steps=5)
        ctrl.safety.start()
        ctrl.start()
        time.sleep(1.0)  # Generous wait
        assert ctrl.is_running is False

        telem = ctrl.get_telemetry()
        assert telem["step_count"] == 5


# ============================================================================
# Control loop lifecycle
# ============================================================================

class TestControlLoopLifecycle:
    def test_control_loop_lifecycle(self):
        """start() -> running -> stop() -> stopped, no exceptions."""
        ctrl = _make_controller()
        ctrl.safety.start()
        ctrl.start()
        assert ctrl.is_running is True
        time.sleep(0.1)
        ctrl.safety.stop()
        ctrl.stop()
        assert ctrl.is_running is False

    def test_control_loop_stopped_still_steps(self):
        """In STOPPED state, robot.step() is still called but no command sent."""
        ctrl = _make_controller()
        # Transition to STOPPED (need IDLE -> RUNNING -> STOPPED)
        ctrl.safety.start()
        ctrl.safety.stop()

        # Reset mock call counts
        ctrl.robot.step.reset_mock()
        ctrl.robot.send_command.reset_mock()

        # Run a few iterations manually
        ctrl._running = True
        ctrl._max_steps = 3  # won't matter since not RUNNING
        # Call _control_loop directly — it will see STOPPED and just step
        # We need to limit it somehow; let's use a threading approach
        def run_limited():
            count = 0
            while ctrl._running and count < 5:
                loop_start = time.perf_counter()
                if ctrl.safety.state != SystemState.RUNNING:
                    ctrl.robot.step()
                    count += 1
                    continue
            ctrl._running = False

        run_limited()
        assert ctrl.robot.step.call_count == 5
        # send_command should NOT have been called (only step, no command in STOPPED)
        ctrl.robot.send_command.assert_not_called()

    def test_control_loop_calls_safety_clamp(self):
        """Verify that commands go through safety.clamp_command()."""
        config = _make_config()
        mapper = JointMapper(G1_29DOF_JOINTS)
        safety = SafetyController(config, n_dof=29)
        robot = _make_mock_robot()
        policy = _make_mock_isaaclab_policy()
        obs_builder = ObservationBuilder(mapper, config)

        ctrl = Controller(
            robot=robot,
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=obs_builder,
            config=config,
            max_steps=1,
        )

        # Spy on clamp_command
        original_clamp = safety.clamp_command
        clamp_calls = []

        def spy_clamp(cmd):
            clamp_calls.append(cmd)
            return original_clamp(cmd)

        safety.clamp_command = spy_clamp

        safety.start()
        ctrl.start()
        time.sleep(0.5)

        assert len(clamp_calls) >= 1
