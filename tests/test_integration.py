"""Integration tests.

Full-runtime headless tests: Config → SimRobot → Policy → Runtime → Logger.
These use real MuJoCo simulation with test ONNX models (zero-output).
No manual intervention required.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import create_isaaclab_onnx, create_beyondmimic_onnx

from unitree_launcher.config import (
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    Q_HOME_23DOF,
    Q_HOME_29DOF,
    load_config,
    resolve_joint_name,
)
from unitree_launcher.control.runtime import Runtime
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.estimation.state_estimator import StateEstimator
from unitree_launcher.datalog.logger import DataLogger
from unitree_launcher.datalog.replay import LogReplay
from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy
from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.sim_robot import SimRobot

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "sim.yaml")


# ============================================================================
# Helpers
# ============================================================================

def _build_isaaclab_runtime(
    tmp_path: Path,
    variant: str = "g1_29dof",
    config_path: str = DEFAULT_CONFIG,
    enable_logger: bool = False,
) -> dict:
    """Wire up a full IsaacLab runtime for integration testing.

    Returns dict with all components for test use.
    """
    config = load_config(config_path)
    config.robot.variant = variant

    # Create ONNX model with correct dims
    robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS
    n_dof = len(robot_joints)
    mapper = JointMapper(robot_joints)

    robot = SimRobot(config)
    q_home_dict = Q_HOME_29DOF if "29" in variant else Q_HOME_23DOF
    q_home = np.array([q_home_dict[j] for j in robot_joints])
    robot.set_home_positions(q_home)

    # Zero-output test policies can't stabilize the robot.
    config.safety.fault_threshold = float("inf")
    config.safety.tilt_check = False
    config.control.transition_steps = 0  # Instant activation for tests

    policy = IsaacLabPolicy(mapper, config)
    obs_dim = policy.observation_dim
    onnx_path = str(tmp_path / "test_isaaclab.onnx")
    create_isaaclab_onnx(obs_dim, n_dof, onnx_path)
    policy.load(onnx_path)
    safety = SafetyController(config, n_dof=robot.n_dof)

    logger = None
    if enable_logger:
        log_dir = str(tmp_path / "logs")
        logger = DataLogger(config.logging, "integration_test", log_dir)

    from unitree_launcher.policy.hold_policy import HoldPolicy
    from unitree_launcher.controller.input import InputManager
    from unitree_launcher.controller.keyboard import KeyboardInput
    default_policy = HoldPolicy(mapper, config)
    kb = KeyboardInput()
    input_mgr = InputManager([kb])

    rt = Runtime(
        robot=robot,
        policy=policy,
        safety=safety,
        joint_mapper=mapper,
        config=config,
        logger=logger,
        default_policy=default_policy,
        default_joint_mapper=mapper,
        input_manager=input_mgr,
    )
    rt._keyboard = kb

    from types import SimpleNamespace
    return SimpleNamespace(
        config=config,
        robot=robot,
        policy=policy,
        safety=safety,
        mapper=mapper,
        runtime=rt,
        logger=logger,
        onnx_path=onnx_path,
    )


# ============================================================================
# Integration Tests
# ============================================================================

class TestHeadlessIsaacLab:
    def test_headless_sim_isaaclab_100_steps(self, tmp_path):
        """Full runtime: config → SimRobot → IsaacLabPolicy → Runtime.
        Run 100 steps via step(), verify no crash."""
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime
        safety = s.safety

        rt.start()
        safety.start()
        for _ in range(100):
            rt.step()
        rt.stop()

        telemetry = rt.get_telemetry()
        assert telemetry["step_count"] >= 100

    def test_headless_sim_isaaclab_with_logger(self, tmp_path):
        """Full runtime with logging. Verify logs contain data."""
        s = _build_isaaclab_runtime(
            tmp_path, enable_logger=True,
        )
        rt = s.runtime
        safety = s.safety
        logger = s.logger

        logger.start()
        rt.start()
        safety.start()
        for _ in range(50):
            rt.step()
        rt.stop()
        logger.stop()

        log_dir = Path(logger._log_dir)
        assert log_dir.exists()
        assert (log_dir / "metadata.yaml").exists()
        assert (log_dir / "data.hdf5").exists()

        import h5py
        with h5py.File(log_dir / "data.hdf5", "r") as f:
            assert f["joint_pos"].shape[0] >= 40


class TestEstopRecovery:
    def test_headless_sim_estop_recovery(self, tmp_path):
        """Start → E-stop → clear → resume → stop via step()."""
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime
        safety = s.safety

        rt.start()
        safety.start()
        for _ in range(5):
            rt.step()
        assert safety.state == SystemState.RUNNING

        rt._keyboard.push_key("backspace")
        rt.step()
        assert safety.state == SystemState.ESTOP

        rt._keyboard.push_key("enter")
        rt.step()
        assert safety.state == SystemState.STOPPED

        rt._keyboard.push_key("space")
        rt.step()
        for _ in range(5):
            rt.step()
        assert safety.state == SystemState.RUNNING

        rt._keyboard.push_key("space")
        rt.step()
        assert safety.state == SystemState.STOPPED

        rt.stop()


class TestPolicyReload:
    def test_headless_sim_policy_reload(self, tmp_path):
        """Start with one policy, reload to another mid-run."""
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime
        safety = s.safety

        onnx_path_2 = str(tmp_path / "test_isaaclab_2.onnx")
        obs_dim = s.policy.observation_dim
        create_isaaclab_onnx(obs_dim, 29, onnx_path_2)

        rt.start()
        safety.start()
        for _ in range(5):
            rt.step()

        rt._keyboard.push_key("space")
        rt.step()
        assert safety.state == SystemState.STOPPED

        rt.reload_policy(onnx_path_2)

        rt._keyboard.push_key("space")
        for _ in range(5):
            rt.step()
        assert safety.state == SystemState.RUNNING

        rt.stop()


class TestSmoke23DOF:
    def test_headless_sim_23dof_smoke(self, tmp_path):
        """Run 10 steps with 23-DOF model via step()."""
        s = _build_isaaclab_runtime(
            tmp_path, variant="g1_23dof", config_path=DEFAULT_CONFIG,
        )
        rt = s.runtime
        safety = s.safety

        assert s.robot.n_dof == 23

        rt.start()
        safety.start()
        for _ in range(10):
            rt.step()
        rt.stop()

        telemetry = rt.get_telemetry()
        assert telemetry["step_count"] >= 10


class TestPerformance:
    @pytest.mark.slow
    def test_headless_performance_step(self, tmp_path):
        """Run 200 steps via step(), verify fast execution."""
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime
        safety = s.safety

        rt.start()
        safety.start()

        start = time.time()
        for _ in range(200):
            rt.step()
        elapsed = time.time() - start

        rt.stop()

        telemetry = rt.get_telemetry()
        assert telemetry["step_count"] >= 200
        # With no sleep, 200 steps should complete in well under 10s
        assert elapsed < 10.0, f"200 steps took {elapsed:.1f}s (too slow)"


# ============================================================================
# BeyondMimic Integration
# ============================================================================

class TestBeyondMimicIntegration:
    def test_beyondmimic_runtime(self, tmp_path):
        """Full runtime: Config → SimRobot → BeyondMimicPolicy → Runtime.

        Zero-output model has trajectory_length=1, so the BM trajectory ends
        immediately and returns to hold. Verify the runtime wires correctly,
        runs without crash, and transitions to STOPPED after trajectory end.
        """
        config = load_config(DEFAULT_CONFIG)
        config.robot.variant = "g1_29dof"
        config.safety.fault_threshold = float("inf")
        config.control.transition_steps = 0

        robot_joints = G1_29DOF_JOINTS
        isaaclab_config_names = [resolve_joint_name(n, "g1_29dof") for n in ISAACLAB_G1_29DOF_JOINTS]
        mapper = JointMapper(
            robot_joints,
            policy_joints=isaaclab_config_names,
        )

        bm_obs_dim = 160
        bm_metadata = {
            "joint_names": ",".join(ISAACLAB_G1_29DOF_JOINTS),
            "joint_stiffness": ",".join(["40.0"] * 29),
            "joint_damping": ",".join(["2.5"] * 29),
            "action_scale": ",".join(["0.5"] * 29),
            "default_joint_pos": ",".join(["0.0"] * 29),
            "anchor_body_name": "torso_link",
            "body_names": "pelvis,torso_link,left_knee_link",
            "observation_names": "command,motion_anchor_pos_b,motion_anchor_ori_b,"
                                 "base_lin_vel,base_ang_vel,joint_pos,joint_vel,actions",
            "observation_history_lengths": ",".join(["1.0"] * 8),
        }

        onnx_path = str(tmp_path / "bm_test.onnx")
        create_beyondmimic_onnx(bm_obs_dim, 29, 29, onnx_path, metadata=bm_metadata)

        robot = SimRobot(config)
        q_home = np.array([Q_HOME_29DOF[j] for j in robot_joints])
        robot.set_home_positions(q_home)

        policy = BeyondMimicPolicy(mapper, bm_obs_dim, use_onnx_metadata=True, config=config)
        policy.load(onnx_path)
        policy.set_robot(robot)
        safety = SafetyController(config, n_dof=29)

        from unitree_launcher.policy.hold_policy import HoldPolicy
        from unitree_launcher.controller.input import InputManager
        rt = Runtime(
            robot=robot,
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            config=config,
            default_policy=HoldPolicy(mapper, config),
            default_joint_mapper=mapper,
            input_manager=InputManager(),
        )

        rt.start()
        safety.start()

        # Step until trajectory completes (hold_steps + trajectory_length)
        for _ in range(30):
            rt.step()
            if safety.state == SystemState.STOPPED:
                break

        assert safety.state == SystemState.STOPPED
        assert rt.get_telemetry()["step_count"] >= 1

        rt.stop()


# ============================================================================
# Estimator Integration
# ============================================================================

class TestEstimatorIntegration:
    def test_estimator_in_loop_100_steps(self, tmp_path):
        """Build runtime with estimator, run 100 steps, verify estimator called."""
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime
        safety = s.safety

        estimator = StateEstimator(s.config)

        # Spy on update calls
        update_count = [0]
        original_update = estimator.update

        def spy_update(*args, **kwargs):
            update_count[0] += 1
            return original_update(*args, **kwargs)

        estimator.update = spy_update

        rt.set_estimator(estimator)
        rt.start()
        safety.start()
        for _ in range(100):
            rt.step()
        rt.stop()

        telemetry = rt.get_telemetry()
        assert telemetry["step_count"] >= 100
        assert update_count[0] >= 100


# ============================================================================
# Log Replay Roundtrip
# ============================================================================

class TestLogReplayRoundtrip:
    def test_log_replay_roundtrip(self, tmp_path):
        """Run 50 steps with DataLogger, stop, replay with LogReplay."""
        s = _build_isaaclab_runtime(
            tmp_path, enable_logger=True,
        )
        rt = s.runtime
        safety = s.safety
        logger = s.logger

        logger.start()
        rt.start()
        safety.start()
        for _ in range(50):
            rt.step()
        rt.stop()
        logger.stop()

        # Replay the log
        log_dir = str(Path(logger._log_dir))
        replay = LogReplay(log_dir)
        replay.load()

        assert replay.n_steps >= 40

        # Values should be finite
        for i in range(min(replay.n_steps, 10)):
            state = replay.get_state_at(i)
            assert np.all(np.isfinite(state.joint_positions))
            assert np.all(np.isfinite(state.joint_velocities))

        # Timestamps monotonically increasing
        import h5py
        with h5py.File(Path(log_dir) / "data.hdf5", "r") as f:
            timestamps = f["timestamps"][:]
            diffs = np.diff(timestamps)
            assert np.all(diffs >= 0), "Timestamps should be monotonically increasing"


# ============================================================================
# Smooth Policy Transitions
# ============================================================================

class TestSmoothTransitions:
    def test_stance_to_policy_interpolates(self, tmp_path):
        """Activation interpolates to policy starting pose — no position jump."""
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime
        safety = s.safety
        robot = s.robot

        rt.start()

        # Run a few steps in default (stance) to establish baseline
        for _ in range(5):
            rt.step()
        state_before = robot.get_state()
        pos_before = state_before.joint_positions.copy()

        # Spy on send_command to capture the first post-activation command
        sent_cmds = []
        original_send = robot.send_command
        def capture_send(cmd):
            sent_cmds.append(cmd)
            return original_send(cmd)
        robot.send_command = capture_send

        # Activate policy — should enter TRANSITION, not run policy
        safety.start()
        rt.step()

        # First transition command should be close to current position
        assert len(sent_cmds) >= 1
        cmd = sent_cmds[0]
        max_jump = np.max(np.abs(cmd.joint_positions - pos_before))
        assert max_jump < 0.5, (
            f"Position jump {max_jump:.3f} rad on activation is too large"
        )
        robot.send_command = original_send
        rt.stop()

    def test_bm_end_returns_instantly(self, tmp_path):
        """BM trajectory end returns to default immediately (no transition)."""
        config = load_config(DEFAULT_CONFIG)
        config.robot.variant = "g1_29dof"
        config.safety.fault_threshold = float("inf")
        config.safety.tilt_check = False
        config.control.transition_steps = 0

        robot_joints = G1_29DOF_JOINTS
        isaaclab_config_names = [
            resolve_joint_name(n, "g1_29dof") for n in ISAACLAB_G1_29DOF_JOINTS
        ]
        mapper = JointMapper(
            robot_joints,
            policy_joints=isaaclab_config_names,
        )

        bm_metadata = {
            "joint_names": ",".join(ISAACLAB_G1_29DOF_JOINTS),
            "joint_stiffness": ",".join(["40.0"] * 29),
            "joint_damping": ",".join(["2.5"] * 29),
            "action_scale": ",".join(["0.5"] * 29),
            "default_joint_pos": ",".join(["0.0"] * 29),
            "anchor_body_name": "torso_link",
            "body_names": "pelvis,torso_link,left_knee_link",
            "observation_names": "command,motion_anchor_pos_b,motion_anchor_ori_b,"
                                 "base_lin_vel,base_ang_vel,joint_pos,joint_vel,actions",
            "observation_history_lengths": ",".join(["1.0"] * 8),
        }

        onnx_path = str(tmp_path / "bm_test.onnx")
        create_beyondmimic_onnx(160, 29, 29, onnx_path, metadata=bm_metadata)

        robot = SimRobot(config)
        q_home = np.array([Q_HOME_29DOF[j] for j in robot_joints])
        robot.set_home_positions(q_home)

        policy = BeyondMimicPolicy(mapper, 160, use_onnx_metadata=True, config=config)
        policy.load(onnx_path)
        policy.set_robot(robot)
        safety_ctrl = SafetyController(config, n_dof=29)

        from unitree_launcher.policy.hold_policy import HoldPolicy
        from unitree_launcher.controller.input import InputManager
        default_policy = HoldPolicy(mapper, config)
        rt = Runtime(
            robot=robot,
            policy=policy,
            safety=safety_ctrl,
            joint_mapper=mapper,
            config=config,
            default_policy=default_policy,
            default_joint_mapper=mapper,
            input_manager=InputManager(),
        )

        rt.start()
        safety_ctrl.start()

        # Run until BM ends — should return to STOPPED instantly
        for _ in range(20):
            rt.step()
            if safety_ctrl.state == SystemState.STOPPED:
                break

        assert safety_ctrl.state == SystemState.STOPPED
        # No transition should be active (instant return)
        assert rt._transition_active is False
        rt.stop()


# ============================================================================
# Additional Integration Tests
# ============================================================================

class TestDefaultPolicyHolds:
    def test_default_policy_holds_in_idle(self, tmp_path):
        """Default policy in IDLE/STOPPED keeps robot near home for 100 steps."""
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime

        # Run in IDLE (not RUNNING) — default policy should hold
        rt.start()
        for _ in range(100):
            rt.step()

        state = s.robot.get_state()
        height = state.base_position[2]
        assert height > 0.5, f"Robot fell in IDLE: height={height:.3f}"
        rt.stop()


class TestPreparePhase:
    def test_prepare_reaches_default_pose(self, tmp_path):
        """Prepare mode sequence interpolates robot to default standing pose."""
        from unitree_launcher.control.safety import ControlMode
        s = _build_isaaclab_runtime(tmp_path)
        rt = s.runtime

        # Configure a short PREPARE mode sequence (2s = 100 steps at 50Hz).
        # Check pose at completion (before the zero-output test policy
        # takes over and destabilizes the robot).
        rt._mode_sequence = [(ControlMode.PREPARE, 2.0)]
        rt._initial_mode_sequence = [(ControlMode.PREPARE, 2.0)]

        rt.start()
        s.safety.start()
        # Run exactly through prepare (100 steps)
        for _ in range(100):
            rt.step()

        state = s.robot.get_state()
        q_home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        max_err = np.max(np.abs(state.joint_positions - q_home))
        assert max_err < 0.1, f"Prepare didn't reach default pose: max_err={max_err:.4f}"
        rt.stop()


class TestAllModesInstantiate:
    def test_eval_config_loads_and_steps(self, tmp_path):
        """Eval config (1000Hz) loads and steps without crash."""
        config = load_config(DEFAULT_CONFIG)
        config.control.sim_frequency = 1000
        config.safety.fault_threshold = float("inf")
        config.safety.tilt_check = False
        config.control.transition_steps = 0

        robot_joints = G1_29DOF_JOINTS
        mapper = JointMapper(robot_joints)
        robot = SimRobot(config)
        q_home = np.array([Q_HOME_29DOF[j] for j in robot_joints])
        robot.set_home_positions(q_home)

        from unitree_launcher.policy.hold_policy import HoldPolicy
        from unitree_launcher.controller.input import InputManager
        policy = HoldPolicy(mapper, config)
        safety = SafetyController(config, n_dof=robot.n_dof)

        rt = Runtime(
            robot=robot,
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            config=config,
            default_policy=HoldPolicy(mapper, config),
            default_joint_mapper=mapper,
            input_manager=InputManager(),
        )

        rt.start()
        safety.start()
        for _ in range(5):
            rt.step()
        rt.stop()
