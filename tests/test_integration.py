"""Automated integration tests — Phase 13.

Full-pipeline headless tests: Config → SimRobot → Policy → Controller → Logger.
These use real MuJoCo simulation with test ONNX models (zero-output).
No manual intervention required.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import create_isaaclab_onnx, create_beyondmimic_onnx

from unitree_launcher.config import (
    Config,
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    Q_HOME_23DOF,
    Q_HOME_29DOF,
    load_config,
    resolve_joint_name,
)
from unitree_launcher.control.controller import Controller
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.estimation.state_estimator import StateEstimator
from unitree_launcher.datalog.logger import DataLogger
from unitree_launcher.datalog.replay import LogReplay
from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy
from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.policy.observations import ObservationBuilder
from unitree_launcher.robot.sim_robot import SimRobot

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


# ============================================================================
# Helpers
# ============================================================================

def _build_isaaclab_pipeline(
    tmp_path: Path,
    variant: str = "g1_29dof",
    config_path: str = DEFAULT_CONFIG,
    max_steps: int = 0,
    max_duration: float = 0.0,
    enable_logger: bool = False,
) -> dict:
    """Wire up a full IsaacLab pipeline for integration testing.

    Returns dict with all components for test use.
    """
    config = load_config(config_path)
    config.robot.variant = variant

    # Create ONNX model with correct dims
    robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS
    n_dof = len(robot_joints)
    mapper = JointMapper(robot_joints)
    obs_builder = ObservationBuilder(mapper, config, use_estimator=True)
    obs_dim = obs_builder.observation_dim
    action_dim = n_dof

    onnx_path = str(tmp_path / "test_isaaclab.onnx")
    create_isaaclab_onnx(obs_dim, action_dim, onnx_path)

    robot = SimRobot(config)
    # Initialize to home position (matches real deployment; avoids state-limit
    # faults from starting at MuJoCo default zeros)
    q_home_dict = Q_HOME_29DOF if "29" in variant else Q_HOME_23DOF
    q_home = np.array([q_home_dict[j] for j in robot_joints])
    robot.set_home_positions(q_home)

    # Zero-output test policies can't stabilize the robot, causing joints
    # to exceed limits.  Disable state-fault monitoring for integration tests
    # (real monitoring tested in test_safety*.py and test_safety_sim.py).
    config.safety.fault_threshold = float("inf")

    policy = IsaacLabPolicy(mapper, obs_dim)
    policy.load(onnx_path)
    safety = SafetyController(config, n_dof=robot.n_dof)

    logger = None
    if enable_logger:
        log_dir = str(tmp_path / "logs")
        logger = DataLogger(config.logging, "integration_test", log_dir)

    controller = Controller(
        robot=robot,
        policy=policy,
        safety=safety,
        joint_mapper=mapper,
        obs_builder=obs_builder,
        config=config,
        logger=logger,
        max_steps=max_steps,
        max_duration=max_duration,
    )

    return {
        "config": config,
        "robot": robot,
        "policy": policy,
        "safety": safety,
        "mapper": mapper,
        "obs_builder": obs_builder,
        "controller": controller,
        "logger": logger,
        "onnx_path": onnx_path,
    }


# ============================================================================
# Integration Tests
# ============================================================================

class TestHeadlessIsaacLab:
    def test_headless_sim_isaaclab_100_steps(self, tmp_path):
        """Full pipeline: config → SimRobot → IsaacLabPolicy → Controller.
        Run 100 steps in headless mode, verify no crash."""
        p = _build_isaaclab_pipeline(tmp_path, max_steps=100)
        ctrl = p["controller"]
        safety = p["safety"]

        ctrl.start()
        safety.start()

        # Wait for control loop to finish (100 steps at 50Hz = ~2s, plus margin)
        deadline = time.time() + 10.0
        while ctrl.is_running and time.time() < deadline:
            time.sleep(0.05)

        ctrl.stop()

        # Verify we ran to completion
        assert not ctrl.is_running
        telemetry = ctrl.get_telemetry()
        assert telemetry["step_count"] >= 100

    def test_headless_sim_isaaclab_with_logger(self, tmp_path):
        """Full pipeline with logging. Verify logs contain data."""
        p = _build_isaaclab_pipeline(
            tmp_path, max_steps=50, enable_logger=True,
        )
        ctrl = p["controller"]
        safety = p["safety"]
        logger = p["logger"]

        logger.start()
        ctrl.start()
        safety.start()

        deadline = time.time() + 10.0
        while ctrl.is_running and time.time() < deadline:
            time.sleep(0.05)

        ctrl.stop()
        logger.stop()

        # Verify log directory was created and has data
        log_dir = Path(logger._log_dir)
        assert log_dir.exists()
        assert (log_dir / "metadata.yaml").exists()
        # Data file should exist (HDF5 by default)
        assert (log_dir / "data.hdf5").exists()

        # Verify logged steps
        import h5py
        with h5py.File(log_dir / "data.hdf5", "r") as f:
            assert f["joint_pos"].shape[0] >= 40  # at least most of the 50 steps


class TestEstopRecovery:
    def test_headless_sim_estop_recovery(self, tmp_path):
        """Start → E-stop → clear → resume → stop.
        Verify state transitions occur correctly."""
        p = _build_isaaclab_pipeline(tmp_path, max_steps=200)
        ctrl = p["controller"]
        safety = p["safety"]

        ctrl.start()
        safety.start()

        # Let it run a few steps
        time.sleep(0.15)
        assert safety.state == SystemState.RUNNING

        # E-stop
        ctrl.handle_key("backspace")
        time.sleep(0.1)
        assert safety.state == SystemState.ESTOP

        # Clear E-stop → STOPPED
        ctrl.handle_key("enter")
        time.sleep(0.05)
        assert safety.state == SystemState.STOPPED

        # Resume → RUNNING
        ctrl.handle_key("space")
        time.sleep(0.15)
        assert safety.state == SystemState.RUNNING

        # Stop
        ctrl.handle_key("space")
        time.sleep(0.05)
        assert safety.state == SystemState.STOPPED

        ctrl.stop()


class TestPolicyReload:
    def test_headless_sim_policy_reload(self, tmp_path):
        """Start with one policy, reload to another mid-run."""
        p = _build_isaaclab_pipeline(tmp_path, max_steps=200)
        ctrl = p["controller"]
        safety = p["safety"]

        # Create a second ONNX with same dims
        onnx_path_2 = str(tmp_path / "test_isaaclab_2.onnx")
        obs_dim = p["obs_builder"].observation_dim
        create_isaaclab_onnx(obs_dim, 29, onnx_path_2)

        ctrl.start()
        safety.start()
        time.sleep(0.15)

        # Stop before reload
        ctrl.handle_key("space")
        time.sleep(0.05)
        assert safety.state == SystemState.STOPPED

        # Reload policy
        ctrl.reload_policy(onnx_path_2)

        # Restart with new policy
        ctrl.handle_key("space")
        time.sleep(0.15)
        assert safety.state == SystemState.RUNNING

        ctrl.stop()


class TestSmoke23DOF:
    def test_headless_sim_23dof_smoke(self, tmp_path):
        """Run 10 steps with 23-DOF model. Verify no crash."""
        config_path = os.path.join(PROJECT_ROOT, "configs", "g1_23dof.yaml")
        p = _build_isaaclab_pipeline(
            tmp_path, variant="g1_23dof", config_path=config_path,
            max_steps=10,
        )
        ctrl = p["controller"]
        safety = p["safety"]

        assert p["robot"].n_dof == 23

        ctrl.start()
        safety.start()

        deadline = time.time() + 10.0
        while ctrl.is_running and time.time() < deadline:
            time.sleep(0.05)

        ctrl.stop()

        telemetry = ctrl.get_telemetry()
        assert telemetry["step_count"] >= 10


class TestPerformance:
    @pytest.mark.slow
    def test_headless_performance_50hz(self, tmp_path):
        """Run 200 steps, verify mean loop time is under 20ms (50 Hz target).

        Skipped in CI unless explicitly requested: pytest -m slow
        """
        p = _build_isaaclab_pipeline(tmp_path, max_steps=200)
        ctrl = p["controller"]
        safety = p["safety"]

        ctrl.start()
        safety.start()

        deadline = time.time() + 15.0
        while ctrl.is_running and time.time() < deadline:
            time.sleep(0.05)

        ctrl.stop()

        telemetry = ctrl.get_telemetry()
        assert telemetry["step_count"] >= 200

        # Mean loop time should be well under 20ms on any modern machine
        # (MuJoCo sim + ONNX inference with zero-output model is very fast)
        policy_hz = telemetry["policy_hz"]
        assert policy_hz > 30.0, (
            f"Policy frequency {policy_hz:.1f} Hz is too low (target: 50 Hz)"
        )


# ============================================================================
# BeyondMimic Integration
# ============================================================================

class TestBeyondMimicIntegration:
    def test_beyondmimic_pipeline(self, tmp_path):
        """Full pipeline: Config → SimRobot → BeyondMimicPolicy → Controller.

        Zero-output model has trajectory_length=1, so the BM trajectory ends
        immediately and returns to hold. Verify the pipeline wires correctly,
        runs without crash, and transitions to STOPPED after trajectory end.
        """
        config = load_config(DEFAULT_CONFIG)
        config.robot.variant = "g1_29dof"
        config.safety.fault_threshold = float("inf")

        robot_joints = G1_29DOF_JOINTS
        isaaclab_config_names = [resolve_joint_name(n, "g1_29dof") for n in ISAACLAB_G1_29DOF_JOINTS]
        mapper = JointMapper(
            robot_joints,
            observed_joints=isaaclab_config_names,
            controlled_joints=isaaclab_config_names,
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

        policy = BeyondMimicPolicy(mapper, bm_obs_dim, use_onnx_metadata=True)
        policy.load(onnx_path)
        safety = SafetyController(config, n_dof=29)

        ctrl = Controller(
            robot=robot,
            policy=policy,
            safety=safety,
            joint_mapper=mapper,
            obs_builder=None,
            config=config,
        )

        ctrl.start()
        safety.start()

        # Wait for trajectory to complete and return to hold
        deadline = time.time() + 5.0
        while safety.state == SystemState.RUNNING and time.time() < deadline:
            time.sleep(0.05)

        # BM trajectory with zero-output model ends immediately → STOPPED
        assert safety.state == SystemState.STOPPED
        assert ctrl.get_telemetry()["step_count"] >= 1

        ctrl.stop()


# ============================================================================
# Estimator Integration
# ============================================================================

class TestEstimatorIntegration:
    def test_estimator_in_loop_100_steps(self, tmp_path):
        """Build pipeline with estimator, run 100 steps, verify estimator called."""
        p = _build_isaaclab_pipeline(tmp_path, max_steps=100)
        ctrl = p["controller"]
        safety = p["safety"]

        estimator = StateEstimator(p["config"])

        # Spy on update calls
        update_count = [0]
        original_update = estimator.update

        def spy_update(*args, **kwargs):
            update_count[0] += 1
            return original_update(*args, **kwargs)

        estimator.update = spy_update

        ctrl.set_estimator(estimator)
        ctrl.start()
        safety.start()

        deadline = time.time() + 10.0
        while ctrl.is_running and time.time() < deadline:
            time.sleep(0.05)

        ctrl.stop()

        telemetry = ctrl.get_telemetry()
        assert telemetry["step_count"] >= 100
        assert update_count[0] >= 100


# ============================================================================
# Graceful Shutdown
# ============================================================================

class TestGracefulShutdown:
    def test_shutdown_cleans_up(self, tmp_path):
        """Build pipeline with logging, run 50 steps, verify clean shutdown."""
        p = _build_isaaclab_pipeline(
            tmp_path, max_steps=50, enable_logger=True,
        )
        ctrl = p["controller"]
        safety = p["safety"]
        logger = p["logger"]

        logger.start()
        ctrl.start()
        safety.start()

        deadline = time.time() + 10.0
        while ctrl.is_running and time.time() < deadline:
            time.sleep(0.05)

        ctrl.stop()
        logger.stop()

        # Verify log files exist
        log_dir = Path(logger._log_dir)
        assert (log_dir / "metadata.yaml").exists()
        assert (log_dir / "data.hdf5").exists()

        # Controller should be stopped
        assert ctrl.is_running is False

        # No dangling threads from the controller
        assert ctrl._thread is None


# ============================================================================
# Log Replay Roundtrip
# ============================================================================

class TestLogReplayRoundtrip:
    def test_log_replay_roundtrip(self, tmp_path):
        """Run 50 steps with DataLogger, stop, replay with LogReplay."""
        p = _build_isaaclab_pipeline(
            tmp_path, max_steps=50, enable_logger=True,
        )
        ctrl = p["controller"]
        safety = p["safety"]
        logger = p["logger"]

        logger.start()
        ctrl.start()
        safety.start()

        deadline = time.time() + 10.0
        while ctrl.is_running and time.time() < deadline:
            time.sleep(0.05)

        ctrl.stop()
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
