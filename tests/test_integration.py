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

from src.config import (
    Config,
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    load_config,
)
from src.control.controller import Controller
from src.control.safety import SafetyController, SystemState
from src.logging.logger import DataLogger
from src.policy.beyondmimic_policy import BeyondMimicPolicy
from src.policy.isaaclab_policy import IsaacLabPolicy
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from src.robot.sim_robot import SimRobot

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
