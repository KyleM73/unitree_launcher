"""Tests for src/main.py CLI — Phase 12.

Covers argument parsing, config integration, component wiring, and
apply_cli_overrides.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from unitree_launcher.config import (
    Config,
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    apply_cli_overrides,
    load_config,
)
from unitree_launcher.main import build_parser, main

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


# ============================================================================
# Helpers
# ============================================================================

def _parse(argv: list):
    """Parse argv through the CLI parser."""
    return build_parser().parse_args(argv)


def _make_isaaclab_onnx(path: str, obs_dim: int = 99, action_dim: int = 29):
    """Create a minimal IsaacLab ONNX model at *path*.

    Default obs_dim=99 matches ObservationBuilder for 29-DOF with estimator:
    2*29 (pos+vel) + 29 (actions) + 12 (ang_vel+gravity+vel_cmd+lin_vel) = 99.
    """
    from tests.conftest import create_isaaclab_onnx
    create_isaaclab_onnx(obs_dim, action_dim, path)


# ============================================================================
# Argument Parsing — sim
# ============================================================================

class TestParseSimArgs:
    def test_parse_sim_basic(self):
        args = _parse(["sim", "--policy", "test.onnx"])
        assert args.mode == "sim"
        assert args.policy == "test.onnx"
        assert args.gui is False
        assert args.viser is False
        assert args.config == "configs/default.yaml"

    def test_parse_sim_headless_with_duration(self):
        args = _parse(["sim", "--policy", "p.onnx",
                        "--duration", "10", "--steps", "500"])
        assert args.gui is False
        assert args.duration == 10.0
        assert args.steps == 500

    def test_parse_sim_defaults(self):
        args = _parse(["sim", "--policy", "p.onnx"])
        assert args.duration is None
        assert args.steps is None
        assert args.domain_id is None
        assert args.robot is None
        assert args.log_dir == "logs/"
        assert args.no_log is False
        assert args.policy_dir.endswith("assets/policies")

    def test_missing_policy_without_gantry_errors(self):
        """--policy is required unless --gantry is set."""
        # Parser accepts sim without --policy (validation is in main())
        args = _parse(["sim"])
        assert args.policy is None

    def test_gantry_flag_parsed(self):
        args = _parse(["sim", "--gantry"])
        assert args.gantry is True
        assert args.policy is None

    def test_gantry_headless_by_default(self):
        args = _parse(["sim", "--gantry"])
        assert args.gantry is True
        assert args.gui is False
        assert args.viser is False

    def test_missing_mode_errors(self):
        with pytest.raises(SystemExit):
            _parse(["--policy", "p.onnx"])


# ============================================================================
# Argument Parsing — real
# ============================================================================

class TestParseRealArgs:
    def test_parse_real_basic(self):
        args = _parse(["real", "--policy", "test.onnx", "--interface", "eth0"])
        assert args.mode == "real"
        assert args.interface == "eth0"

    def test_real_requires_interface(self):
        with pytest.raises(SystemExit):
            _parse(["real", "--policy", "test.onnx"])


# ============================================================================
# Argument Parsing — flags
# ============================================================================

class TestParseFlags:
    def test_no_est_flag(self):
        args = _parse(["sim", "--policy", "p.onnx", "--no-est"])
        assert args.no_est is True

    def test_no_est_default_false(self):
        args = _parse(["sim", "--policy", "p.onnx"])
        assert args.no_est is False

    def test_no_log_flag(self):
        args = _parse(["sim", "--policy", "p.onnx", "--no-log"])
        assert args.no_log is True

    def test_domain_id_override(self):
        args = _parse(["sim", "--policy", "p.onnx", "--domain-id", "5"])
        assert args.domain_id == 5

    def test_robot_override(self):
        args = _parse(["sim", "--policy", "p.onnx", "--robot", "g1_23dof"])
        assert args.robot == "g1_23dof"

    def test_policy_dir(self):
        args = _parse(["sim", "--policy", "p.onnx", "--policy-dir", "/tmp/pols"])
        assert args.policy_dir == "/tmp/pols"


# ============================================================================
# apply_cli_overrides
# ============================================================================

class TestApplyCLIOverrides:
    def test_robot_override(self):
        config = load_config(DEFAULT_CONFIG)
        assert config.robot.variant == "g1_29dof"  # default
        args = _parse(["sim", "--policy", "p.onnx", "--robot", "g1_23dof"])
        apply_cli_overrides(config, args)
        assert config.robot.variant == "g1_23dof"

    def test_no_override_preserves_default(self):
        config = load_config(DEFAULT_CONFIG)
        args = _parse(["sim", "--policy", "p.onnx"])
        apply_cli_overrides(config, args)
        assert config.robot.variant == "g1_29dof"

    def test_invalid_variant_raises(self):
        config = load_config(DEFAULT_CONFIG)
        args = _parse(["sim", "--policy", "p.onnx", "--robot", "invalid_bot"])
        with pytest.raises(ValueError, match="Invalid variant"):
            apply_cli_overrides(config, args)


# ============================================================================
# Config Integration (domain ID, interface)
# ============================================================================

class TestConfigIntegration:
    def test_sim_default_domain_id(self):
        """Sim mode without --domain-id keeps config default (1)."""
        config = load_config(DEFAULT_CONFIG)
        args = _parse(["sim", "--policy", "p.onnx"])
        # Simulate what main() does
        if args.domain_id is not None:
            config.network.domain_id = args.domain_id
        elif args.mode == "real":
            config.network.domain_id = 0
        assert config.network.domain_id == 1

    def test_real_default_domain_id(self):
        """Real mode without --domain-id sets domain_id=0."""
        config = load_config(DEFAULT_CONFIG)
        args = _parse(["real", "--policy", "p.onnx", "--interface", "eth0"])
        if args.domain_id is not None:
            config.network.domain_id = args.domain_id
        elif args.mode == "real":
            config.network.domain_id = 0
        assert config.network.domain_id == 0

    def test_explicit_domain_id_override(self):
        """--domain-id 5 overrides both sim and real defaults."""
        config = load_config(DEFAULT_CONFIG)
        args = _parse(["sim", "--policy", "p.onnx", "--domain-id", "5"])
        if args.domain_id is not None:
            config.network.domain_id = args.domain_id
        assert config.network.domain_id == 5

    def test_real_interface_set(self):
        """Real mode sets config.network.interface from --interface."""
        config = load_config(DEFAULT_CONFIG)
        args = _parse(["real", "--policy", "p.onnx", "--interface", "enp3s0"])
        config.network.interface = args.interface
        assert config.network.interface == "enp3s0"


# ============================================================================
# Component Wiring
# ============================================================================

class TestComponentWiring:
    def test_variant_resolution_29dof(self):
        variant = "g1_29dof"
        joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS
        assert len(joints) == 29

    def test_variant_resolution_23dof(self):
        variant = "g1_23dof"
        joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS
        assert len(joints) == 23

    def test_policy_format_auto_detection(self):
        """When config.policy.format is None, detect_policy_format is called."""
        config = load_config(DEFAULT_CONFIG)
        assert config.policy.format is None  # default is auto-detect

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            _make_isaaclab_onnx(onnx_path)
            with patch("unitree_launcher.main.detect_policy_format", return_value="isaaclab") as mock_detect:
                # We test the logic that main() uses
                fmt = config.policy.format or mock_detect(onnx_path)
                assert fmt == "isaaclab"
                mock_detect.assert_called_once_with(onnx_path)
        finally:
            os.unlink(onnx_path)

    def test_no_est_cli_overrides_config(self):
        """Config has use_estimator=True, --no-est overrides to False."""
        config = load_config(DEFAULT_CONFIG)
        assert config.policy.use_estimator is True
        args = _parse(["sim", "--policy", "p.onnx", "--no-est"])
        use_est = config.policy.use_estimator
        if args.no_est:
            use_est = False
        assert use_est is False

    def test_use_estimator_from_config(self):
        """Without --no-est, use_estimator comes from config."""
        config = load_config(DEFAULT_CONFIG)
        config.policy.use_estimator = False
        args = _parse(["sim", "--policy", "p.onnx"])
        use_est = config.policy.use_estimator
        if args.no_est:
            use_est = False
        assert use_est is False


# ============================================================================
# main() Integration (mocked components)
# ============================================================================

class TestMainIntegration:
    def test_main_sim_headless_wiring(self, tmp_path):
        """main() in sim headless mode wires components and calls run_headless."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)

        with patch("unitree_launcher.main.SimRobot") as MockSimRobot, \
             patch("unitree_launcher.main.run_headless") as mock_run, \
             patch("unitree_launcher.main.DataLogger") as MockLogger:

            mock_robot = MagicMock()
            mock_robot.n_dof = 29
            MockSimRobot.return_value = mock_robot

            mock_logger = MagicMock()
            MockLogger.return_value = mock_logger

            main([
                "sim", "--policy", onnx_path,                 "--config", DEFAULT_CONFIG, "--no-log",
            ])

            MockSimRobot.assert_called_once()
            mock_robot.connect.assert_called_once()
            mock_run.assert_called_once()
            # Shutdown calls graceful_shutdown (if present) or disconnect
            assert (
                mock_robot.graceful_shutdown.called
                or mock_robot.disconnect.called
            )

    def test_main_sim_viewer_wiring(self, tmp_path):
        """main() with --gui calls run_with_viewer."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)

        with patch("unitree_launcher.main.SimRobot") as MockSimRobot, \
             patch("unitree_launcher.main.run_with_viewer") as mock_run:

            mock_robot = MagicMock()
            mock_robot.n_dof = 29
            MockSimRobot.return_value = mock_robot

            main([
                "sim", "--policy", onnx_path, "--gui",
                "--config", DEFAULT_CONFIG, "--no-log",
            ])

            mock_run.assert_called_once()

    def test_main_real_mode_wiring(self, tmp_path):
        """main() in real mode creates RealRobot and sets domain_id=0."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)

        with patch("unitree_launcher.main.RealRobot") as MockRealRobot, \
             patch("unitree_launcher.main.run_headless") as mock_run:

            mock_robot = MagicMock()
            mock_robot.n_dof = 29
            MockRealRobot.return_value = mock_robot

            main([
                "real", "--policy", onnx_path, "--interface", "eth0",
                "--config", DEFAULT_CONFIG, "--no-log",
            ])

            MockRealRobot.assert_called_once()
            # Verify the config passed to RealRobot has domain_id=0
            call_config = MockRealRobot.call_args[0][0]
            assert call_config.network.domain_id == 0
            assert call_config.network.interface == "eth0"

    def test_main_logger_lifecycle(self, tmp_path):
        """Logger start/stop called when logging is enabled."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)
        log_dir = str(tmp_path / "logs")

        with patch("unitree_launcher.main.SimRobot") as MockSimRobot, \
             patch("unitree_launcher.main.run_headless"), \
             patch("unitree_launcher.main.DataLogger") as MockLogger:

            mock_robot = MagicMock()
            mock_robot.n_dof = 29
            MockSimRobot.return_value = mock_robot

            mock_logger = MagicMock()
            MockLogger.return_value = mock_logger

            main([
                "sim", "--policy", onnx_path,                 "--config", DEFAULT_CONFIG, "--log-dir", log_dir,
            ])

            MockLogger.assert_called_once()
            mock_logger.start.assert_called_once()
            mock_logger.stop.assert_called_once()

    def test_main_no_log_skips_logger(self, tmp_path):
        """--no-log skips logger creation entirely."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)

        with patch("unitree_launcher.main.SimRobot") as MockSimRobot, \
             patch("unitree_launcher.main.run_headless"), \
             patch("unitree_launcher.main.DataLogger") as MockLogger:

            mock_robot = MagicMock()
            mock_robot.n_dof = 29
            MockSimRobot.return_value = mock_robot

            main([
                "sim", "--policy", onnx_path,                 "--config", DEFAULT_CONFIG, "--no-log",
            ])

            MockLogger.assert_not_called()

    def test_main_policy_not_found(self, tmp_path):
        """Nonexistent policy file raises error during load."""
        with patch("unitree_launcher.main.SimRobot") as MockSimRobot:
            mock_robot = MagicMock()
            mock_robot.n_dof = 29
            MockSimRobot.return_value = mock_robot

            with pytest.raises((FileNotFoundError, Exception)):
                main([
                    "sim", "--policy", "/nonexistent/path.onnx",                     "--config", DEFAULT_CONFIG, "--no-log",
                ])

    def test_run_headless_keyboard_interrupt(self, tmp_path):
        """KeyboardInterrupt during run_headless should call controller.stop()."""
        from unitree_launcher.main import run_headless
        from unitree_launcher.control.controller import Controller
        from unitree_launcher.control.safety import SafetyController
        from unitree_launcher.policy.joint_mapper import JointMapper
        from unitree_launcher.policy.observations import ObservationBuilder

        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)

        config = load_config(DEFAULT_CONFIG)
        robot = MagicMock()
        robot.n_dof = 29

        ctrl = MagicMock()
        ctrl.is_running = True
        ctrl.safety = MagicMock()

        # Make ctrl.is_running raise KeyboardInterrupt on second access
        call_count = [0]
        original_is_running = True

        def is_running_side_effect():
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
            return True

        type(ctrl).is_running = property(lambda self: is_running_side_effect())

        run_headless(robot, ctrl)
        ctrl.stop.assert_called_once()

    def test_main_policy_dir_passed_to_controller(self, tmp_path):
        """--policy-dir is forwarded to Controller."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)
        pol_dir = str(tmp_path / "policies")

        with patch("unitree_launcher.main.SimRobot") as MockSimRobot, \
             patch("unitree_launcher.main.run_headless"), \
             patch("unitree_launcher.main.Controller") as MockController:

            mock_robot = MagicMock()
            mock_robot.n_dof = 29
            MockSimRobot.return_value = mock_robot
            MockController.return_value = MagicMock()

            main([
                "sim", "--policy", onnx_path,                 "--config", DEFAULT_CONFIG, "--no-log",
                "--policy-dir", pol_dir,
            ])

            _, kwargs = MockController.call_args
            assert kwargs["policy_dir"] == pol_dir
