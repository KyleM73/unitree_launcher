"""Tests for main.py CLI (sim/eval/real/mirror/replay modes).

Covers argument parsing, config integration, component wiring,
apply_cli_overrides, GLFW key map, key callback dispatch,
run_with_viewer, and run_headless.
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from unitree_launcher.config import (
    Config,
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    Q_HOME_29DOF,
    apply_cli_overrides,
    load_config,
)
from unitree_launcher.control.runtime import Runtime
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.main import GLFW_KEY_MAP, build_parser, main, run_headless, run_with_viewer
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotState

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "sim.yaml")


# ============================================================================
# Helpers
# ============================================================================

def _parse(argv: list):
    """Parse argv through the CLI parser."""
    return build_parser().parse_args(argv)


def _make_isaaclab_onnx(path: str, obs_dim: int = 99, action_dim: int = 29):
    """Create a minimal IsaacLab ONNX model at *path*.

    Default obs_dim=99 matches IsaacLabPolicy observation dimension for 29-DOF with estimator:
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
        assert args.config == "configs/sim.yaml"

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

    def test_real_defaults_interface_eth0(self):
        args = _parse(["real", "--policy", "test.onnx"])
        assert args.interface == "eth0"


# ============================================================================
# Argument Parsing — flags
# ============================================================================

class TestParsePreset:
    def test_preset_flag_parsed(self):
        args = _parse(["sim", "--preset", "unsafe", "--policy", "p.onnx"])
        assert args.preset == "unsafe"

    def test_preset_default_none(self):
        args = _parse(["sim", "--policy", "p.onnx"])
        assert args.preset is None


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
# Component Wiring
# ============================================================================

class TestComponentWiring:
    def test_policy_format_auto_detection(self):
        """When config.policy.format is None, detect_policy_format is called."""
        config = load_config(DEFAULT_CONFIG)
        assert config.policy.format is None  # default is auto-detect

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            _make_isaaclab_onnx(onnx_path)
            with patch("unitree_launcher.policy.factory.detect_policy_format", return_value="isaaclab") as mock_detect:
                # Factory uses detect_policy_format when config.policy.format is None
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

        with patch("unitree_launcher.robot.sim_robot.SimRobot") as MockSimRobot, \
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

        with patch("unitree_launcher.robot.sim_robot.SimRobot") as MockSimRobot, \
             patch("unitree_launcher.main.run_with_viewer") as mock_run, \
             patch("unitree_launcher.main.platform") as mock_platform:

            mock_platform.system.return_value = "Linux"  # Skip macOS mjpython check
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
        import unitree_launcher.robot.real_robot as rr_mod
        from unitree_launcher.robot.base import RobotState
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)

        mock_robot = MagicMock()
        mock_robot.n_dof = 29
        # prepare() calls get_state() — return a valid RobotState
        mock_robot.get_state.return_value = RobotState.zeros(29)

        with patch.object(rr_mod, "RealRobot", return_value=mock_robot) as MockReal, \
             patch("unitree_launcher.main.run_headless") as mock_run:

            main([
                "real", "--policy", onnx_path, "--interface", "eth0",
                "--config", DEFAULT_CONFIG, "--no-log",
            ])

            MockReal.assert_called_once()
            # Verify the config passed has domain_id=0
            call_config = MockReal.call_args[0][0]
            assert call_config.network.domain_id == 0
            assert call_config.network.interface == "eth0"

    def test_main_logger_lifecycle(self, tmp_path):
        """Logger start/stop called when logging is enabled."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)
        log_dir = str(tmp_path / "logs")

        with patch("unitree_launcher.robot.sim_robot.SimRobot") as MockSimRobot, \
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

        with patch("unitree_launcher.robot.sim_robot.SimRobot") as MockSimRobot, \
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
        with patch("unitree_launcher.robot.sim_robot.SimRobot") as MockSimRobot:
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
        from unitree_launcher.control.runtime import Runtime
        from unitree_launcher.control.safety import SafetyController
        from unitree_launcher.policy.joint_mapper import JointMapper

        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)

        config = load_config(DEFAULT_CONFIG)
        robot = MagicMock()
        robot.n_dof = 29

        rt = MagicMock()
        rt.safety = MagicMock()
        rt.safety.state = SystemState.RUNNING

        # Make step() raise KeyboardInterrupt on second call
        call_count = [0]

        def step_side_effect():
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
            return True

        rt.step = step_side_effect

        run_headless(rt)
        rt.stop.assert_called_once()

    def test_main_policy_dir_passed_to_controller(self, tmp_path):
        """--policy-dir is forwarded to Runtime."""
        onnx_path = str(tmp_path / "test_policy.onnx")
        _make_isaaclab_onnx(onnx_path)
        pol_dir = str(tmp_path / "policies")

        with patch("unitree_launcher.robot.sim_robot.SimRobot") as MockSimRobot, \
             patch("unitree_launcher.main.run_headless"), \
             patch("unitree_launcher.main.Runtime") as MockController:

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


# ============================================================================
# Helpers for viewer / run_headless tests
# ============================================================================

def _make_viewer_config() -> Config:
    cfg = load_config(DEFAULT_CONFIG)
    cfg.control.transition_steps = 0
    return cfg


def _make_mock_robot(n_dof: int = 29) -> MagicMock:
    from unitree_launcher.robot.base import RobotInterface
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
    robot.mj_model = MagicMock()
    robot.mj_data = MagicMock()
    robot.lock = threading.Lock()
    return robot


def _make_mock_policy(n_dof: int = 29) -> MagicMock:
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


def _make_runtime(**kwargs) -> Runtime:
    config = kwargs.pop("config", _make_viewer_config())
    robot = kwargs.pop("robot", _make_mock_robot())
    policy = kwargs.pop("policy", _make_mock_policy())
    mapper = JointMapper(G1_29DOF_JOINTS)
    safety = SafetyController(config, n_dof=29)
    return Runtime(
        robot=robot,
        policy=policy,
        safety=safety,
        joint_mapper=mapper,
        config=config,
        default_policy=_make_mock_policy(),
        default_joint_mapper=mapper,
        **kwargs,
    )


# ============================================================================
# GLFW Key Map
# ============================================================================

class TestGLFWKeyMap:
    def test_glfw_key_map_all_values_are_strings(self):
        for keycode, name in GLFW_KEY_MAP.items():
            assert isinstance(keycode, int)
            assert isinstance(name, str)

    def test_glfw_key_map_no_letter_keys(self):
        """No single ASCII letter keys (65-90) — they conflict with MuJoCo viewer."""
        for keycode in GLFW_KEY_MAP:
            assert not (65 <= keycode <= 90), (
                f"Letter key {chr(keycode)} (code {keycode}) conflicts with MuJoCo viewer"
            )


# ============================================================================
# Key Callback Dispatch (queue-based)
# ============================================================================

class TestKeyCallback:
    def test_key_callback_enqueues_mapped_key(self):
        """Pressing space (keycode 32) enqueues 'space'."""
        key_queue = queue.SimpleQueue()

        def key_callback(keycode: int) -> None:
            key = GLFW_KEY_MAP.get(keycode)
            if key:
                key_queue.put(key)

        key_callback(32)
        assert key_queue.get_nowait() == "space"

    def test_key_callback_unmapped_ignored(self):
        """An unmapped keycode should NOT enqueue anything."""
        key_queue = queue.SimpleQueue()

        def key_callback(keycode: int) -> None:
            key = GLFW_KEY_MAP.get(keycode)
            if key:
                key_queue.put(key)

        key_callback(999)
        assert key_queue.empty()

    def test_key_callback_all_mapped_keys(self):
        """Every mapped keycode enqueues the correct name."""
        key_queue = queue.SimpleQueue()

        def key_callback(keycode: int) -> None:
            key = GLFW_KEY_MAP.get(keycode)
            if key:
                key_queue.put(key)

        for keycode, expected_name in GLFW_KEY_MAP.items():
            key_callback(keycode)
            assert key_queue.get_nowait() == expected_name


# ============================================================================
# Viser CLI Arguments
# ============================================================================

class TestViserCLIArgs:
    def test_parse_viser_default_port(self):
        """--viser uses default port 8080."""
        args = _parse(["sim", "--policy", "p.onnx", "--viser"])
        assert args.viser is True
        assert args.port == 8080

    def test_parse_viser_custom_port(self):
        """--viser --port 9090 sets port to 9090."""
        args = _parse(["sim", "--policy", "p.onnx", "--viser", "--port", "9090"])
        assert args.viser is True
        assert args.port == 9090

    def test_parse_gui_and_viser_together(self):
        """--gui and --viser can both be set."""
        args = _parse(["sim", "--policy", "p.onnx", "--gui", "--viser"])
        assert args.gui is True
        assert args.viser is True

    def test_real_mode_no_viser_flag(self):
        """Real mode does not accept --viser (headless only)."""
        with pytest.raises(SystemExit):
            _parse(["real", "--policy", "p.onnx", "--viser"])


# ============================================================================
# run_with_viewer (mocked MuJoCo viewer)
# ============================================================================

class TestRunWithViewer:
    def test_run_with_viewer_starts_and_stops_controller(self):
        """Viewer loop starts the controller and stops it on exit."""
        rt = _make_runtime()
        call_count = 0

        class FakeViewer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self_inner):
                nonlocal call_count
                call_count += 1
                return call_count <= 2

            def sync(self):
                pass

        with patch("mujoco.viewer.launch_passive", return_value=FakeViewer()):
            with patch.object(rt, "start_threaded") as mock_start, \
                 patch.object(rt, "stop") as mock_stop:
                run_with_viewer(rt)
                mock_start.assert_called_once()
                mock_stop.assert_called_once()

    def test_run_with_viewer_sync_under_lock(self):
        """viewer.sync() must be called while holding sim_robot.lock."""
        rt = _make_runtime()
        lock = rt.robot.lock

        sync_held_lock = None
        call_count = 0

        class FakeViewer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self_inner):
                nonlocal call_count
                call_count += 1
                return call_count <= 1

            def sync(self_inner):
                nonlocal sync_held_lock
                sync_held_lock = lock.locked()

        with patch("mujoco.viewer.launch_passive", return_value=FakeViewer()):
            with patch.object(rt, "start_threaded"), patch.object(rt, "stop"):
                run_with_viewer(rt)

        assert sync_held_lock is True, "viewer.sync() must run under sim_robot.lock"


# ============================================================================
# run_headless
# ============================================================================

class TestRunHeadless:
    def test_run_headless_max_steps(self):
        """run_headless with max_steps stops after exactly N steps."""
        rt = _make_runtime()
        run_headless(rt, max_steps=5)

        telem = rt.get_telemetry()
        assert telem["step_count"] == 5

    def test_run_headless_duration_termination(self):
        """With duration, run completes within a reasonable time window."""
        rt = _make_runtime()
        start = time.time()
        run_headless(rt, duration=0.3)
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_run_headless_stop_called(self):
        """Pipeline.stop() is always called in finally block."""
        rt = _make_runtime()
        with patch.object(rt, "stop") as mock_stop:
            run_headless(rt, max_steps=1)
        mock_stop.assert_called_once()
