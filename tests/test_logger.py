"""Tests for Phase 10: Logging System (DataLogger + LogReplay)."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from unitree_launcher.config import LoggingConfig
from unitree_launcher.control.safety import SystemState
from unitree_launcher.datalog.logger import DataLogger, _state_to_int
from unitree_launcher.datalog.replay import LogReplay
from unitree_launcher.robot.base import RobotCommand, RobotState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DOF = 29
OBS_DIM = 123
ACTION_DIM = 29


def _make_state(step: int) -> RobotState:
    """Create a RobotState with deterministic values based on step index."""
    return RobotState(
        timestamp=step * 0.02,
        joint_positions=np.full(N_DOF, step * 0.01, dtype=np.float64),
        joint_velocities=np.full(N_DOF, step * 0.001, dtype=np.float64),
        joint_torques=np.full(N_DOF, step * 0.1, dtype=np.float64),
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_angular_velocity=np.array([0.0, 0.0, step * 0.01]),
        imu_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        base_position=np.array([step * 0.001, 0.0, 0.793]),
        base_velocity=np.array([step * 0.01, 0.0, 0.0]),
    )


def _make_command(step: int) -> RobotCommand:
    return RobotCommand(
        joint_positions=np.full(N_DOF, step * 0.01, dtype=np.float64),
        joint_velocities=np.zeros(N_DOF),
        joint_torques=np.zeros(N_DOF),
        kp=np.full(N_DOF, 100.0),
        kd=np.full(N_DOF, 10.0),
    )


def _log_n_steps(logger: DataLogger, n: int = 100) -> None:
    """Log n steps of deterministic data."""
    for i in range(n):
        state = _make_state(i)
        obs = np.full(OBS_DIM, i * 0.01, dtype=np.float32)
        action = np.full(ACTION_DIM, i * 0.001, dtype=np.float32)
        cmd = _make_command(i)
        vel_cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        timing = {"inference_ms": 1.5, "loop_ms": 3.0}
        logger.log_step(
            timestamp=state.timestamp,
            robot_state=state,
            observation=obs,
            action=action,
            command=cmd,
            system_state=SystemState.RUNNING,
            velocity_command=vel_cmd,
            timing=timing,
        )


def _make_logger(tmp_log_dir: str, fmt: str = "hdf5", metadata: dict = None) -> DataLogger:
    config = LoggingConfig(enabled=True, format=fmt)
    return DataLogger(
        config=config,
        run_name="test_run",
        log_dir=tmp_log_dir,
        metadata=metadata,
    )


# ===========================================================================
# DataLogger Tests
# ===========================================================================


class TestDataLoggerDirectory:
    def test_logger_creates_directory(self, tmp_log_dir):
        logger = _make_logger(tmp_log_dir)
        logger.start()
        assert Path(tmp_log_dir, "test_run").is_dir()
        logger.stop()

    def test_logger_creates_nested_directory(self, tmp_log_dir):
        config = LoggingConfig(enabled=True, format="hdf5")
        logger = DataLogger(config, "nested/run", tmp_log_dir)
        logger.start()
        assert Path(tmp_log_dir, "nested", "run").is_dir()
        logger.stop()


class TestDataLoggerMetadata:
    def test_logger_writes_metadata(self, tmp_log_dir):
        meta = {
            "robot": {"variant": "g1_29dof"},
            "policy": {"format": "isaaclab"},
            "control": {"policy_frequency": 50, "kp": 100.0, "kd": 10.0},
        }
        logger = _make_logger(tmp_log_dir, metadata=meta)
        logger.start()
        logger.stop()

        meta_path = Path(tmp_log_dir, "test_run", "metadata.yaml")
        assert meta_path.exists()
        loaded = yaml.safe_load(meta_path.read_text())
        assert loaded["robot"]["variant"] == "g1_29dof"
        assert loaded["policy"]["format"] == "isaaclab"
        assert loaded["control"]["policy_frequency"] == 50


class TestDataLoggerHDF5:
    def test_logger_writes_data_hdf5(self, tmp_log_dir):
        import h5py
        logger = _make_logger(tmp_log_dir, fmt="hdf5")
        logger.start()
        _log_n_steps(logger, 100)
        logger.stop()

        with h5py.File(Path(tmp_log_dir, "test_run", "data.hdf5"), "r") as f:
            assert f["timestamps"].shape == (100,)
            assert f["joint_pos"].shape == (100, N_DOF)
            assert f["joint_vel"].shape == (100, N_DOF)
            assert f["joint_torques"].shape == (100, N_DOF)
            assert f["imu_quat"].shape == (100, 4)
            assert f["imu_gyro"].shape == (100, 3)
            assert f["imu_accel"].shape == (100, 3)
            assert f["base_pos"].shape == (100, 3)
            assert f["base_vel"].shape == (100, 3)
            assert f["observations"].shape == (100, OBS_DIM)
            assert f["actions"].shape == (100, ACTION_DIM)
            assert f["cmd_pos"].shape == (100, N_DOF)
            assert f["cmd_kp"].shape == (100, N_DOF)
            assert f["cmd_kd"].shape == (100, N_DOF)
            assert f["system_state"].shape == (100,)
            assert f["vel_cmd"].shape == (100, 3)
            assert f["inference_ms"].shape == (100,)
            assert f["loop_ms"].shape == (100,)

    def test_logger_compression(self, tmp_log_dir):
        import h5py
        logger = _make_logger(tmp_log_dir, fmt="hdf5")
        logger.start()
        _log_n_steps(logger, 100)
        logger.stop()

        with h5py.File(Path(tmp_log_dir, "test_run", "data.hdf5"), "r") as f:
            assert f["joint_pos"].compression == "gzip"
            assert f["timestamps"].compression == "gzip"

    def test_logger_roundtrip_hdf5(self, tmp_log_dir):
        import h5py
        logger = _make_logger(tmp_log_dir, fmt="hdf5")
        logger.start()
        _log_n_steps(logger, 50)
        logger.stop()

        with h5py.File(Path(tmp_log_dir, "test_run", "data.hdf5"), "r") as f:
            # Verify specific values from step 10
            np.testing.assert_allclose(f["timestamps"][10], 10 * 0.02, atol=1e-6)
            np.testing.assert_allclose(f["joint_pos"][10], np.full(N_DOF, 10 * 0.01), atol=1e-5)
            np.testing.assert_allclose(f["base_pos"][10], [10 * 0.001, 0.0, 0.793], atol=1e-5)
            np.testing.assert_allclose(f["vel_cmd"][10], [0.3, 0.0, 0.0], atol=1e-5)
            np.testing.assert_allclose(f["inference_ms"][10], 1.5, atol=1e-3)


class TestDataLoggerNPZ:
    def test_logger_writes_data_npz(self, tmp_log_dir):
        logger = _make_logger(tmp_log_dir, fmt="npz")
        logger.start()
        _log_n_steps(logger, 100)
        logger.stop()

        data_path = Path(tmp_log_dir, "test_run", "data.npz")
        with np.load(data_path) as d:
            assert d["timestamps"].shape == (100,)
            assert d["joint_pos"].shape == (100, N_DOF)
            assert d["observations"].shape == (100, OBS_DIM)
            assert d["actions"].shape == (100, ACTION_DIM)
            assert d["system_state"].shape == (100,)

    def test_logger_roundtrip_npz(self, tmp_log_dir):
        logger = _make_logger(tmp_log_dir, fmt="npz")
        logger.start()
        _log_n_steps(logger, 50)
        logger.stop()

        data_path = Path(tmp_log_dir, "test_run", "data.npz")
        with np.load(data_path) as d:
            np.testing.assert_allclose(d["timestamps"][10], 10 * 0.02, atol=1e-6)
            np.testing.assert_allclose(d["joint_pos"][10], np.full(N_DOF, 10 * 0.01), atol=1e-5)
            np.testing.assert_allclose(d["vel_cmd"][10], [0.3, 0.0, 0.0], atol=1e-5)


class TestDataLoggerEvents:
    def test_logger_writes_events(self, tmp_log_dir):
        logger = _make_logger(tmp_log_dir)
        logger.start()
        logger.log_event("start", {"mode": "sim"})
        logger.log_event("estop", {"reason": "orientation"})
        logger.stop()

        events_path = Path(tmp_log_dir, "test_run", "events.json")
        assert events_path.exists()
        events = json.loads(events_path.read_text())
        assert len(events) == 2
        assert events[0]["type"] == "start"
        assert events[0]["data"]["mode"] == "sim"
        assert events[1]["type"] == "estop"
        assert events[1]["data"]["reason"] == "orientation"
        assert "timestamp" in events[0]
        assert isinstance(events[0]["timestamp"], float)


class TestDataLoggerMisc:
    def test_logger_stop_prints_summary(self, tmp_log_dir, capsys):
        logger = _make_logger(tmp_log_dir)
        logger.start()
        _log_n_steps(logger, 10)
        logger.stop()

        captured = capsys.readouterr()
        assert "test_run" in captured.out
        assert "10 steps" in captured.out

    def test_logger_handles_empty_run(self, tmp_log_dir):
        logger = _make_logger(tmp_log_dir)
        logger.start()
        logger.stop()
        # Should not raise

        events_path = Path(tmp_log_dir, "test_run", "events.json")
        assert events_path.exists()
        events = json.loads(events_path.read_text())
        assert events == []

    def test_logger_nonblocking(self, tmp_log_dir):
        logger = _make_logger(tmp_log_dir, fmt="hdf5")
        logger.start()

        start_time = time.monotonic()
        _log_n_steps(logger, 100)
        elapsed = time.monotonic() - start_time

        logger.stop()
        # 100 steps at <1ms each = <100ms total
        # Be generous with the threshold for CI environments
        assert elapsed < 2.0, f"Logging 100 steps took {elapsed:.3f}s (expected <2s)"
        per_step = elapsed / 100
        assert per_step < 0.01, f"Per-step logging took {per_step*1000:.1f}ms (expected <10ms)"


class TestStateToInt:
    def test_state_mapping(self):
        assert _state_to_int(SystemState.IDLE) == 0
        assert _state_to_int(SystemState.RUNNING) == 1
        assert _state_to_int(SystemState.STOPPED) == 2
        assert _state_to_int(SystemState.ESTOP) == 3


# ===========================================================================
# LogReplay Tests
# ===========================================================================


def _create_logged_run(tmp_log_dir: str, fmt: str = "hdf5", n_steps: int = 50) -> str:
    """Create a complete logged run and return the run directory path."""
    meta = {"robot": {"variant": "g1_29dof"}, "control": {"kp": 100}}
    logger = _make_logger(tmp_log_dir, fmt=fmt, metadata=meta)
    logger.start()
    _log_n_steps(logger, n_steps)
    logger.log_event("start", {"mode": "sim"})
    logger.stop()
    return str(Path(tmp_log_dir) / "test_run")


class TestLogReplayLoad:
    def test_replay_load(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir)
        replay = LogReplay(run_dir)
        replay.load()
        assert replay.n_steps == 50

    def test_replay_load_missing_data(self, tmp_path):
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        replay = LogReplay(str(empty_dir))
        with pytest.raises(FileNotFoundError):
            replay.load()


class TestLogReplayMetadata:
    def test_replay_metadata(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir)
        replay = LogReplay(run_dir)
        replay.load()
        assert replay.metadata["robot"]["variant"] == "g1_29dof"
        assert replay.metadata["control"]["kp"] == 100


class TestLogReplayStateAt:
    def test_replay_state_at(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir)
        replay = LogReplay(run_dir)
        replay.load()

        state = replay.get_state_at(10)
        assert isinstance(state, RobotState)
        np.testing.assert_allclose(state.timestamp, 10 * 0.02, atol=1e-5)
        np.testing.assert_allclose(state.joint_positions, np.full(N_DOF, 10 * 0.01), atol=1e-4)
        np.testing.assert_allclose(state.base_position, [10 * 0.001, 0.0, 0.793], atol=1e-4)

    def test_replay_state_at_bounds(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir, n_steps=10)
        replay = LogReplay(run_dir)
        replay.load()

        replay.get_state_at(0)   # first
        replay.get_state_at(9)   # last
        with pytest.raises(IndexError):
            replay.get_state_at(10)
        with pytest.raises(IndexError):
            replay.get_state_at(-1)


class TestLogReplayObservationAction:
    def test_replay_observation_at(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir)
        replay = LogReplay(run_dir)
        replay.load()

        obs = replay.get_observation_at(5)
        assert obs.shape == (OBS_DIM,)
        np.testing.assert_allclose(obs, np.full(OBS_DIM, 5 * 0.01), atol=1e-5)

    def test_replay_action_at(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir)
        replay = LogReplay(run_dir)
        replay.load()

        action = replay.get_action_at(5)
        assert action.shape == (ACTION_DIM,)
        np.testing.assert_allclose(action, np.full(ACTION_DIM, 5 * 0.001), atol=1e-5)


class TestLogReplayCSV:
    def test_replay_to_csv(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir)
        replay = LogReplay(run_dir)
        replay.load()

        csv_path = str(Path(tmp_log_dir) / "export.csv")
        replay.to_csv(csv_path)

        assert Path(csv_path).exists()
        lines = Path(csv_path).read_text().strip().split("\n")
        assert len(lines) == 51  # header + 50 data rows

        header = lines[0].split(",")
        assert header[0] == "timestamp"
        assert "joint_pos_0" in header
        assert "system_state" in header


class TestLogReplaySummary:
    def test_replay_summary(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir)
        replay = LogReplay(run_dir)
        replay.load()

        summary = replay.summary()
        assert "test_run" in summary
        assert "50" in summary  # steps
        assert "29" in summary  # DOFs
        assert "hdf5" in summary


class TestLogReplayAutoDetect:
    def test_replay_auto_detect_format_hdf5(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir, fmt="hdf5")
        replay = LogReplay(run_dir)
        replay.load()
        assert replay.format == "hdf5"
        assert replay.n_steps == 50

    def test_replay_auto_detect_format_npz(self, tmp_log_dir):
        run_dir = _create_logged_run(tmp_log_dir, fmt="npz")
        replay = LogReplay(run_dir)
        replay.load()
        assert replay.format == "npz"
        assert replay.n_steps == 50

    def test_replay_roundtrip_npz(self, tmp_log_dir):
        """Full roundtrip: log via NPZ, replay, verify values."""
        run_dir = _create_logged_run(tmp_log_dir, fmt="npz", n_steps=20)
        replay = LogReplay(run_dir)
        replay.load()

        state = replay.get_state_at(15)
        np.testing.assert_allclose(state.timestamp, 15 * 0.02, atol=1e-5)
        np.testing.assert_allclose(state.joint_positions, np.full(N_DOF, 15 * 0.01), atol=1e-4)

        obs = replay.get_observation_at(15)
        np.testing.assert_allclose(obs, np.full(OBS_DIM, 15 * 0.01), atol=1e-5)
