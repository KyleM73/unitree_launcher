"""Data logging for robot state, observations, actions, and events.

Supports HDF5 and NPZ formats with buffered writes.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from unitree_launcher.config import LoggingConfig
from unitree_launcher.control.safety import SystemState
from unitree_launcher.robot.base import RobotCommand, RobotState


class DataLogger:
    """Logs time-series robot data and discrete events to disk.

    File structure:
        logs/{timestamp}_{mode}_{policy_name}/
        ├── metadata.yaml      # Run configuration snapshot
        ├── data.hdf5          # Compressed time-series data (or data.npz)
        └── events.json        # Discrete events
    """

    DATASET_KEYS = [
        "timestamps", "joint_pos", "joint_vel", "joint_torques",
        "imu_quat", "imu_gyro", "imu_accel", "base_pos", "base_vel",
        "observations", "actions", "cmd_pos", "cmd_kp", "cmd_kd",
        "system_state", "vel_cmd", "inference_ms", "loop_ms",
    ]

    def __init__(
        self,
        config: LoggingConfig,
        run_name: str,
        log_dir: str,
        metadata: Optional[dict] = None,
    ) -> None:
        self._config = config
        self._run_name = run_name
        self._log_dir = Path(log_dir) / run_name
        self._metadata = metadata or {}
        self._format = config.format.lower()
        if self._format not in ("hdf5", "npz"):
            raise ValueError(f"Unsupported logging format: {self._format}")

        self._buffers: Dict[str, List[np.ndarray]] = {k: [] for k in self.DATASET_KEYS}
        self._events: List[dict] = []
        self._step_count = 0
        self._buffer_size = 100
        self._started = False
        self._lock = threading.Lock()

        # For HDF5 incremental writes
        self._hdf5_file = None

    def start(self) -> None:
        """Create log directory and open files."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        with open(self._log_dir / "metadata.yaml", "w") as f:
            yaml.dump(self._metadata, f, default_flow_style=False)

        if self._format == "hdf5":
            import h5py
            self._hdf5_file = h5py.File(self._log_dir / "data.hdf5", "w")

        self._started = True

    def log_step(
        self,
        timestamp: float,
        robot_state: RobotState,
        observation: np.ndarray,
        action: np.ndarray,
        command: RobotCommand,
        system_state: SystemState,
        velocity_command: np.ndarray,
        timing: dict,
    ) -> None:
        """Log one timestep. Called from control loop."""
        if not self._started:
            return

        with self._lock:
            self._buffers["timestamps"].append(np.float64(timestamp))
            self._buffers["joint_pos"].append(np.asarray(robot_state.joint_positions, dtype=np.float32))
            self._buffers["joint_vel"].append(np.asarray(robot_state.joint_velocities, dtype=np.float32))
            self._buffers["joint_torques"].append(np.asarray(robot_state.joint_torques, dtype=np.float32))
            self._buffers["imu_quat"].append(np.asarray(robot_state.imu_quaternion, dtype=np.float32))
            self._buffers["imu_gyro"].append(np.asarray(robot_state.imu_angular_velocity, dtype=np.float32))
            self._buffers["imu_accel"].append(np.asarray(robot_state.imu_linear_acceleration, dtype=np.float32))
            self._buffers["base_pos"].append(np.asarray(robot_state.base_position, dtype=np.float32))
            self._buffers["base_vel"].append(np.asarray(robot_state.base_velocity, dtype=np.float32))
            self._buffers["observations"].append(np.asarray(observation, dtype=np.float32))
            self._buffers["actions"].append(np.asarray(action, dtype=np.float32))
            self._buffers["cmd_pos"].append(np.asarray(command.joint_positions, dtype=np.float32))
            self._buffers["cmd_kp"].append(np.asarray(command.kp, dtype=np.float32))
            self._buffers["cmd_kd"].append(np.asarray(command.kd, dtype=np.float32))
            self._buffers["system_state"].append(np.int32(system_state.value if isinstance(system_state.value, int) else _state_to_int(system_state)))
            self._buffers["vel_cmd"].append(np.asarray(velocity_command, dtype=np.float32))
            self._buffers["inference_ms"].append(np.float32(timing.get("inference_ms", 0.0)))
            self._buffers["loop_ms"].append(np.float32(timing.get("loop_ms", 0.0)))
            self._step_count += 1

            if self._step_count % self._buffer_size == 0:
                self._flush_buffer()

    def log_event(self, event_type: str, data: dict) -> None:
        """Log discrete event (start, stop, e-stop, etc.)."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        }
        with self._lock:
            self._events.append(event)

    def stop(self) -> None:
        """Flush and close files. Print summary statistics."""
        if not self._started:
            return

        with self._lock:
            self._flush_buffer()

        # Write events
        with open(self._log_dir / "events.json", "w") as f:
            json.dump(self._events, f, indent=2)

        if self._format == "hdf5":
            if self._hdf5_file is not None:
                self._hdf5_file.close()
                self._hdf5_file = None
        # NPZ: all data written in _flush_buffer, final flush done above

        self._started = False

        # Summary
        duration = 0.0
        if self._step_count > 0 and self._format == "hdf5":
            import h5py
            with h5py.File(self._log_dir / "data.hdf5", "r") as f:
                if "timestamps" in f and len(f["timestamps"]) > 1:
                    duration = float(f["timestamps"][-1] - f["timestamps"][0])
        elif self._step_count > 0 and self._format == "npz":
            data_path = self._log_dir / "data.npz"
            if data_path.exists():
                with np.load(data_path) as d:
                    if "timestamps" in d and len(d["timestamps"]) > 1:
                        duration = float(d["timestamps"][-1] - d["timestamps"][0])

        summary = (
            f"[logger] Run '{self._run_name}' complete: "
            f"{self._step_count} steps, "
            f"{duration:.1f}s duration, "
            f"{len(self._events)} events, "
            f"format={self._format}"
        )
        print(summary)
        return summary

    def _flush_buffer(self) -> None:
        """Write buffered data to disk. Must be called with lock held."""
        if not self._buffers["timestamps"]:
            return

        if self._format == "hdf5":
            self._flush_hdf5()
        else:
            self._flush_npz()

        # Clear buffers
        for k in self._buffers:
            self._buffers[k].clear()

    def _flush_hdf5(self) -> None:
        """Flush buffer to HDF5 file with gzip compression."""
        import h5py
        f = self._hdf5_file
        if f is None:
            return

        for key in self.DATASET_KEYS:
            buf = self._buffers[key]
            if not buf:
                continue
            arr = np.array(buf)
            if key not in f:
                maxshape = (None,) + arr.shape[1:]
                f.create_dataset(
                    key, data=arr, maxshape=maxshape,
                    compression="gzip", chunks=True,
                )
            else:
                ds = f[key]
                old_len = ds.shape[0]
                ds.resize(old_len + arr.shape[0], axis=0)
                ds[old_len:] = arr
        f.flush()

    def _flush_npz(self) -> None:
        """Flush buffer to NPZ file (full rewrite with compression)."""
        data_path = self._log_dir / "data.npz"

        # Load existing data if present
        existing = {}
        if data_path.exists():
            with np.load(data_path) as d:
                existing = {k: d[k] for k in d.files}

        # Merge buffers
        merged = {}
        for key in self.DATASET_KEYS:
            buf = self._buffers[key]
            if not buf:
                new_arr = np.array([])
            else:
                new_arr = np.array(buf)
            if key in existing and existing[key].size > 0:
                merged[key] = np.concatenate([existing[key], new_arr])
            elif new_arr.size > 0:
                merged[key] = new_arr

        if merged:
            np.savez_compressed(data_path, **merged)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def log_path(self) -> str:
        return str(self._log_dir)


def _state_to_int(state: SystemState) -> int:
    """Convert SystemState enum to int for storage."""
    mapping = {
        SystemState.IDLE: 0,
        SystemState.RUNNING: 1,
        SystemState.STOPPED: 2,
        SystemState.ESTOP: 3,
    }
    return mapping.get(state, -1)
