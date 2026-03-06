"""Data logging for robot state, observations, actions, and events.

Supports HDF5 and NPZ formats. Buffered data is flushed to disk on a
background thread so the control loop is never blocked by I/O.
"""
from __future__ import annotations

import json
import queue
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
        "observations", "actions",
        "cmd_pos", "cmd_vel", "cmd_torque", "cmd_kp", "cmd_kd",
        "system_state", "vel_cmd", "inference_ms", "loop_ms",
        # Raw sensor state (pre-estimator) for sim2real comparison
        "raw_base_pos", "raw_base_vel", "raw_imu_quat", "raw_imu_gyro",
        # Estimator diagnostics
        "est_contact_left", "est_contact_right",
        "est_gyro_bias", "est_accel_bias",
        # SDK hardware state (real robot only, zeros in sim)
        "sdk_tick", "sdk_mode_pr", "sdk_mode_machine",
        "sdk_motor_mode", "sdk_motor_ddq", "sdk_motor_temperature",
        "sdk_motor_voltage", "sdk_motor_state_flags",
        "sdk_imu_rpy", "sdk_imu_temperature",
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

        # Background writer
        self._write_queue: queue.Queue[Optional[Dict[str, np.ndarray]]] = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None

        # For HDF5 incremental writes (only touched by writer thread)
        self._hdf5_file = None

    def start(self) -> None:
        """Create log directory, open files, start writer thread."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        with open(self._log_dir / "metadata.yaml", "w") as f:
            yaml.dump(self._metadata, f, default_flow_style=False)

        if self._format == "hdf5":
            import h5py
            self._hdf5_file = h5py.File(self._log_dir / "data.hdf5", "w")

        self._started = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="log-writer", daemon=True
        )
        self._writer_thread.start()

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
        raw_state: Optional[RobotState] = None,
        estimator_info: Optional[dict] = None,
    ) -> None:
        """Log one timestep. Called from control loop.

        Only appends to in-memory buffers — no disk I/O on this thread.

        Args:
            raw_state: Pre-estimator robot state (for sim2real comparison).
                If None, raw fields are copied from robot_state.
            estimator_info: Dict with contact_left, contact_right,
                gyro_bias, accel_bias from the state estimator.
        """
        if not self._started:
            return

        raw = raw_state if raw_state is not None else robot_state
        est = estimator_info or {}

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
            self._buffers["cmd_vel"].append(np.asarray(command.joint_velocities, dtype=np.float32))
            self._buffers["cmd_torque"].append(np.asarray(command.joint_torques, dtype=np.float32))
            self._buffers["cmd_kp"].append(np.asarray(command.kp, dtype=np.float32))
            self._buffers["cmd_kd"].append(np.asarray(command.kd, dtype=np.float32))
            self._buffers["system_state"].append(np.int32(system_state.value if isinstance(system_state.value, int) else _state_to_int(system_state)))
            self._buffers["vel_cmd"].append(np.asarray(velocity_command, dtype=np.float32))
            self._buffers["inference_ms"].append(np.float32(timing.get("inference_ms", 0.0)))
            self._buffers["loop_ms"].append(np.float32(timing.get("loop_ms", 0.0)))
            # Raw sensor state (pre-estimator)
            self._buffers["raw_base_pos"].append(np.asarray(raw.base_position, dtype=np.float32))
            self._buffers["raw_base_vel"].append(np.asarray(raw.base_velocity, dtype=np.float32))
            self._buffers["raw_imu_quat"].append(np.asarray(raw.imu_quaternion, dtype=np.float32))
            self._buffers["raw_imu_gyro"].append(np.asarray(raw.imu_angular_velocity, dtype=np.float32))
            # Estimator diagnostics
            self._buffers["est_contact_left"].append(np.float32(est.get("contact_left", 0)))
            self._buffers["est_contact_right"].append(np.float32(est.get("contact_right", 0)))
            self._buffers["est_gyro_bias"].append(np.asarray(est.get("gyro_bias", np.zeros(3)), dtype=np.float32))
            self._buffers["est_accel_bias"].append(np.asarray(est.get("accel_bias", np.zeros(3)), dtype=np.float32))
            # SDK hardware state
            sdk = robot_state.sdk_state
            self._buffers["sdk_tick"].append(np.uint32(sdk.tick if sdk else 0))
            self._buffers["sdk_mode_pr"].append(np.uint8(sdk.mode_pr if sdk else 0))
            self._buffers["sdk_mode_machine"].append(np.uint8(sdk.mode_machine if sdk else 0))
            self._buffers["sdk_motor_mode"].append(sdk.motor_mode if sdk and sdk.motor_mode is not None else np.zeros(35, dtype=np.uint8))
            self._buffers["sdk_motor_ddq"].append(sdk.motor_ddq if sdk and sdk.motor_ddq is not None else np.zeros(35, dtype=np.float32))
            self._buffers["sdk_motor_temperature"].append(sdk.motor_temperature if sdk and sdk.motor_temperature is not None else np.zeros((35, 2), dtype=np.int16))
            self._buffers["sdk_motor_voltage"].append(sdk.motor_voltage if sdk and sdk.motor_voltage is not None else np.zeros(35, dtype=np.float32))
            self._buffers["sdk_motor_state_flags"].append(sdk.motor_state_flags if sdk and sdk.motor_state_flags is not None else np.zeros(35, dtype=np.uint32))
            self._buffers["sdk_imu_rpy"].append(sdk.imu_rpy if sdk and sdk.imu_rpy is not None else np.zeros(3, dtype=np.float32))
            self._buffers["sdk_imu_temperature"].append(np.int16(sdk.imu_temperature if sdk else 0))
            self._step_count += 1

            if self._step_count % self._buffer_size == 0:
                # Swap buffers: hand the full buffer to the writer, start fresh
                batch = {k: np.array(v) for k, v in self._buffers.items() if v}
                for k in self._buffers:
                    self._buffers[k] = []
                self._write_queue.put(batch)

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
        """Flush remaining data, stop writer thread, close files."""
        if not self._started:
            return

        # Flush any remaining buffered data
        with self._lock:
            if any(self._buffers[k] for k in self._buffers):
                batch = {k: np.array(v) for k, v in self._buffers.items() if v}
                for k in self._buffers:
                    self._buffers[k] = []
                self._write_queue.put(batch)

        # Signal writer thread to stop and wait for it
        self._write_queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)

        # Write events
        with open(self._log_dir / "events.json", "w") as f:
            json.dump(self._events, f, indent=2)

        if self._format == "hdf5":
            if self._hdf5_file is not None:
                self._hdf5_file.close()
                self._hdf5_file = None

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

        rate = self._step_count / duration if duration > 0 else 0
        summary = (
            f"[logger] Run '{self._run_name}' complete: "
            f"{self._step_count} steps, "
            f"{duration:.1f}s duration, "
            f"{rate:.1f} Hz, "
            f"{len(self._events)} events, "
            f"format={self._format}"
        )
        print(summary)
        return summary

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Background thread: drain write queue and flush to disk."""
        while True:
            batch = self._write_queue.get()
            if batch is None:
                break  # Shutdown signal
            self._flush_batch(batch)

    def _flush_batch(self, batch: Dict[str, np.ndarray]) -> None:
        """Write a batch of numpy arrays to disk."""
        if self._format == "hdf5":
            self._flush_hdf5(batch)
        else:
            self._flush_npz(batch)

    def _flush_hdf5(self, batch: Dict[str, np.ndarray]) -> None:
        """Flush batch to HDF5 file with gzip compression."""
        f = self._hdf5_file
        if f is None:
            return

        for key in self.DATASET_KEYS:
            if key not in batch:
                continue
            arr = batch[key]
            if arr.size == 0:
                continue
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

    def _flush_npz(self, batch: Dict[str, np.ndarray]) -> None:
        """Flush batch to NPZ file (full rewrite with compression)."""
        data_path = self._log_dir / "data.npz"

        # Load existing data if present
        existing = {}
        if data_path.exists():
            with np.load(data_path) as d:
                existing = {k: d[k] for k in d.files}

        # Merge
        merged = {}
        for key in self.DATASET_KEYS:
            new_arr = batch.get(key)
            if key in existing and existing[key].size > 0:
                if new_arr is not None and new_arr.size > 0:
                    merged[key] = np.concatenate([existing[key], new_arr])
                else:
                    merged[key] = existing[key]
            elif new_arr is not None and new_arr.size > 0:
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
