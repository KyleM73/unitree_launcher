"""Log replay: load recorded data and reconstruct robot states.

Auto-detects HDF5 vs NPZ format from file extension.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from unitree_launcher.robot.base import RobotState


class LogReplay:
    """Load and replay logged robot data.

    Usage:
        replay = LogReplay("/path/to/logs/run_name")
        replay.load()
        state = replay.get_state_at(0)
    """

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._data: dict = {}
        self._metadata: dict = {}
        self._format: Optional[str] = None
        self._loaded = False

    def load(self) -> None:
        """Load data and metadata from disk. Auto-detects format."""
        # Load metadata
        meta_path = self._log_dir / "metadata.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = yaml.safe_load(f) or {}

        # Auto-detect format
        hdf5_path = self._log_dir / "data.hdf5"
        npz_path = self._log_dir / "data.npz"

        if hdf5_path.exists():
            self._format = "hdf5"
            self._load_hdf5(hdf5_path)
        elif npz_path.exists():
            self._format = "npz"
            self._load_npz(npz_path)
        else:
            raise FileNotFoundError(
                f"No data file found in {self._log_dir}. "
                "Expected data.hdf5 or data.npz."
            )

        self._loaded = True

    def _load_hdf5(self, path: Path) -> None:
        import h5py
        with h5py.File(path, "r") as f:
            self._data = {key: f[key][:] for key in f.keys()}

    def _load_npz(self, path: Path) -> None:
        with np.load(path) as d:
            self._data = {key: d[key] for key in d.files}

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def format(self) -> str:
        if self._format is None:
            raise RuntimeError("Call load() first")
        return self._format

    @property
    def duration(self) -> float:
        ts = self._data.get("timestamps")
        if ts is None or len(ts) < 2:
            return 0.0
        return float(ts[-1] - ts[0])

    @property
    def n_steps(self) -> int:
        ts = self._data.get("timestamps")
        if ts is None:
            return 0
        return len(ts)

    def get_state_at(self, step: int) -> RobotState:
        """Reconstruct a RobotState from logged data at the given step index."""
        if not self._loaded:
            raise RuntimeError("Call load() first")
        if step < 0 or step >= self.n_steps:
            raise IndexError(f"Step {step} out of range [0, {self.n_steps})")

        return RobotState(
            timestamp=float(self._data["timestamps"][step]),
            joint_positions=self._data["joint_pos"][step].astype(np.float64),
            joint_velocities=self._data["joint_vel"][step].astype(np.float64),
            joint_torques=self._data["joint_torques"][step].astype(np.float64),
            imu_quaternion=self._data["imu_quat"][step].astype(np.float64),
            imu_angular_velocity=self._data["imu_gyro"][step].astype(np.float64),
            imu_linear_acceleration=self._data["imu_accel"][step].astype(np.float64),
            base_position=self._data["base_pos"][step].astype(np.float64),
            base_velocity=self._data["base_vel"][step].astype(np.float64),
        )

    def get_observation_at(self, step: int) -> np.ndarray:
        """Return the observation vector at the given step."""
        if not self._loaded:
            raise RuntimeError("Call load() first")
        if step < 0 or step >= self.n_steps:
            raise IndexError(f"Step {step} out of range [0, {self.n_steps})")
        return self._data["observations"][step].copy()

    def get_action_at(self, step: int) -> np.ndarray:
        """Return the action vector at the given step."""
        if not self._loaded:
            raise RuntimeError("Call load() first")
        if step < 0 or step >= self.n_steps:
            raise IndexError(f"Step {step} out of range [0, {self.n_steps})")
        return self._data["actions"][step].copy()

    def to_csv(self, output_path: str) -> None:
        """Export logged data to CSV."""
        if not self._loaded:
            raise RuntimeError("Call load() first")

        n = self.n_steps
        if n == 0:
            return

        n_dof = self._data["joint_pos"].shape[1] if "joint_pos" in self._data else 0

        # Build header
        header = ["timestamp"]
        for i in range(n_dof):
            header.append(f"joint_pos_{i}")
        for i in range(n_dof):
            header.append(f"joint_vel_{i}")
        for i in range(n_dof):
            header.append(f"joint_torque_{i}")
        header.extend(["imu_qw", "imu_qx", "imu_qy", "imu_qz"])
        header.extend(["imu_gx", "imu_gy", "imu_gz"])
        header.extend(["imu_ax", "imu_ay", "imu_az"])
        header.extend(["base_x", "base_y", "base_z"])
        header.extend(["base_vx", "base_vy", "base_vz"])
        header.extend(["vel_cmd_x", "vel_cmd_y", "vel_cmd_yaw"])
        header.extend(["system_state", "inference_ms", "loop_ms"])

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range(n):
                row = [f"{self._data['timestamps'][i]:.6f}"]
                row.extend(f"{v:.6f}" for v in self._data["joint_pos"][i])
                row.extend(f"{v:.6f}" for v in self._data["joint_vel"][i])
                row.extend(f"{v:.6f}" for v in self._data["joint_torques"][i])
                row.extend(f"{v:.6f}" for v in self._data["imu_quat"][i])
                row.extend(f"{v:.6f}" for v in self._data["imu_gyro"][i])
                row.extend(f"{v:.6f}" for v in self._data["imu_accel"][i])
                row.extend(f"{v:.6f}" for v in self._data["base_pos"][i])
                row.extend(f"{v:.6f}" for v in self._data["base_vel"][i])
                row.extend(f"{v:.6f}" for v in self._data["vel_cmd"][i])
                row.append(str(int(self._data["system_state"][i])))
                row.append(f"{self._data['inference_ms'][i]:.3f}")
                row.append(f"{self._data['loop_ms'][i]:.3f}")
                writer.writerow(row)

    def summary(self) -> str:
        """Return a human-readable summary of the logged data."""
        if not self._loaded:
            raise RuntimeError("Call load() first")

        n = self.n_steps
        dur = self.duration
        fmt = self._format or "unknown"
        n_dof = self._data["joint_pos"].shape[1] if "joint_pos" in self._data and n > 0 else 0

        lines = [
            f"Log: {self._log_dir.name}",
            f"Format: {fmt}",
            f"Steps: {n}",
            f"Duration: {dur:.2f}s",
            f"DOFs: {n_dof}",
        ]

        if n > 0 and "inference_ms" in self._data:
            inf_ms = self._data["inference_ms"]
            lines.append(f"Inference: mean={np.mean(inf_ms):.2f}ms, max={np.max(inf_ms):.2f}ms")

        if n > 0 and "loop_ms" in self._data:
            loop_ms = self._data["loop_ms"]
            lines.append(f"Loop: mean={np.mean(loop_ms):.2f}ms, max={np.max(loop_ms):.2f}ms")

        return "\n".join(lines)
