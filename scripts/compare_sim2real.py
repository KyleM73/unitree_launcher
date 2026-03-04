#!/usr/bin/env python3
"""Compare sim vs real gantry arm test logs.

Loads two HDF5 log directories (sim run and real run) and plots
joint position tracking, error, and torques for the sinusoid joint.

Usage:
    uv run python scripts/compare_sim2real.py logs/<sim_run>/ logs/<real_run>/
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare sim vs real gantry arm test logs")
    parser.add_argument("sim_log", help="Path to sim log directory")
    parser.add_argument("real_log", help="Path to real log directory")
    parser.add_argument("--joint", default="right_shoulder_pitch",
                        help="Joint name to compare (default: right_shoulder_pitch)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save plot to file instead of showing")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: uv pip install matplotlib")
        sys.exit(1)

    import h5py
    from unitree_launcher.config import G1_29DOF_JOINTS

    joint_idx = G1_29DOF_JOINTS.index(args.joint)

    datasets = {}
    for label, log_dir in [("sim", args.sim_log), ("real", args.real_log)]:
        data_path = Path(log_dir) / "data.hdf5"
        if not data_path.exists():
            print(f"ERROR: {data_path} not found")
            sys.exit(1)

        with h5py.File(data_path, "r") as f:
            datasets[label] = {
                "timestamps": f["timestamps"][:],
                "joint_pos": f["joint_pos"][:, joint_idx],
                "joint_vel": f["joint_vel"][:, joint_idx],
                "joint_torque": f["joint_torque"][:, joint_idx] if "joint_torque" in f else None,
                "cmd_pos": f["cmd_pos"][:, joint_idx] if "cmd_pos" in f else None,
            }

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Sim vs Real: {args.joint}")

    for label, d in datasets.items():
        t = d["timestamps"] - d["timestamps"][0]

        # Position tracking
        axes[0].plot(t, np.degrees(d["joint_pos"]), label=f"{label} actual")
        if d["cmd_pos"] is not None:
            axes[0].plot(t, np.degrees(d["cmd_pos"]), "--", alpha=0.5,
                         label=f"{label} command")
        axes[0].set_ylabel("Position (deg)")
        axes[0].legend()

        # Velocity
        axes[1].plot(t, d["joint_vel"], label=label)
        axes[1].set_ylabel("Velocity (rad/s)")
        axes[1].legend()

        # Torque
        if d["joint_torque"] is not None:
            axes[2].plot(t, d["joint_torque"], label=label)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
