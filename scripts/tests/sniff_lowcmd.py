#!/usr/bin/env python3
"""Sniff LowCmd_ and LowState_ DDS messages and log to NPZ.

Run this WHILE another controller (MTC, Unitree app, etc.) operates the
robot. Records the raw motor_cmd and motor_state fields so you can compare
what the working controller sends vs what our code sends.

Usage:
    # While MTC or Unitree firmware controls the robot:
    python scripts/tests/sniff_lowcmd.py --interface en8 --duration 10 --capture /tmp/sniff_mtc.npz

    # Compare with our code's commands:
    python scripts/tests/sniff_lowcmd.py --plot /tmp/sniff_mtc.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from unitree_launcher.compat import patch_unitree_b2_import, patch_unitree_threading
patch_unitree_b2_import()
patch_unitree_threading()

N_MOTORS = 29
SHOULDER_IDX = 23  # right_shoulder_roll in IDL order


def run_sniff(args):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    ChannelFactoryInitialize(0, args.interface)

    # Storage
    timestamps = []
    cmd_q = []
    cmd_dq = []
    cmd_tau = []
    cmd_kp = []
    cmd_kd = []
    cmd_mode = []
    state_q = []
    state_dq = []
    state_tau = []

    def on_cmd(msg):
        timestamps.append(time.time())
        cmd_q.append([msg.motor_cmd[i].q for i in range(N_MOTORS)])
        cmd_dq.append([msg.motor_cmd[i].dq for i in range(N_MOTORS)])
        cmd_tau.append([msg.motor_cmd[i].tau for i in range(N_MOTORS)])
        cmd_kp.append([msg.motor_cmd[i].kp for i in range(N_MOTORS)])
        cmd_kd.append([msg.motor_cmd[i].kd for i in range(N_MOTORS)])
        cmd_mode.append([msg.motor_cmd[i].mode for i in range(N_MOTORS)])

    def on_state(msg):
        state_q.append([msg.motor_state[i].q for i in range(N_MOTORS)])
        state_dq.append([msg.motor_state[i].dq for i in range(N_MOTORS)])
        state_tau.append([msg.motor_state[i].tau_est for i in range(N_MOTORS)])

    cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    cmd_sub.Init(on_cmd, 10)

    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    state_sub.Init(on_state, 10)

    print(f"[sniff] Listening on {args.interface} for {args.duration}s...")
    print(f"[sniff] Shoulder (idx {SHOULDER_IDX}) fields will be highlighted.")

    try:
        start = time.time()
        while time.time() - start < args.duration:
            time.sleep(0.5)
            n = len(timestamps)
            if n > 0:
                last = timestamps[-1] - timestamps[0] if n > 1 else 0
                hz = n / last if last > 0 else 0
                kp_sh = cmd_kp[-1][SHOULDER_IDX]
                kd_sh = cmd_kd[-1][SHOULDER_IDX]
                q_sh = cmd_q[-1][SHOULDER_IDX]
                mode_sh = cmd_mode[-1][SHOULDER_IDX]
                print(f"  {n} cmds ({hz:.0f} Hz)  shoulder: mode={mode_sh:#x} "
                      f"q={q_sh:+.4f} kp={kp_sh:.1f} kd={kd_sh:.1f}")
    except KeyboardInterrupt:
        print("\n[sniff] Stopped.")

    n_cmd = len(timestamps)
    n_state = len(state_q)
    print(f"[sniff] Captured {n_cmd} LowCmd, {n_state} LowState messages.")

    if n_cmd > 0:
        np.savez_compressed(
            args.capture,
            timestamps=np.array(timestamps),
            cmd_q=np.array(cmd_q),
            cmd_dq=np.array(cmd_dq),
            cmd_tau=np.array(cmd_tau),
            cmd_kp=np.array(cmd_kp),
            cmd_kd=np.array(cmd_kd),
            cmd_mode=np.array(cmd_mode),
            state_q=np.array(state_q[:n_cmd]) if n_state >= n_cmd else np.array(state_q),
            state_dq=np.array(state_dq[:n_cmd]) if n_state >= n_cmd else np.array(state_dq),
            state_tau=np.array(state_tau[:n_cmd]) if n_state >= n_cmd else np.array(state_tau),
            shoulder_idx=SHOULDER_IDX,
        )
        print(f"[sniff] Saved to {args.capture}")


def plot_sniff(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True,
                              gridspec_kw={"hspace": 0.3})
    colors = ["#1976D2", "#E65100", "#2E7D32", "#C62828"]

    for pi, p in enumerate(paths):
        d = dict(np.load(p, allow_pickle=True))
        label = p.replace(".npz", "").split("/")[-1]
        si = int(d["shoulder_idx"])
        ts = d["timestamps"]
        ts = ts - ts[0]  # relative time
        c = colors[pi % len(colors)]
        n = len(ts)
        hz = n / ts[-1] if ts[-1] > 0 else 0

        print(f"[plot] {label}: {n} samples, {hz:.0f} Hz")
        print(f"  Shoulder mode: {d['cmd_mode'][n//2, si]:#x}")
        print(f"  Shoulder kp: {np.mean(d['cmd_kp'][:, si]):.2f}")
        print(f"  Shoulder kd: {np.mean(d['cmd_kd'][:, si]):.2f}")
        print(f"  Shoulder cmd_dq range: [{np.min(d['cmd_dq'][:, si]):.4f}, {np.max(d['cmd_dq'][:, si]):.4f}]")
        print(f"  Shoulder cmd_tau range: [{np.min(d['cmd_tau'][:, si]):.4f}, {np.max(d['cmd_tau'][:, si]):.4f}]")

        axes[0].plot(ts, d["cmd_q"][:, si], color=c, lw=0.8, label=f"{label} cmd_q")
        axes[0].set_ylabel("cmd_q (rad)")
        axes[0].set_title("Shoulder: Commanded Position")

        axes[1].plot(ts, d["cmd_dq"][:, si], color=c, lw=0.8, label=f"{label} cmd_dq")
        axes[1].set_ylabel("cmd_dq (rad/s)")
        axes[1].set_title("Shoulder: Commanded Velocity")

        axes[2].plot(ts, d["cmd_tau"][:, si], color=c, lw=0.8, label=f"{label} cmd_tau")
        axes[2].set_ylabel("cmd_tau (Nm)")
        axes[2].set_title("Shoulder: Commanded Torque (feedforward)")

        axes[3].plot(ts, d["cmd_kp"][:, si], color=c, lw=0.8, label=f"{label} kp")
        axes[3].set_ylabel("kp")
        axes[3].set_title("Shoulder: Position Gain")

        n_state = len(d.get("state_dq", []))
        if n_state > 0:
            st = ts[:n_state] if n_state <= len(ts) else ts
            axes[4].plot(st[:n_state], d["state_dq"][:n_state, si], color=c, lw=0.5,
                        alpha=0.7, label=f"{label} state_dq")
        axes[4].set_ylabel("state dq (rad/s)")
        axes[4].set_title("Shoulder: Measured Velocity")

        if n_state > 0:
            axes[5].plot(st[:n_state], d["state_tau"][:n_state, si], color=c, lw=0.5,
                        alpha=0.7, label=f"{label} state_tau")
        axes[5].set_ylabel("state tau (Nm)")
        axes[5].set_title("Shoulder: Measured Torque")
        axes[5].set_xlabel("Time (s)")

    for ax in axes:
        ax.legend(fontsize=8)

    out = "sniff_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[plot] Saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Sniff LowCmd/LowState DDS messages")
    parser.add_argument("--interface", default="en8", help="Network interface")
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration (s)")
    parser.add_argument("--capture", default="/tmp/sniff.npz", help="Output NPZ path")
    parser.add_argument("--plot", nargs="+", metavar="NPZ",
                        help="Plot comparison from captured NPZ files")
    args = parser.parse_args()

    if args.plot:
        plot_sniff(args.plot)
    else:
        run_sniff(args)


if __name__ == "__main__":
    main()
