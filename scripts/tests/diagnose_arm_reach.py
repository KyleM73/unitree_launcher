#!/usr/bin/env python3
"""Diagnostic: capture sim vs real arm reach data and compare.

Runs a simplified shoulder sinusoid (5s) and logs commanded vs actual
positions, velocities, torques, and timing at every control step.
Saves data to NPZ files for comparison. Run with --plot to generate
a multi-panel comparison figure.

Usage:
    # 1. Capture sim data
    python scripts/tests/diagnose_arm_reach.py sim --capture sim_arm.npz

    # 2. Capture real data (robot must be on gantry, connected)
    python scripts/tests/diagnose_arm_reach.py real --interface en8 --capture real_arm.npz

    # 3. Compare
    python scripts/tests/diagnose_arm_reach.py --plot sim_arm.npz real_arm.npz
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    JOINT_LIMITS_29DOF,
    load_config,
)
from unitree_launcher.gantry import (
    ElasticBand,
    build_gain_arrays,
    enable_gantry,
    get_torso_body_id,
    setup_gantry_band,
    smooth_alpha,
)
from unitree_launcher.robot.base import RobotCommand

from unitree_launcher.script_utils import phase_settle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLICY_HZ = 50
DT = 1.0 / POLICY_HZ
SETTLE_DURATION = 3.0
GAIN_RAMP_DURATION = 0.5
SINUSOID_DURATION = 5.0
SINUSOID_FREQ_HZ = 0.2
N_DOF = 29

SHOULDER_JOINT = "right_shoulder_roll"
SHOULDER_IDX = G1_29DOF_JOINTS.index(SHOULDER_JOINT)
SHOULDER_CLEAR_OFFSET = -0.30


def _build_shoulder_amplitude(init_q):
    lo, hi = JOINT_LIMITS_29DOF[SHOULDER_JOINT]
    home = init_q[SHOULDER_IDX]
    eighth_rom = (hi - lo) / 4.0
    return min(eighth_rom, hi - home, home - lo)


# ---------------------------------------------------------------------------
# Data capture
# ---------------------------------------------------------------------------

class DiagCapture:
    """Captures per-step diagnostic data."""

    def __init__(self):
        self.timestamps = []
        self.wall_times = []
        self.cmd_pos = []
        self.cmd_vel = []
        self.cmd_kp = []
        self.cmd_kd = []
        self.cmd_tau = []
        self.actual_pos = []
        self.actual_vel = []
        self.actual_torques = []

    def record(self, t, cmd, state):
        self.timestamps.append(t)
        self.wall_times.append(time.perf_counter())
        self.cmd_pos.append(cmd.joint_positions.copy())
        self.cmd_vel.append(cmd.joint_velocities.copy())
        self.cmd_kp.append(cmd.kp.copy())
        self.cmd_kd.append(cmd.kd.copy())
        self.cmd_tau.append(cmd.joint_torques.copy())
        self.actual_pos.append(state.joint_positions.copy())
        self.actual_vel.append(state.joint_velocities.copy())
        self.actual_torques.append(state.joint_torques.copy())

    def save(self, path):
        wall = np.array(self.wall_times)
        np.savez_compressed(
            path,
            timestamps=np.array(self.timestamps),
            wall_dt=np.diff(wall, prepend=wall[0]) if len(wall) > 0 else np.array([]),
            cmd_pos=np.array(self.cmd_pos),
            cmd_vel=np.array(self.cmd_vel),
            cmd_kp=np.array(self.cmd_kp),
            cmd_kd=np.array(self.cmd_kd),
            cmd_tau=np.array(self.cmd_tau),
            actual_pos=np.array(self.actual_pos),
            actual_vel=np.array(self.actual_vel),
            actual_torques=np.array(self.actual_torques),
            shoulder_idx=SHOULDER_IDX,
            policy_hz=POLICY_HZ,
            sinusoid_freq=SINUSOID_FREQ_HZ,
            joint_names=G1_29DOF_JOINTS,
            gains=np.array([]),  # placeholder for metadata
        )
        print(f"[diag] Saved {len(self.timestamps)} steps to {path}")


# ---------------------------------------------------------------------------
# Run capture
# ---------------------------------------------------------------------------

def run_capture(args):
    from unitree_launcher.script_utils import create_robot

    config = load_config(args.config)
    mode = args.mode
    is_sim = (mode == "sim")
    robot = create_robot(
        mode, config,
        interface=getattr(args, "interface", None),
        backend=getattr(args, "backend", "python"),
    )

    cmd_hz = args.hz
    cmd_dt = 1.0 / cmd_hz

    passive = getattr(args, "passive", False)
    gains = args.gains
    kp, kd = build_gain_arrays(gains)
    kp *= args.kp_scale
    kd *= args.kd_scale
    torque_mode = args.torque_mode

    def make_cmd(target_q, target_dq, state):
        """Build RobotCommand — either position PD or computed torque."""
        if torque_mode:
            # Compute PD torque in Python, send as feedforward with zero gains.
            # This bypasses the motor's internal PD and avoids 500Hz PD vs 50Hz
            # target interaction.
            tau = kp * (target_q - state.joint_positions) \
                + kd * (target_dq - state.joint_velocities)
            return RobotCommand(
                joint_positions=state.joint_positions,  # hold current (unused with kp=0)
                joint_velocities=np.zeros(N_DOF),
                joint_torques=tau,
                kp=np.zeros(N_DOF),
                kd=np.zeros(N_DOF),
            )
        else:
            return RobotCommand(
                joint_positions=target_q,
                joint_velocities=target_dq,
                joint_torques=np.zeros(N_DOF),
                kp=kp,
                kd=kd,
            )

    band = ElasticBand()
    torso_id = get_torso_body_id(robot.mj_model) if is_sim else 0

    if is_sim:
        enable_gantry(robot)
        setup_gantry_band(robot, band, torso_id)

    robot.connect()
    mode_str = "TORQUE" if torque_mode else "POSITION PD"
    print(f"[diag] Connected ({mode}). Control={mode_str}, {cmd_hz} Hz, Gains={gains}: "
          f"shoulder kp={kp[SHOULDER_IDX]:.1f}, kd={kd[SHOULDER_IDX]:.1f}")

    cap = DiagCapture()

    try:
        # Passive mode: observe only, zero torque.
        if passive:
            obs_dur = 5.0
            obs_steps = int(obs_dur * cmd_hz)
            print(f"[diag] PASSIVE observation ({obs_dur}s, kp=0 kd=0 tau=0)...")
            zero_cmd = RobotCommand(
                joint_positions=np.zeros(N_DOF),
                joint_velocities=np.zeros(N_DOF),
                joint_torques=np.zeros(N_DOF),
                kp=np.zeros(N_DOF),
                kd=np.zeros(N_DOF),
            )
            for step_i in range(obs_steps):
                step_start = time.perf_counter()
                t = (step_i + 1) * cmd_dt
                state = robot.get_state()
                robot.send_command(zero_cmd)
                robot.step()
                cap.record(t, zero_cmd, state)
                remaining = cmd_dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                if (step_i + 1) % cmd_hz == 0:
                    vel = state.joint_velocities[SHOULDER_IDX]
                    all_vel = np.max(np.abs(state.joint_velocities))
                    print(f"  t={t:.1f}s  shoulder_dq={vel:+.4f}  max|dq|={all_vel:.4f}")
            cap.save(args.capture)
            robot.graceful_shutdown()
            return

        # Gravity compensation mode: constant torque, no PD.
        grav_tau = getattr(args, "gravity_comp", None)
        if grav_tau is not None:
            obs_dur = 5.0
            obs_steps = int(obs_dur * cmd_hz)
            print(f"[diag] GRAVITY COMP ({obs_dur}s, shoulder tau={grav_tau:.2f} Nm, kp=0 kd=0)...")
            tau_vec = np.zeros(N_DOF)
            tau_vec[SHOULDER_IDX] = grav_tau
            gc_cmd = RobotCommand(
                joint_positions=np.zeros(N_DOF),
                joint_velocities=np.zeros(N_DOF),
                joint_torques=tau_vec,
                kp=np.zeros(N_DOF),
                kd=np.zeros(N_DOF),
            )
            for step_i in range(obs_steps):
                step_start = time.perf_counter()
                t = (step_i + 1) * cmd_dt
                state = robot.get_state()
                robot.send_command(gc_cmd)
                robot.step()
                cap.record(t, gc_cmd, state)
                remaining = cmd_dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                if (step_i + 1) % cmd_hz == 0:
                    pos = state.joint_positions[SHOULDER_IDX]
                    vel = state.joint_velocities[SHOULDER_IDX]
                    print(f"  t={t:.1f}s  pos={pos:+.4f}  dq={vel:+.4f}")
            cap.save(args.capture)
            robot.graceful_shutdown()
            return

        # Settle (sim only)
        phase_settle(robot, band, torso_id, is_sim, duration=SETTLE_DURATION)

        state = robot.get_state()
        init_q = state.joint_positions.copy()

        # Gain ramp at current position
        print(f"[diag] Gain ramp ({GAIN_RAMP_DURATION}s)...")
        ramp_steps = int(GAIN_RAMP_DURATION * cmd_hz)
        orig_kp, orig_kd = kp.copy(), kd.copy()
        for step_i in range(ramp_steps):
            step_start = time.perf_counter()
            t = (step_i + 1) * cmd_dt
            alpha = smooth_alpha(t, GAIN_RAMP_DURATION)
            kp[:] = alpha * orig_kp
            kd[:] = alpha * orig_kd
            state = robot.get_state()
            cmd = make_cmd(init_q, np.zeros(N_DOF), state)
            robot.send_command(cmd)
            robot.step()
            cap.record(t, cmd, state)
            remaining = cmd_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        # Restore full gains after ramp.
        kp[:] = orig_kp
        kd[:] = orig_kd

        # Nudge shoulder to safe start
        safe_q = init_q.copy()
        lo, hi = JOINT_LIMITS_29DOF[SHOULDER_JOINT]
        safe_q[SHOULDER_IDX] = np.clip(
            init_q[SHOULDER_IDX] + SHOULDER_CLEAR_OFFSET, lo, hi,
        )

        print(f"[diag] Moving to safe start (1s)...")
        move_steps = int(1.0 * cmd_hz)
        for step_i in range(move_steps):
            step_start = time.perf_counter()
            t_total = GAIN_RAMP_DURATION + (step_i + 1) * cmd_dt
            alpha = (step_i + 1) / move_steps
            target_q = init_q + alpha * (safe_q - init_q)
            state = robot.get_state()
            cmd = make_cmd(target_q, np.zeros(N_DOF), state)
            robot.send_command(cmd)
            robot.step()
            cap.record(t_total, cmd, state)
            remaining = cmd_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        # Hold-only mode: just hold the safe position for 5s.
        hold_only = getattr(args, "hold_only", False)
        if hold_only:
            hold_dur = 5.0
            hold_steps = int(hold_dur * cmd_hz)
            print(f"[diag] Hold-only test ({hold_dur}s at safe start)...")
            for step_i in range(hold_steps):
                step_start = time.perf_counter()
                t = GAIN_RAMP_DURATION + 1.0 + (step_i + 1) * cmd_dt
                state = robot.get_state()
                cmd = make_cmd(safe_q, np.zeros(N_DOF), state)
                robot.send_command(cmd)
                robot.step()
                cap.record(t, cmd, state)
                remaining = cmd_dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                if (step_i + 1) % cmd_hz == 0:
                    err = abs(state.joint_positions[SHOULDER_IDX] - safe_q[SHOULDER_IDX])
                    max_vel = np.max(np.abs(state.joint_velocities))
                    print(f"  t={t:.1f}s  pos_err={err:.4f}  max|dq|={max_vel:.4f}")

        # Sinusoid (skipped in hold-only mode)
        center_q = safe_q.copy()
        amp = _build_shoulder_amplitude(center_q) if not hold_only else 0.0
        if not hold_only:
            print(f"[diag] Shoulder sinusoid ({SINUSOID_DURATION}s, amp={amp:.3f} rad)...")

        sin_steps = int(SINUSOID_DURATION * cmd_hz) if not hold_only else 0
        for step_i in range(sin_steps):
            step_start = time.perf_counter()
            t = (step_i + 1) * cmd_dt
            t_total = GAIN_RAMP_DURATION + 1.0 + t

            target_q = center_q.copy()
            cos_val = math.cos(2.0 * math.pi * SINUSOID_FREQ_HZ * t)
            wave = (cos_val - 1.0) / 2.0
            target_q[SHOULDER_IDX] = center_q[SHOULDER_IDX] + amp * wave

            target_dq = np.zeros(N_DOF)
            omega = 2.0 * math.pi * SINUSOID_FREQ_HZ
            sin_val = math.sin(2.0 * math.pi * SINUSOID_FREQ_HZ * t)
            target_dq[SHOULDER_IDX] = -amp * omega * sin_val / 2.0

            state = robot.get_state()
            cmd = make_cmd(target_q, target_dq, state)
            robot.send_command(cmd)
            robot.step()
            cap.record(t_total, cmd, state)

            remaining = cmd_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

            if (step_i + 1) % cmd_hz == 0:
                err = abs(state.joint_positions[SHOULDER_IDX] - target_q[SHOULDER_IDX])
                actual_hz = 1.0 / cap.wall_times[-1] if len(cap.wall_times) > 1 else 0
                print(f"  t={t:.1f}s  err={err:.4f}  actual={state.joint_positions[SHOULDER_IDX]:+.3f}")

    except KeyboardInterrupt:
        print("\n[diag] Ctrl+C")
    finally:
        robot.graceful_shutdown()

    cap.save(args.capture)


# ---------------------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------------------

def plot_comparison(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = {}
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        label = Path(p).stem
        datasets[label] = d
        n = len(d["timestamps"])
        print(f"[plot] {label}: {n} steps")

    si = int(datasets[list(datasets.keys())[0]]["shoulder_idx"])

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=False,
                              gridspec_kw={"hspace": 0.3})
    colors = {"sim": "#1976D2", "real": "#E65100"}

    def get_color(label):
        for k, c in colors.items():
            if k in label.lower():
                return c
        return "#333333"

    # Panel 1: Shoulder position — commanded vs actual
    ax = axes[0]
    for label, d in datasets.items():
        ts = d["timestamps"]
        c = get_color(label)
        ax.plot(ts, d["cmd_pos"][:, si], color=c, ls="--", lw=0.8, alpha=0.6, label=f"{label} cmd")
        ax.plot(ts, d["actual_pos"][:, si], color=c, lw=1.5, label=f"{label} actual")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Shoulder Roll: Commanded vs Actual Position")
    ax.legend(fontsize=8, ncol=2)

    # Panel 2: Tracking error
    ax = axes[1]
    for label, d in datasets.items():
        ts = d["timestamps"]
        err = np.abs(d["actual_pos"][:, si] - d["cmd_pos"][:, si])
        c = get_color(label)
        ax.plot(ts, err, color=c, lw=1.2, label=f"{label} |err|")
    ax.set_ylabel("Position Error (rad)")
    ax.set_title("Shoulder Roll: Tracking Error")
    ax.legend(fontsize=8)

    # Panel 3: Shoulder velocity — commanded vs actual
    ax = axes[2]
    for label, d in datasets.items():
        ts = d["timestamps"]
        c = get_color(label)
        ax.plot(ts, d["cmd_vel"][:, si], color=c, ls="--", lw=0.8, alpha=0.6, label=f"{label} cmd_dq")
        ax.plot(ts, d["actual_vel"][:, si], color=c, lw=1.2, label=f"{label} actual_dq")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Shoulder Roll: Velocity")
    ax.legend(fontsize=8, ncol=2)

    # Panel 4: Torques
    ax = axes[3]
    for label, d in datasets.items():
        ts = d["timestamps"]
        c = get_color(label)
        ax.plot(ts, d["actual_torques"][:, si], color=c, lw=1.2, label=f"{label} torque")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Shoulder Roll: Measured Torque")
    ax.legend(fontsize=8)

    # Panel 5: Control loop timing
    ax = axes[4]
    for label, d in datasets.items():
        wall_dt = d["wall_dt"]
        if len(wall_dt) > 1:
            c = get_color(label)
            ax.plot(d["timestamps"][1:], wall_dt[1:] * 1000, color=c, lw=0.8,
                    alpha=0.7, label=f"{label} dt")
            ax.axhline(1000 / 50, color=c, ls="--", lw=0.5, alpha=0.4)
            mean_dt = np.mean(wall_dt[1:]) * 1000
            jitter = np.std(wall_dt[1:]) * 1000
            print(f"[plot] {label}: mean dt={mean_dt:.1f}ms, jitter(std)={jitter:.2f}ms, "
                  f"actual hz={1000/mean_dt:.1f}")
    ax.set_ylabel("Step dt (ms)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Control Loop Timing (target: 20ms)")
    ax.legend(fontsize=8)

    # Summary stats
    for label, d in datasets.items():
        err = np.abs(d["actual_pos"][:, si] - d["cmd_pos"][:, si])
        vel_err = np.abs(d["actual_vel"][:, si] - d["cmd_vel"][:, si])
        print(f"\n[{label}] Shoulder tracking:")
        print(f"  Position error: mean={np.mean(err):.4f}, max={np.max(err):.4f} rad")
        print(f"  Velocity error: mean={np.mean(vel_err):.4f}, max={np.max(vel_err):.4f} rad/s")
        print(f"  Torque range: [{np.min(d['actual_torques'][:, si]):.2f}, "
              f"{np.max(d['actual_torques'][:, si]):.2f}] Nm")
        # Gains used
        print(f"  Gains: kp={d['cmd_kp'][len(d['cmd_kp'])//2, si]:.1f}, "
              f"kd={d['cmd_kd'][len(d['cmd_kd'])//2, si]:.1f}")

    out = "arm_reach_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[plot] Saved to {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose arm reach sim vs real")
    subparsers = parser.add_subparsers(dest="mode")

    sim_p = subparsers.add_parser("sim")
    sim_p.add_argument("--config", default="configs/sim.yaml")
    sim_p.add_argument("--capture", required=True, help="Output NPZ path")
    sim_p.add_argument("--gains", choices=["isaaclab", "standby", "unitree"],
                        default="isaaclab", help="Gain set (default: isaaclab)")
    sim_p.add_argument("--kp-scale", type=float, default=1.0,
                        help="Scale kp by this factor (default: 1.0)")
    sim_p.add_argument("--kd-scale", type=float, default=1.0,
                        help="Scale kd by this factor (default: 1.0)")
    sim_p.add_argument("--torque-mode", action="store_true",
                        help="Compute PD torque in Python, bypass motor PD")
    sim_p.add_argument("--hz", type=int, default=50,
                        help="Command rate in Hz (default: 50)")
    sim_p.add_argument("--hold-only", action="store_true",
                        help="Skip sinusoid, just hold position for 5s (stability test)")

    real_p = subparsers.add_parser("real")
    real_p.add_argument("--config", default="configs/sim.yaml")
    real_p.add_argument("--interface", default="en8")
    real_p.add_argument("--backend", choices=["python", "cpp"], default="python")
    real_p.add_argument("--capture", required=True, help="Output NPZ path")
    real_p.add_argument("--gains", choices=["isaaclab", "standby", "unitree"],
                        default="standby", help="Gain set (default: standby)")
    real_p.add_argument("--kp-scale", type=float, default=1.0,
                        help="Scale kp by this factor (default: 1.0)")
    real_p.add_argument("--kd-scale", type=float, default=1.0,
                        help="Scale kd by this factor (default: 1.0)")
    real_p.add_argument("--torque-mode", action="store_true",
                        help="Compute PD torque in Python, bypass motor PD (DANGEROUS)")
    real_p.add_argument("--hz", type=int, default=50,
                        help="Command rate in Hz (default: 50)")
    real_p.add_argument("--hold-only", action="store_true",
                        help="Skip sinusoid, just hold position for 5s (stability test)")
    real_p.add_argument("--passive", action="store_true",
                        help="kp=kd=tau=0 (limp motors, observe only). Safe on gantry.")
    real_p.add_argument("--gravity-comp", type=float, default=None, metavar="TAU",
                        help="Constant feedforward torque on shoulder, kp=kd=0 (e.g. -1.5)")

    parser.add_argument("--plot", nargs="+", metavar="NPZ",
                        help="Plot comparison from captured NPZ files")

    args = parser.parse_args()

    if args.plot:
        plot_comparison(args.plot)
    elif args.mode:
        run_capture(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
