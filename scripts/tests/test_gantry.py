#!/usr/bin/env python3
"""Gantry hang test: connect, interpolate to home, hold, disconnect.

Works identically in sim and real mode. The control loop (phases 1-2) uses
the uniform ``send_command()`` / ``step()`` pattern — no backend branching.

Phases:
    0. Settle (sim only) — hang with zero torques, find gravity equilibrium.
    1. Interpolate to home (5s) — positions AND gains ramp smoothly from
       zero to IsaacLab values. Torque stays smooth throughout.
    2. Hold at home (3s) — full IsaacLab gains, verify stable position hold.
    3. Graceful disconnect.

Usage:
    python scripts/tests/test_gantry.py sim
    python scripts/tests/test_gantry.py sim --gui
    python scripts/tests/test_gantry.py sim --viser
    python scripts/tests/test_gantry.py real --interface enp3s0
"""
from __future__ import annotations

import time

import numpy as np

from unitree_launcher.config import load_config
from unitree_launcher.gantry import (
    ElasticBand,
    build_gain_arrays,
    build_home_positions,
    enable_gantry,
    get_torso_body_id,
    plan_interpolation_trajectory,
    setup_gantry_band,
    smooth_alpha,
)
from unitree_launcher.robot.base import RobotCommand

from unitree_launcher.script_utils import build_script_parser, create_robot, ScriptContext, phase_settle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_HZ = 50            # Control loop frequency
INTERP_DURATION = 5.0     # Phase 1: interpolate to home (s)
GAIN_RAMP_DURATION = 0.5  # Ramp gains to full over this many seconds
HOLD_DURATION = 3.0       # Phase 2: hold at home with IsaacLab gains (s)


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------

def phase_interpolate_to_home(robot, sync=lambda: None, realtime=False, check_estop=lambda: False):
    """Phase 1: smoothly interpolate from current position to home."""
    n = robot.n_dof
    total_steps = int(INTERP_DURATION * POLICY_HZ)
    dt = 1.0 / POLICY_HZ

    home_q = build_home_positions()
    kp_final, kd_final = build_gain_arrays("isaaclab")

    state = robot.get_state()
    start_q = state.joint_positions.copy()

    waypoints = plan_interpolation_trajectory(
        start_q, home_q, duration=INTERP_DURATION, policy_hz=POLICY_HZ,
    )

    print(f"\n[phase1] Interpolating to home over {INTERP_DURATION}s "
          f"({total_steps} steps) [mink]...")
    print(f"  Start positions (first 6): {start_q[:6].round(3)}")
    print(f"  Home  positions (first 6): {home_q[:6].round(3)}")

    for step_i in range(total_steps):
        if check_estop():
            print("[phase1] E-STOP. Aborting.")
            return
        t = (step_i + 1) * dt
        gain_alpha = smooth_alpha(t, GAIN_RAMP_DURATION)
        target_q = waypoints[step_i].copy()

        cmd = RobotCommand(
            joint_positions=target_q,
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=gain_alpha * kp_final,
            kd=gain_alpha * kd_final,
        )
        robot.send_command(cmd)
        robot.step()
        sync()
        if realtime:
            time.sleep(dt)

        if (step_i + 1) % POLICY_HZ == 0:
            state = robot.get_state()
            pos_err = np.max(np.abs(state.joint_positions - target_q))
            print(f"  t={t:.1f}s  gain={gain_alpha:.3f}  max|pos_err|={pos_err:.4f}  "
                  f"max_kp={np.max(gain_alpha * kp_final):.1f}")

    state = robot.get_state()
    final_err = np.max(np.abs(state.joint_positions - home_q))
    print(f"[phase1] Done. Final max|err|={final_err:.4f} rad")


def phase_hold(robot, sync=lambda: None, realtime=False, check_estop=lambda: False):
    """Phase 2: hold at home with IsaacLab gains."""
    n = robot.n_dof
    total_steps = int(HOLD_DURATION * POLICY_HZ)
    dt = 1.0 / POLICY_HZ

    kp_hold, kd_hold = build_gain_arrays("isaaclab")
    home_q = build_home_positions()

    print(f"\n[phase2] Hold at home for {HOLD_DURATION}s "
          f"(IsaacLab gains, max_kp={np.max(kp_hold):.1f})...")

    for step_i in range(total_steps):
        if check_estop():
            print("[phase2] E-STOP. Aborting.")
            return
        t = (step_i + 1) * dt

        cmd = RobotCommand(
            joint_positions=home_q,
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=kp_hold,
            kd=kd_hold,
        )
        robot.send_command(cmd)
        robot.step()
        sync()
        if realtime:
            time.sleep(dt)

        if (step_i + 1) % POLICY_HZ == 0:
            state = robot.get_state()
            max_vel = np.max(np.abs(state.joint_velocities))
            pos_err = np.max(np.abs(state.joint_positions - home_q))
            print(f"  t={t:.1f}s  max|dq|={max_vel:.4f}  max|pos_err|={pos_err:.4f}")

    print("[phase2] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_test(args) -> dict:
    """Run the gantry hang test. Returns summary dict for assertions."""
    mode = args.mode
    config = load_config(args.config)
    is_sim = (mode == "sim")
    robot = create_robot(
        mode, config,
        interface=args.interface,
    )

    home_q = build_home_positions()
    results = {}

    band = ElasticBand()
    torso_id = get_torso_body_id(robot.mj_model) if is_sim else 0

    if is_sim:
        enable_gantry(robot)
        setup_gantry_band(robot, band, torso_id)

    print(f"{'='*60}")
    print(f"  GANTRY HANG TEST — {mode.upper()} mode")
    print(f"  Robot: {config.robot.variant} ({robot.n_dof} DOF)")
    if is_sim:
        print(f"  Elastic band: K={band.stiffness}, D={band.damping}, "
              f"anchor={band.point.tolist()}")
    print(f"{'='*60}")

    robot.connect()
    print(f"\n[connect] Connected successfully.")

    with ScriptContext(
        robot, config, is_sim=is_sim,
        gui=args.gui,
        viser=args.viser,
        port=args.port,
        gamepad=args.gamepad,
        record_path=args.record,
    ) as ctx:
        try:
            state = robot.get_state()
            print(f"\n[init] Joint positions (first 6): "
                  f"{state.joint_positions[:6].round(3)}")
            print(f"[init] IMU quaternion: {state.imu_quaternion.round(3)}")

            phase_settle(
                robot, band, torso_id, is_sim,
                ctx.sync, ctx.realtime, ctx.check_estop,
            )

            state = robot.get_state()
            results["pre_interp_positions"] = state.joint_positions.copy()

            phase_interpolate_to_home(robot, ctx.sync, ctx.realtime, ctx.check_estop)

            state = robot.get_state()
            results["post_interp_positions"] = state.joint_positions.copy()

            phase_hold(robot, ctx.sync, ctx.realtime, ctx.check_estop)

            state = robot.get_state()
            results["post_damping_velocities"] = state.joint_velocities.copy()
            results["post_damping_max_vel"] = float(
                np.max(np.abs(state.joint_velocities))
            )
            results["final_position_error"] = float(
                np.max(np.abs(state.joint_positions - home_q))
            )

        except KeyboardInterrupt:
            print("\n[ABORT] Ctrl+C received.")

    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    if "final_position_error" in results:
        err = results["final_position_error"]
        ok = err < 0.3
        print(f"  Position error (converged): {err:.4f} rad  "
              f"{'PASS' if ok else 'FAIL'}  (threshold: 0.3 rad)")
    if "post_damping_max_vel" in results:
        vel = results["post_damping_max_vel"]
        print(f"  Max velocity after hold:    {vel:.4f} rad/s")
    print(f"{'='*60}\n")

    return results


def main():
    parser = build_script_parser("Gantry hang test")
    parser.set_defaults(mode="sim")
    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
