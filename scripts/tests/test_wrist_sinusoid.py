#!/usr/bin/env python3
"""Wrist sinusoid sim-to-real test.

Puts the robot on the gantry, settles, interpolates to home, holds for 5s,
then sweeps wrist pitch joints through a slow sinusoid at +/-quarter ROM.

Usage:
    python scripts/tests/test_wrist_sinusoid.py sim --viser
    mjpython scripts/tests/test_wrist_sinusoid.py sim --gui
    python scripts/tests/test_wrist_sinusoid.py real --interface en8
    python scripts/tests/test_wrist_sinusoid.py real --interface en8 --gui
"""
from __future__ import annotations

import math
import time

import numpy as np

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    JOINT_LIMITS_29DOF,
    load_config,
)
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
POLICY_HZ = 50
INTERP_DURATION = 5.0
GAIN_RAMP_DURATION = 0.5
HOME_HOLD_DURATION = 5.0
SINUSOID_DURATION = 30.0
SINUSOID_FREQ_HZ = 0.2
N_DOF = 29

WRIST_JOINTS = ["left_wrist_pitch", "right_wrist_pitch"]
WRIST_INDICES = [G1_29DOF_JOINTS.index(j) for j in WRIST_JOINTS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_wrist_amplitudes() -> np.ndarray:
    home_q = build_home_positions()
    amps = np.zeros(N_DOF, dtype=np.float64)
    for idx in WRIST_INDICES:
        name = G1_29DOF_JOINTS[idx]
        lo, hi = JOINT_LIMITS_29DOF[name]
        home = home_q[idx]
        quarter_rom = (hi - lo) / 8.0
        max_above = hi - home
        max_below = home - lo
        amps[idx] = min(quarter_rom, max_above, max_below)
    return amps


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------

def phase_interpolate_to_home(robot, band, torso_id, is_sim, sync, realtime,
                              check_estop=lambda: False):
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

    print(f"\n[phase1] Interpolating to home over {INTERP_DURATION}s [mink]...")

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
            print(f"  t={t:.1f}s  gain={gain_alpha:.3f}  max|err|={pos_err:.4f}")

    state = robot.get_state()
    print(f"[phase1] Done. Max|err|="
          f"{np.max(np.abs(state.joint_positions - home_q)):.4f} rad")


def phase_hold(robot, sync, realtime, check_estop=lambda: False):
    n = robot.n_dof
    total_steps = int(HOME_HOLD_DURATION * POLICY_HZ)
    dt = 1.0 / POLICY_HZ
    kp, kd = build_gain_arrays("isaaclab")
    home_q = build_home_positions()

    print(f"\n[phase2] Hold at home for {HOME_HOLD_DURATION}s...")

    for step_i in range(total_steps):
        if check_estop():
            print("[phase2] E-STOP. Aborting.")
            return
        t = (step_i + 1) * dt
        cmd = RobotCommand(
            joint_positions=home_q,
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=kp,
            kd=kd,
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
            print(f"  t={t:.1f}s  max|dq|={max_vel:.4f}  max|err|={pos_err:.4f}")

    print("[phase2] Done.")


def phase_wrist_sinusoid(robot, sync, realtime, check_estop=lambda: False):
    n = robot.n_dof
    total_steps = int(SINUSOID_DURATION * POLICY_HZ)
    dt = 1.0 / POLICY_HZ
    kp, kd = build_gain_arrays("isaaclab")
    home_q = build_home_positions()
    amps = _build_wrist_amplitudes()

    wrist_names_short = [j.replace("left_", "L_").replace("right_", "R_") for j in WRIST_JOINTS]

    print(f"\n[phase3] Wrist sinusoid test ({SINUSOID_DURATION}s, "
          f"freq={SINUSOID_FREQ_HZ} Hz)...")
    print(f"  Joints: {WRIST_JOINTS}")
    print(f"  Amplitudes (rad): "
          f"{[f'{amps[i]:.3f}' for i in WRIST_INDICES]}")

    max_tracking_err = np.zeros(len(WRIST_INDICES))

    for step_i in range(total_steps):
        if check_estop():
            print("[phase3] E-STOP. Aborting.")
            break
        t = (step_i + 1) * dt

        target_q = home_q.copy()
        sin_val = math.sin(2.0 * math.pi * SINUSOID_FREQ_HZ * t)
        for idx in WRIST_INDICES:
            target_q[idx] = home_q[idx] + amps[idx] * sin_val

        target_dq = np.zeros(n)
        cos_val = math.cos(2.0 * math.pi * SINUSOID_FREQ_HZ * t)
        omega = 2.0 * math.pi * SINUSOID_FREQ_HZ
        for idx in WRIST_INDICES:
            target_dq[idx] = amps[idx] * omega * cos_val

        cmd = RobotCommand(
            joint_positions=target_q,
            joint_velocities=target_dq,
            joint_torques=np.zeros(n),
            kp=kp,
            kd=kd,
        )
        robot.send_command(cmd)
        robot.step()
        sync()
        if realtime:
            time.sleep(dt)

        state = robot.get_state()
        for wi, idx in enumerate(WRIST_INDICES):
            err = abs(state.joint_positions[idx] - target_q[idx])
            if err > max_tracking_err[wi]:
                max_tracking_err[wi] = err

        if (step_i + 1) % POLICY_HZ == 0:
            actual_wrist = [state.joint_positions[i] for i in WRIST_INDICES]
            cmd_wrist = [target_q[i] for i in WRIST_INDICES]
            errs = [abs(a - c) for a, c in zip(actual_wrist, cmd_wrist)]
            err_strs = "  ".join(
                f"{wrist_names_short[i]}={errs[i]:.4f}"
                for i in range(len(WRIST_INDICES))
            )
            print(f"  t={t:.1f}s  sin={sin_val:+.3f}  tracking err: {err_strs}")

    print(f"\n[phase3] Done. Peak tracking errors:")
    for wi, idx in enumerate(WRIST_INDICES):
        print(f"  {WRIST_JOINTS[wi]:25s}  amp={amps[idx]:.3f} rad  "
              f"peak_err={max_tracking_err[wi]:.4f} rad  "
              f"({max_tracking_err[wi]/amps[idx]*100:.1f}% of amplitude)")

    return max_tracking_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_test(args) -> dict:
    mode = args.mode
    config = load_config(args.config)
    is_sim = (mode == "sim")
    robot = create_robot(
        mode, config,
        interface=args.interface,
        backend=args.backend,
    )

    home_q = build_home_positions()

    band = ElasticBand()
    torso_id = get_torso_body_id(robot.mj_model) if is_sim else 0

    if is_sim:
        enable_gantry(robot)
        setup_gantry_band(robot, band, torso_id)

    # Real mode: mirror robot state into MuJoCo viewer.
    mirror = None
    if not is_sim and args.gui:
        from unitree_launcher.mirror import RealtimeMirror
        mirror = RealtimeMirror(base_height=1.5)

    print(f"{'='*60}")
    print(f"  WRIST SINUSOID TEST — {mode.upper()} mode")
    print(f"  Robot: {config.robot.variant} ({N_DOF} DOF)")
    print(f"  Sinusoid: {SINUSOID_FREQ_HZ} Hz, {SINUSOID_DURATION}s")
    print(f"{'='*60}")

    robot.connect()
    print(f"\n[connect] Connected.")

    results = {}

    with ScriptContext(
        robot, config, is_sim=is_sim,
        gui=args.gui,
        viser=args.viser,
        port=args.port,
        gamepad=args.gamepad,
        record_path=args.record,
        mirror=mirror,
    ) as ctx:
        try:
            state = robot.get_state()
            print(f"[init] IMU quaternion: {state.imu_quaternion.round(3)}")

            phase_settle(
                robot, band, torso_id, is_sim,
                ctx.sync, ctx.realtime, ctx.check_estop,
            )
            phase_interpolate_to_home(
                robot, band, torso_id, is_sim,
                ctx.sync, ctx.realtime, ctx.check_estop,
            )
            phase_hold(robot, ctx.sync, ctx.realtime, ctx.check_estop)

            state = robot.get_state()
            results["home_position_error"] = float(
                np.max(np.abs(state.joint_positions - home_q))
            )

            peak_errs = phase_wrist_sinusoid(
                robot, ctx.sync, ctx.realtime, ctx.check_estop,
            )
            results["wrist_peak_errors"] = {
                WRIST_JOINTS[i]: float(peak_errs[i]) for i in range(len(WRIST_JOINTS))
            }
            results["wrist_max_peak_error"] = float(np.max(peak_errs))

        except KeyboardInterrupt:
            print("\n[ABORT] Ctrl+C received.")

    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    if "home_position_error" in results:
        err = results["home_position_error"]
        print(f"  Home hold error:        {err:.4f} rad")
    if "wrist_peak_errors" in results:
        print(f"  Wrist peak tracking errors:")
        for name, err in results["wrist_peak_errors"].items():
            short = name.replace("left_", "L_").replace("right_", "R_")
            print(f"    {short:20s} {err:.4f} rad")
        max_err = results["wrist_max_peak_error"]
        ok = max_err < 0.35
        print(f"  Worst wrist error:      {max_err:.4f} rad  "
              f"{'PASS' if ok else 'FAIL'}  (threshold: 0.35 rad)")
    print(f"{'='*60}\n")

    return results


def main():
    parser = build_script_parser("Wrist sinusoid sim-to-real test")
    parser.set_defaults(mode="sim")
    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
