#!/usr/bin/env python3
"""Right shoulder sinusoid test.

Reads current joint positions, uses mink to plan a short collision-free
move to a safe starting pose (shoulder rolled out a few degrees), then
sweeps right_shoulder_roll through a slow sinusoid at 1/8 ROM.

Usage:
    python scripts/tests/test_arm_reach.py sim --viser
    mjpython scripts/tests/test_arm_reach.py sim --gui
    python scripts/tests/test_arm_reach.py real --interface en8
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
    enable_gantry,
    get_torso_body_id,
    setup_gantry_band,
    smooth_alpha,
)
from unitree_launcher.robot.base import RobotCommand
from unitree_launcher.trajectory import plan_trajectory, resample_trajectory

from unitree_launcher.script_utils import build_script_parser, create_robot, ScriptContext, phase_settle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLICY_HZ = 50
DT = 1.0 / POLICY_HZ
GAIN_RAMP_DURATION = 0.5
MOVE_DURATION = 2.0          # seconds to move to safe start
HOLD_DURATION = 2.0
SINUSOID_DURATION = 30.0
SINUSOID_FREQ_HZ = 0.2       # 5-second period
N_DOF = 29

SHOULDER_JOINT = "right_shoulder_roll"
SHOULDER_IDX = G1_29DOF_JOINTS.index(SHOULDER_JOINT)
SHOULDER_CLEAR_OFFSET = -0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_shoulder_amplitude(init_q: np.ndarray) -> float:
    lo, hi = JOINT_LIMITS_29DOF[SHOULDER_JOINT]
    home = init_q[SHOULDER_IDX]
    eighth_rom = (hi - lo) / 4.0
    max_above = hi - home
    max_below = home - lo
    return min(eighth_rom, max_above, max_below)


def _build_safe_start(init_q: np.ndarray) -> np.ndarray:
    safe_q = init_q.copy()
    lo, hi = JOINT_LIMITS_29DOF[SHOULDER_JOINT]
    safe_q[SHOULDER_IDX] = np.clip(
        init_q[SHOULDER_IDX] + SHOULDER_CLEAR_OFFSET, lo, hi,
    )
    return safe_q


def _sleep_tick(step_start: float) -> None:
    remaining = DT - (time.perf_counter() - step_start)
    if remaining > 0:
        time.sleep(remaining)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_gain_ramp(robot, init_q, kp, kd, sync, realtime, check_estop=lambda: False):
    total_steps = int(GAIN_RAMP_DURATION * POLICY_HZ)
    print(f"\n[gain_ramp] Ramping gains over {GAIN_RAMP_DURATION}s...")

    for step_i in range(total_steps):
        if check_estop():
            print("[gain_ramp] E-STOP. Aborting.")
            return
        step_start = time.perf_counter()
        t = (step_i + 1) * DT
        alpha = smooth_alpha(t, GAIN_RAMP_DURATION)
        cmd = RobotCommand(
            joint_positions=init_q,
            joint_velocities=np.zeros(N_DOF),
            joint_torques=np.zeros(N_DOF),
            kp=alpha * kp,
            kd=alpha * kd,
        )
        robot.send_command(cmd)
        robot.step()
        sync()
        if realtime:
            _sleep_tick(step_start)

    print("[gain_ramp] Done.")


def phase_move_to_safe(robot, waypoints, kp, kd, sync, realtime, check_estop=lambda: False):
    total_steps = len(waypoints)
    print(f"\n[move] Moving to safe start ({total_steps} steps) [mink]...")

    for step_i in range(total_steps):
        if check_estop():
            print("[move] E-STOP. Aborting.")
            return
        step_start = time.perf_counter()
        target_q = waypoints[step_i]
        cmd = RobotCommand(
            joint_positions=target_q,
            joint_velocities=np.zeros(N_DOF),
            joint_torques=np.zeros(N_DOF),
            kp=kp,
            kd=kd,
        )
        robot.send_command(cmd)
        robot.step()
        sync()
        if realtime:
            _sleep_tick(step_start)

        if (step_i + 1) % POLICY_HZ == 0:
            state = robot.get_state()
            t = (step_i + 1) * DT
            pos_err = np.max(np.abs(state.joint_positions - target_q))
            print(f"  t={t:.1f}s  max|err|={pos_err:.4f}")

    print("[move] Done.")


def phase_hold(robot, q, kp, kd, sync, realtime, duration=HOLD_DURATION, check_estop=lambda: False):
    total_steps = int(duration * POLICY_HZ)
    print(f"\n[hold] Holding for {duration}s...")

    for step_i in range(total_steps):
        if check_estop():
            print("[hold] E-STOP. Aborting.")
            return
        step_start = time.perf_counter()
        cmd = RobotCommand(
            joint_positions=q,
            joint_velocities=np.zeros(N_DOF),
            joint_torques=np.zeros(N_DOF),
            kp=kp,
            kd=kd,
        )
        robot.send_command(cmd)
        robot.step()
        sync()
        if realtime:
            _sleep_tick(step_start)

        if (step_i + 1) % POLICY_HZ == 0:
            state = robot.get_state()
            t = (step_i + 1) * DT
            sh_err = abs(state.joint_positions[SHOULDER_IDX] - q[SHOULDER_IDX])
            print(f"  t={t:.1f}s  shoulder|err|={sh_err:.4f}")

    print("[hold] Done.")


def phase_shoulder_sinusoid(robot, center_q, kp, kd, sync, realtime, check_estop=lambda: False):
    total_steps = int(SINUSOID_DURATION * POLICY_HZ)
    amp = _build_shoulder_amplitude(center_q)

    print(f"\n[sinusoid] Shoulder sinusoid ({SINUSOID_DURATION}s, "
          f"freq={SINUSOID_FREQ_HZ} Hz)...")
    print(f"  Joint: {SHOULDER_JOINT}")
    print(f"  Center (inward limit): {center_q[SHOULDER_IDX]:+.3f} rad")
    print(f"  Outward extent:        {center_q[SHOULDER_IDX] - amp:+.3f} rad")
    print(f"  Amplitude: {amp:.3f} rad ({np.degrees(amp):.1f} deg)")

    max_tracking_err = 0.0

    for step_i in range(total_steps):
        if check_estop():
            print("[sinusoid] E-STOP. Aborting.")
            break
        step_start = time.perf_counter()
        t = (step_i + 1) * DT

        target_q = center_q.copy()
        cos_val = math.cos(2.0 * math.pi * SINUSOID_FREQ_HZ * t)
        wave = (cos_val - 1.0) / 2.0
        target_q[SHOULDER_IDX] = center_q[SHOULDER_IDX] + amp * wave

        target_dq = np.zeros(N_DOF)
        omega = 2.0 * math.pi * SINUSOID_FREQ_HZ
        sin_val = math.sin(2.0 * math.pi * SINUSOID_FREQ_HZ * t)
        target_dq[SHOULDER_IDX] = -amp * omega * sin_val / 2.0

        cmd = RobotCommand(
            joint_positions=target_q,
            joint_velocities=target_dq,
            joint_torques=np.zeros(N_DOF),
            kp=kp,
            kd=kd,
        )
        robot.send_command(cmd)
        robot.step()
        sync()
        if realtime:
            _sleep_tick(step_start)

        state = robot.get_state()
        err = abs(state.joint_positions[SHOULDER_IDX] - target_q[SHOULDER_IDX])
        if err > max_tracking_err:
            max_tracking_err = err

        if (step_i + 1) % POLICY_HZ == 0:
            actual = state.joint_positions[SHOULDER_IDX]
            cmd_val = target_q[SHOULDER_IDX]
            print(f"  t={t:.1f}s  wave={wave:+.3f}  "
                  f"cmd={cmd_val:+.3f}  actual={actual:+.3f}  "
                  f"err={abs(actual - cmd_val):.4f}")

    print(f"\n[sinusoid] Done. Peak tracking error: {max_tracking_err:.4f} rad "
          f"({np.degrees(max_tracking_err):.1f} deg)")
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

    gain_kind = "isaaclab"
    kp, kd = build_gain_arrays(gain_kind)

    band = ElasticBand()
    torso_id = get_torso_body_id(robot.mj_model) if is_sim else 0

    if is_sim:
        enable_gantry(robot)
        setup_gantry_band(robot, band, torso_id)

    print(f"{'='*60}")
    print(f"  RIGHT SHOULDER SINUSOID TEST — {mode.upper()} mode")
    print(f"  Gains: {gain_kind}")
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
    ) as ctx:
        try:
            state = robot.get_state()
            init_q = state.joint_positions.copy()
            print(f"[init] shoulder_roll: {init_q[SHOULDER_IDX]:+.4f} rad")

            phase_settle(
                robot, band, torso_id, is_sim,
                ctx.sync, ctx.realtime, ctx.check_estop,
            )

            state = robot.get_state()
            init_q = state.joint_positions.copy()

            safe_q = _build_safe_start(init_q)
            print(f"\n[plan] Safe start: shoulder_roll "
                  f"{init_q[SHOULDER_IDX]:+.3f} -> {safe_q[SHOULDER_IDX]:+.3f}")
            safe_plan = plan_trajectory(init_q, safe_q)
            safe_waypoints = resample_trajectory(safe_plan, MOVE_DURATION, DT)
            print(f"  Converged: {safe_plan.converged}  Waypoints: {len(safe_waypoints)}")

            phase_gain_ramp(robot, init_q, kp, kd, ctx.sync, ctx.realtime, ctx.check_estop)
            phase_move_to_safe(robot, safe_waypoints, kp, kd, ctx.sync, ctx.realtime, ctx.check_estop)
            phase_hold(robot, safe_q, kp, kd, ctx.sync, ctx.realtime, check_estop=ctx.check_estop)

            peak_err = phase_shoulder_sinusoid(
                robot, safe_q, kp, kd, ctx.sync, ctx.realtime, ctx.check_estop,
            )
            results["shoulder_peak_error"] = float(peak_err)

        except KeyboardInterrupt:
            print("\n[ABORT] Ctrl+C received.")

    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    if "shoulder_peak_error" in results:
        err = results["shoulder_peak_error"]
        ok = err < 0.35
        print(f"  Shoulder peak error:    {err:.4f} rad  "
              f"{'PASS' if ok else 'FAIL'}  (threshold: 0.35 rad)")
    print(f"{'='*60}\n")

    return results


def main():
    parser = build_script_parser("Right shoulder sinusoid test")
    parser.set_defaults(mode="sim")
    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
