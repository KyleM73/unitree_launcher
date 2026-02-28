#!/usr/bin/env python3
"""Right shoulder sinusoid test.

Reads current joint positions, uses mink to plan a short collision-free
move to a safe starting pose (shoulder rolled out a few degrees), then
sweeps right_shoulder_roll through a slow sinusoid at 1/8 ROM.

All other joints hold at their initial (read from robot) positions.

Phases:
    0. Settle (sim only) — gravity equilibrium.
    1. Gain ramp (0.5s) — smoothly engage gains at current position.
    2. Move to safe start (2s) — mink trajectory, shoulder out ~5 deg.
    3. Hold (2s) — verify stable.
    4. Shoulder sinusoid (30s) — sweep right_shoulder_roll.

Usage:
    mjpython scripts/right_arm_reach.py sim --viewer
    python scripts/right_arm_reach.py real --interface en8
"""
from __future__ import annotations

import argparse
import math
import time

import mujoco
import numpy as np

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    JOINT_LIMITS_29DOF,
    load_config,
)
from unitree_launcher.gantry import (
    ElasticBand,
    apply_band,
    build_gain_arrays,
    build_home_positions,
    enable_gantry,
    get_torso_body_id,
    setup_gantry_band,
    smooth_alpha,
)
from unitree_launcher.robot.base import RobotCommand
from unitree_launcher.trajectory import plan_trajectory, resample_trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLICY_HZ = 50
DT = 1.0 / POLICY_HZ
SETTLE_DURATION = 5.0
GAIN_RAMP_DURATION = 0.5
MOVE_DURATION = 2.0          # seconds to move to safe start
HOLD_DURATION = 2.0
SINUSOID_DURATION = 30.0
SINUSOID_FREQ_HZ = 0.2       # 5-second period
N_DOF = 29

SHOULDER_JOINT = "right_shoulder_roll"
SHOULDER_IDX = G1_29DOF_JOINTS.index(SHOULDER_JOINT)

# How far to nudge the shoulder outward for a collision-free start (rad)
SHOULDER_CLEAR_OFFSET = -0.30  # ~17 deg outward — clears hand from hip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_shoulder_amplitude(init_q: np.ndarray) -> float:
    """1/8 ROM amplitude, clamped to stay within limits from init pos."""
    lo, hi = JOINT_LIMITS_29DOF[SHOULDER_JOINT]
    home = init_q[SHOULDER_IDX]
    eighth_rom = (hi - lo) / 4.0
    max_above = hi - home
    max_below = home - lo
    return min(eighth_rom, max_above, max_below)


def _build_safe_start(init_q: np.ndarray) -> np.ndarray:
    """Nudge shoulder outward a few degrees for collision clearance."""
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

def phase_settle(robot, band, torso_id, is_sim, sync, realtime):
    """Sim only: find gravity equilibrium."""
    if not is_sim:
        return

    model = robot.mj_model
    data = robot.mj_data
    dt = model.opt.timestep
    n_iters = int(SETTLE_DURATION / dt)

    print(f"[settle] Finding gravity equilibrium ({SETTLE_DURATION}s)...")

    orig_gainprm = model.actuator_gainprm.copy()
    orig_biasprm = model.actuator_biasprm.copy()
    orig_dof_damping = model.dof_damping.copy()
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    model.dof_damping[6:] = 5.0

    report_interval = int(1.0 / dt)
    for i in range(n_iters):
        apply_band(data, band, torso_id)
        mujoco.mj_step(model, data)
        if realtime:
            sync()
            time.sleep(dt)
        elif (i + 1) % report_interval == 0:
            sync()
        if (i + 1) % report_interval == 0:
            max_vel = np.max(np.abs(data.qvel[6:]))
            print(f"  t={((i+1)*dt):.1f}s  max|dq|={max_vel:.4f}")

    model.actuator_gainprm[:] = orig_gainprm
    model.actuator_biasprm[:] = orig_biasprm
    model.dof_damping[:] = orig_dof_damping
    print("[settle] Done.")
    sync()


def phase_gain_ramp(robot, init_q, kp, kd, sync, realtime):
    """Ramp gains from 0 to full while holding current position."""
    total_steps = int(GAIN_RAMP_DURATION * POLICY_HZ)
    print(f"\n[gain_ramp] Ramping gains over {GAIN_RAMP_DURATION}s...")

    for step_i in range(total_steps):
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


def phase_move_to_safe(robot, waypoints, kp, kd, sync, realtime):
    """Follow mink trajectory to safe start position."""
    total_steps = len(waypoints)
    print(f"\n[move] Moving to safe start ({total_steps} steps) [mink]...")

    for step_i in range(total_steps):
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


def phase_hold(robot, q, kp, kd, sync, realtime, duration=HOLD_DURATION):
    """Hold at a fixed position."""
    total_steps = int(duration * POLICY_HZ)
    print(f"\n[hold] Holding for {duration}s...")

    for step_i in range(total_steps):
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


def phase_shoulder_sinusoid(robot, center_q, kp, kd, sync, realtime):
    """Sweep right_shoulder_roll outward from center_q and back.

    Uses a one-sided waveform: ``center + amp * (cos(t) - 1) / 2``
    which ranges from ``center`` (inward limit, never past center)
    to ``center - amp`` (full outward extension).  This guarantees
    the arm never swings inward past its safe starting position.
    """
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
        step_start = time.perf_counter()
        t = (step_i + 1) * DT

        target_q = center_q.copy()
        # One-sided: (cos - 1)/2 ranges from 0 (at t=0) to -1 (at half period)
        cos_val = math.cos(2.0 * math.pi * SINUSOID_FREQ_HZ * t)
        wave = (cos_val - 1.0) / 2.0  # 0 to -1
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

def run_test(
    mode: str,
    config_path: str,
    interface: str | None = None,
    viewer: bool = False,
    headless: bool = False,
) -> dict:
    from unitree_launcher.robot.sim_robot import SimRobot

    config = load_config(config_path)
    is_sim = (mode == "sim")

    if is_sim:
        config.network.domain_id = 1
        robot = SimRobot(config)
    else:
        from unitree_launcher.robot.real_robot import RealRobot
        config.network.domain_id = 0
        config.network.interface = interface
        robot = RealRobot(config)

    gain_kind = "isaaclab"
    kp, kd = build_gain_arrays(gain_kind)

    band = ElasticBand()
    torso_id = get_torso_body_id(robot.mj_model) if is_sim else 0

    if is_sim:
        enable_gantry(robot)
        setup_gantry_band(robot, band, torso_id)

    _viewer = None
    show_viewer = not headless and (viewer or not is_sim)
    if show_viewer:
        import mujoco.viewer
        if is_sim:
            _viewer = mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data)

    realtime = show_viewer or not is_sim

    def _sync():
        if _viewer is not None:
            _viewer.sync()

    print(f"{'='*60}")
    print(f"  RIGHT SHOULDER SINUSOID TEST — {mode.upper()} mode")
    print(f"  Gains: {gain_kind}")
    print(f"  Sinusoid: {SINUSOID_FREQ_HZ} Hz, {SINUSOID_DURATION}s")
    print(f"{'='*60}")

    robot.connect()
    print(f"\n[connect] Connected.")

    results = {}

    try:
        state = robot.get_state()
        init_q = state.joint_positions.copy()
        print(f"[init] shoulder_roll: {init_q[SHOULDER_IDX]:+.4f} rad")

        # Settle (sim only)
        phase_settle(robot, band, torso_id, is_sim, _sync, realtime)

        # Re-read after settle
        state = robot.get_state()
        init_q = state.joint_positions.copy()

        # Plan short move to safe start (shoulder out a few degrees)
        safe_q = _build_safe_start(init_q)
        print(f"\n[plan] Safe start: shoulder_roll "
              f"{init_q[SHOULDER_IDX]:+.3f} -> {safe_q[SHOULDER_IDX]:+.3f}")
        safe_plan = plan_trajectory(init_q, safe_q)
        safe_waypoints = resample_trajectory(safe_plan, MOVE_DURATION, DT)
        print(f"  Converged: {safe_plan.converged}  Waypoints: {len(safe_waypoints)}")

        # Execute
        phase_gain_ramp(robot, init_q, kp, kd, _sync, realtime)
        phase_move_to_safe(robot, safe_waypoints, kp, kd, _sync, realtime)
        phase_hold(robot, safe_q, kp, kd, _sync, realtime)

        peak_err = phase_shoulder_sinusoid(
            robot, safe_q, kp, kd, _sync, realtime,
        )
        results["shoulder_peak_error"] = float(peak_err)

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl+C received.")
    finally:
        print(f"\n[shutdown] Disconnecting...")
        robot.graceful_shutdown()
        if _viewer is not None:
            _viewer.close()
        print("[shutdown] Done.")

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
    parser = argparse.ArgumentParser(
        description="Right shoulder sinusoid test"
    )
    subparsers = parser.add_subparsers(dest="mode")

    sim_parser = subparsers.add_parser("sim", help="Simulation mode (default)")
    sim_parser.add_argument("--config", default="configs/default.yaml")
    sim_parser.add_argument("--viewer", action="store_true")

    real_parser = subparsers.add_parser("real", help="Real robot mode")
    real_parser.add_argument("--config", default="configs/default.yaml")
    real_parser.add_argument("--interface", default="en8")
    real_parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()
    mode = args.mode or "sim"

    run_test(
        mode=mode,
        config_path=getattr(args, "config", "configs/default.yaml"),
        interface=getattr(args, "interface", None),
        viewer=getattr(args, "viewer", False),
        headless=getattr(args, "headless", False),
    )


if __name__ == "__main__":
    main()
