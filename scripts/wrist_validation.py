#!/usr/bin/env python3
"""Wrist sinusoid sim-to-real test.

Puts the robot on the gantry, settles, interpolates to home, holds for 5s,
then sweeps wrist pitch joints through a slow sinusoid at ±quarter ROM.
Logs commanded vs actual positions so you can compare sim and real tracking.

In real mode the MuJoCo viewer mirrors the live robot state (unless --headless).

Phases:
    0. Settle (sim only) — hang with zero torques, gravity equilibrium.
    1. Interpolate to home (5s) — positions AND gains ramp together.
    2. Hold at home (5s) — verify stable position hold.
    3. Wrist sinusoid (30s) — wrist pitch joints track sin wave, rest hold home.

Sim vs real:
    The control loop (phases 1-3) is identical for both backends — the same
    ``send_command()`` / ``step()`` / ``sync()`` pattern runs unchanged.
    The differences are confined to setup:

    - **Sim only**: ``enable_gantry()`` positions the robot at the anchor
      point, ``setup_gantry_band()`` registers an elastic band substep
      callback, and ``phase_settle()`` runs raw physics to find gravity
      equilibrium.  ``step()`` advances MuJoCo physics with N substeps.
    - **Real only**: ``step()`` is a no-op (hardware PD runs at ~500 Hz).
      ``graceful_shutdown()`` sends damping commands before disconnecting.
      A ``RealtimeMirror`` mirrors live state into a MuJoCo viewer.

Usage:
    # Sim (default) — headless
    mjpython scripts/wrist_validation.py

    # Sim with viewer
    mjpython scripts/wrist_validation.py sim --viewer

    # Real robot with live MuJoCo mirror
    mjpython scripts/wrist_validation.py real --interface en8

    # Real robot headless (no viewer)
    python scripts/wrist_validation.py real --interface en8 --headless
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
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.gantry import (
    ElasticBand,
    apply_band,
    build_gain_arrays,
    build_home_positions,
    enable_gantry,
    get_torso_body_id,
    plan_interpolation_trajectory,
    setup_gantry_band,
    smooth_alpha,
)
from unitree_launcher.mirror import RealtimeMirror
from unitree_launcher.robot.base import RobotCommand

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLICY_HZ = 50
SETTLE_DURATION = 5.0
INTERP_DURATION = 5.0
GAIN_RAMP_DURATION = 0.5  # Ramp gains to full over this many seconds
HOME_HOLD_DURATION = 5.0
SINUSOID_DURATION = 30.0
SINUSOID_FREQ_HZ = 0.2       # 5-second period
N_DOF = 29

# Wrist pitch joints — flex the hand up/down (like elbow motion)
WRIST_JOINTS = [
    "left_wrist_pitch",
    "right_wrist_pitch",
]

# Map wrist names to their config-order indices
WRIST_INDICES = [G1_29DOF_JOINTS.index(j) for j in WRIST_JOINTS]


# ---------------------------------------------------------------------------
# Compute wrist sinusoid amplitudes
# ---------------------------------------------------------------------------

def _build_wrist_amplitudes() -> np.ndarray:
    """Quarter-ROM amplitude for each wrist joint, clamped to stay within limits."""
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

def phase_settle(robot, band, torso_id, is_sim, sync, realtime, check_estop=None):
    """Sim only: find gravity equilibrium."""
    if not is_sim:
        return

    model = robot.mj_model
    data = robot.mj_data
    dt = model.opt.timestep
    n_iters = int(SETTLE_DURATION / dt)

    print(f"[settle] Finding gravity equilibrium ({SETTLE_DURATION}s, "
          f"{n_iters} physics steps)...")

    orig_gainprm = model.actuator_gainprm.copy()
    orig_biasprm = model.actuator_biasprm.copy()
    orig_dof_damping = model.dof_damping.copy()

    # Zero all gains so the robot finds gravity equilibrium.  This matches
    # what the real robot looks like when hanging unpowered on the gantry.
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    model.dof_damping[6:] = 5.0

    report_interval = int(1.0 / dt)
    for i in range(n_iters):
        if check_estop and check_estop():
            print("[settle] E-STOP. Aborting.")
            return
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


def phase_interpolate_to_home(robot, band, torso_id, is_sim, sync, realtime, check_estop=None):
    """Smoothly interpolate from current position to home."""
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
        if check_estop and check_estop():
            print("[phase1] E-STOP. Aborting.")
            return
        t = (step_i + 1) * dt
        # Gains ramp quickly (0.5s) so actuators engage before gravity
        # pulls joints away.  The mink trajectory keeps position targets
        # close to the current configuration, so fast gains are safe.
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


def phase_hold(robot, band, torso_id, is_sim, sync, realtime, check_estop=None):
    """Hold at home for HOME_HOLD_DURATION seconds."""
    n = robot.n_dof
    total_steps = int(HOME_HOLD_DURATION * POLICY_HZ)
    dt = 1.0 / POLICY_HZ
    kp, kd = build_gain_arrays("isaaclab")
    home_q = build_home_positions()

    print(f"\n[phase2] Hold at home for {HOME_HOLD_DURATION}s...")

    for step_i in range(total_steps):
        if check_estop and check_estop():
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


def phase_wrist_sinusoid(robot, band, torso_id, is_sim, sync, realtime, check_estop=None):
    """Sweep wrist joints through a sinusoid; all other joints hold home."""
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
        if check_estop and check_estop():
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

def run_test(
    mode: str,
    config_path: str,
    interface: str | None = None,
    viewer: bool = False,
    headless: bool = False,
    record_path: str | None = None,
    gamepad: bool = False,
) -> dict:
    """Run the full wrist sinusoid test. Returns summary dict."""
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

    home_q = build_home_positions()

    band = ElasticBand()
    torso_id = get_torso_body_id(robot.mj_model) if is_sim else 0

    if is_sim:
        enable_gantry(robot)
        setup_gantry_band(robot, band, torso_id)

    # ---- Viewer setup ----
    _viewer = None
    _mirror = None
    show_viewer = not headless and (viewer or not is_sim)

    if show_viewer:
        import mujoco.viewer

        if is_sim:
            _viewer = mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data)
        else:
            _mirror = RealtimeMirror(base_height=1.5)
            _viewer = mujoco.viewer.launch_passive(_mirror.model, _mirror.data)

    realtime = show_viewer or not is_sim  # Real mode always paces at policy Hz

    # Video recorder (sim only — single-threaded loop, no lock contention).
    _recorder = None
    if record_path and is_sim:
        from unitree_launcher.recording import VideoRecorder, normalize_record_path
        _recorder = VideoRecorder(
            normalize_record_path(record_path), robot.mj_model, robot.mj_data,
        )

    def _sync():
        if _mirror is not None:
            state = robot.get_state()
            _mirror.update(state)
        if _viewer is not None:
            _viewer.sync()
        if _recorder is not None:
            _recorder.capture()

    print(f"{'='*60}")
    print(f"  WRIST SINUSOID TEST — {mode.upper()} mode")
    print(f"  Robot: {config.robot.variant} ({N_DOF} DOF)")
    print(f"  Sinusoid: {SINUSOID_FREQ_HZ} Hz, {SINUSOID_DURATION}s")
    if is_sim:
        print(f"  Elastic band: K={band.stiffness}, D={band.damping}")
    if show_viewer:
        vtype = "sim physics" if is_sim else "real robot mirror"
        print(f"  Viewer: {vtype}")
    print(f"{'='*60}")

    robot.connect()
    print(f"\n[connect] Connected.")

    # ---- Gamepad e-stop ----
    safety = SafetyController(config, n_dof=robot.n_dof)
    safety.start()  # IDLE -> RUNNING so estop() can latch

    gamepad_monitor = None
    if gamepad:
        try:
            from unitree_launcher.control.gamepad import GamepadMonitor
            gamepad_monitor = GamepadMonitor(safety)
            gamepad_monitor.start()
        except Exception as exc:
            print(f"[main] WARNING: Gamepad init failed: {exc}")

    def _check_estop():
        """Return True if e-stopped (caller should abort)."""
        return safety.state == SystemState.ESTOP

    results = {}

    try:
        state = robot.get_state()
        print(f"[init] IMU quaternion: {state.imu_quaternion.round(3)}")

        phase_settle(robot, band, torso_id, is_sim, _sync, realtime, _check_estop)
        phase_interpolate_to_home(robot, band, torso_id, is_sim, _sync, realtime, _check_estop)
        phase_hold(robot, band, torso_id, is_sim, _sync, realtime, _check_estop)

        state = robot.get_state()
        results["home_position_error"] = float(
            np.max(np.abs(state.joint_positions - home_q))
        )

        peak_errs = phase_wrist_sinusoid(
            robot, band, torso_id, is_sim, _sync, realtime, _check_estop
        )
        results["wrist_peak_errors"] = {
            WRIST_JOINTS[i]: float(peak_errs[i]) for i in range(len(WRIST_JOINTS))
        }
        results["wrist_max_peak_error"] = float(np.max(peak_errs))

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl+C received.")
    finally:
        print(f"\n[shutdown] Disconnecting...")
        if gamepad_monitor is not None:
            gamepad_monitor.stop()
        if _recorder is not None:
            _recorder.close()
        robot.graceful_shutdown()
        if _viewer is not None:
            _viewer.close()
        print("[shutdown] Done.")

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
    parser = argparse.ArgumentParser(
        description="Wrist sinusoid sim-to-real test"
    )
    subparsers = parser.add_subparsers(dest="mode")

    sim_parser = subparsers.add_parser("sim", help="Simulation mode (default)")
    sim_parser.add_argument("--config", default="configs/default.yaml")
    sim_parser.add_argument("--viewer", action="store_true",
                            help="Launch MuJoCo viewer (use mjpython)")
    sim_parser.add_argument("--record", nargs="?", const="sim.mp4", default=None,
                            metavar="PATH",
                            help="Record video to MP4 (default: recording.mp4)")
    sim_parser.add_argument("--gamepad", action="store_true",
                            help="Enable gamepad e-stop (Logitech F310)")

    real_parser = subparsers.add_parser("real", help="Real robot mode")
    real_parser.add_argument("--config", default="configs/default.yaml")
    real_parser.add_argument("--interface", default="en8",
                             help="Network interface (default: en8)")
    real_parser.add_argument("--headless", action="store_true",
                             help="Skip MuJoCo viewer for real mode")
    real_parser.add_argument("--gamepad", action="store_true",
                            help="Enable gamepad e-stop (Logitech F310)")

    args = parser.parse_args()

    mode = args.mode or "sim"

    run_test(
        mode=mode,
        config_path=getattr(args, "config", "configs/default.yaml"),
        interface=getattr(args, "interface", None),
        viewer=getattr(args, "viewer", False),
        headless=getattr(args, "headless", False),
        record_path=getattr(args, "record", None),
        gamepad=getattr(args, "gamepad", False),
    )


if __name__ == "__main__":
    main()
