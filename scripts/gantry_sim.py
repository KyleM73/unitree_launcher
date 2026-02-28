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

Sim vs real:
    - **Sim only**: ``enable_gantry()`` positions the robot at the anchor
      point, ``setup_gantry_band()`` registers an elastic band force as a
      per-substep callback, and ``phase_settle()`` runs raw physics to find
      gravity equilibrium.  ``step()`` advances MuJoCo by N substeps.
    - **Real only**: ``step()`` is a no-op (hardware PD runs at ~500 Hz).
      ``graceful_shutdown()`` sends damping commands before disconnecting.

Usage:
    # Sim headless (test first — always!)
    python scripts/gantry_sim.py sim

    # Sim with MuJoCo viewer
    python scripts/gantry_sim.py sim --viewer

    # Real (robot must be on gantry, powered, Ethernet connected)
    python scripts/gantry_sim.py real --interface enp3s0
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from unitree_launcher.config import load_config
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
from unitree_launcher.robot.base import RobotCommand

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_HZ = 50            # Control loop frequency
SETTLE_DURATION = 5.0     # Sim: hang time before test (s)
INTERP_DURATION = 5.0     # Phase 1: interpolate to home (s)
GAIN_RAMP_DURATION = 0.5  # Ramp gains to full over this many seconds
HOLD_DURATION = 3.0       # Phase 2: hold at home with IsaacLab gains (s)


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------

def phase_settle(
    robot, band: ElasticBand, torso_id: int,
    is_sim: bool, sync=lambda: None, realtime: bool = False,
) -> None:
    """Sim only: find gravity equilibrium for the hanging robot."""
    if not is_sim:
        return

    model = robot.mj_model
    data = robot.mj_data
    dt = model.opt.timestep
    n_iters = int(SETTLE_DURATION / dt)

    print(f"[settle] Finding gravity equilibrium ({SETTLE_DURATION}s, "
          f"{n_iters} physics steps)...")

    original_gainprm = model.actuator_gainprm.copy()
    original_biasprm = model.actuator_biasprm.copy()
    original_dof_damping = model.dof_damping.copy()

    # Zero all gains so the robot finds gravity equilibrium.  This matches
    # what the real robot looks like when hanging unpowered on the gantry.
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
            base_z = data.qpos[2]
            print(f"  t={((i+1)*dt):.1f}s  max|dq|={max_vel:.4f}  base_z={base_z:.3f}")

    model.actuator_gainprm[:] = original_gainprm
    model.actuator_biasprm[:] = original_biasprm
    model.dof_damping[:] = original_dof_damping

    state = robot.get_state()
    print(f"[settle] Done. Equilibrium (first 6): "
          f"{state.joint_positions[:6].round(4)}")
    sync()


def phase_interpolate_to_home(
    robot, band: ElasticBand, torso_id: int,
    is_sim: bool, sync=lambda: None, realtime: bool = False,
) -> None:
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
            print(f"  t={t:.1f}s  gain={gain_alpha:.3f}  max|pos_err|={pos_err:.4f}  "
                  f"max_kp={np.max(gain_alpha * kp_final):.1f}")

    state = robot.get_state()
    final_err = np.max(np.abs(state.joint_positions - home_q))
    print(f"[phase1] Done. Final max|err|={final_err:.4f} rad")


def phase_hold(
    robot, band: ElasticBand, torso_id: int,
    is_sim: bool, sync=lambda: None, realtime: bool = False,
) -> None:
    """Phase 2: hold at home with IsaacLab gains."""
    n = robot.n_dof
    total_steps = int(HOLD_DURATION * POLICY_HZ)
    dt = 1.0 / POLICY_HZ

    kp_hold, kd_hold = build_gain_arrays("isaaclab")
    home_q = build_home_positions()

    print(f"\n[phase2] Hold at home for {HOLD_DURATION}s "
          f"(IsaacLab gains, max_kp={np.max(kp_hold):.1f})...")

    for step_i in range(total_steps):
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
# Main entry
# ---------------------------------------------------------------------------

def run_test(
    mode: str,
    config_path: str,
    interface: str | None = None,
    viewer: bool = False,
    record_path: str | None = None,
) -> dict:
    """Run the gantry hang test. Returns summary dict for assertions."""
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
    results = {}

    band = ElasticBand()
    torso_id = get_torso_body_id(robot.mj_model) if is_sim else 0

    if is_sim:
        enable_gantry(robot)
        setup_gantry_band(robot, band, torso_id)

    _viewer = None
    _realtime = (viewer and is_sim) or not is_sim  # Real mode always paces at policy Hz
    if _realtime:
        import mujoco.viewer
        _viewer = mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data)

    # Video recorder (sim only — single-threaded loop, no lock contention).
    _recorder = None
    if record_path and is_sim:
        from unitree_launcher.recording import VideoRecorder, normalize_record_path
        _recorder = VideoRecorder(
            normalize_record_path(record_path), robot.mj_model, robot.mj_data,
        )

    def _sync():
        if _viewer is not None:
            _viewer.sync()
        if _recorder is not None:
            _recorder.capture()

    print(f"{'='*60}")
    print(f"  GANTRY HANG TEST — {mode.upper()} mode")
    print(f"  Robot: {config.robot.variant} ({robot.n_dof} DOF)")
    if is_sim:
        print(f"  Elastic band: K={band.stiffness}, D={band.damping}, "
              f"anchor={band.point.tolist()}")
    print(f"{'='*60}")

    robot.connect()
    print(f"\n[connect] Connected successfully.")

    try:
        state = robot.get_state()
        print(f"\n[init] Joint positions (first 6): "
              f"{state.joint_positions[:6].round(3)}")
        print(f"[init] IMU quaternion: {state.imu_quaternion.round(3)}")

        phase_settle(robot, band, torso_id, is_sim, _sync, _realtime)

        state = robot.get_state()
        results["pre_interp_positions"] = state.joint_positions.copy()

        phase_interpolate_to_home(
            robot, band, torso_id, is_sim, _sync, _realtime
        )

        state = robot.get_state()
        results["post_interp_positions"] = state.joint_positions.copy()

        phase_hold(robot, band, torso_id, is_sim, _sync, _realtime)

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
    finally:
        print(f"\n[shutdown] Disconnecting...")
        if _recorder is not None:
            _recorder.close()
        robot.graceful_shutdown()
        if _viewer is not None:
            _viewer.close()
        print("[shutdown] Done.")

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
    parser = argparse.ArgumentParser(description="Gantry hang test")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sim_parser = subparsers.add_parser("sim", help="Simulation mode")
    sim_parser.add_argument("--config", default="configs/default.yaml")
    sim_parser.add_argument("--viewer", action="store_true",
                            help="Launch MuJoCo viewer (use mjpython)")
    sim_parser.add_argument("--record", nargs="?", const="sim.mp4", default=None,
                            metavar="PATH",
                            help="Record video to MP4 (default: recording.mp4)")

    real_parser = subparsers.add_parser("real", help="Real robot mode")
    real_parser.add_argument("--config", default="configs/default.yaml")
    real_parser.add_argument("--interface", required=True,
                             help="Network interface (e.g. enp3s0)")

    args = parser.parse_args()

    run_test(
        mode=args.mode,
        config_path=args.config,
        interface=getattr(args, "interface", None),
        viewer=getattr(args, "viewer", False),
        record_path=getattr(args, "record", None),
    )


if __name__ == "__main__":
    main()
