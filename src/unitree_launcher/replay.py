"""Replay mode: play back logged robot data in MuJoCo viewer.

Provides ``run_replay()`` (the ``uv run replay`` entry point). Loads logged
data via ``LogReplay``, reconstructs ``RobotState`` at each step, and
renders through the same ``RealtimeMirror`` display used by mirror mode.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from unitree_launcher.datalog.replay import LogReplay
from unitree_launcher.mirror import RealtimeMirror


def run_replay(args: argparse.Namespace) -> None:
    """Run the replay viewer: load logged data and play back in viewer.

    Supports --gui (MuJoCo GLFW viewer), --viser (web viewer),
    and text summary/csv export (no viewer flags).
    """
    replay = LogReplay(args.log_dir)
    replay.load()
    n = replay.n_steps

    if n == 0:
        print(f"No data in {args.log_dir}")
        return

    # Always print summary
    print(replay.summary())
    print()

    # Text-only modes (no viewer)
    if not args.gui and not args.viser:
        if args.format == "csv":
            output_path = args.output or str(Path(args.log_dir) / "export.csv")
            replay.to_csv(output_path)
            print(f"Exported {n} steps to {output_path}")
        return

    # Viewer modes: use actual per-step timestamps for real-time playback
    import mujoco

    timestamps = replay._data["timestamps"]
    speed = args.speed
    mirror = RealtimeMirror()

    state = replay.get_state_at(0)
    mirror.update(state)

    if args.gui:
        with mujoco.viewer.launch_passive(mirror.model, mirror.data) as viewer:
            step = 0
            wall_start = time.monotonic()
            sim_start = float(timestamps[0])
            while viewer.is_running():
                state = replay.get_state_at(step)
                mirror.update(state)
                viewer.sync()

                step += 1
                if step >= n:
                    if args.loop:
                        step = 0
                        wall_start = time.monotonic()
                        sim_start = float(timestamps[0])
                    else:
                        break

                # Sleep until wall clock catches up to the next timestamp
                sim_elapsed = (float(timestamps[step]) - sim_start) / speed
                wall_elapsed = time.monotonic() - wall_start
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    elif args.viser:
        from unitree_launcher.viz.viser_viewer import ViserViewer
        viser_viewer = ViserViewer(mirror.model, port=args.port)
        viser_viewer.setup()
        try:
            step = 0
            wall_start = time.monotonic()
            sim_start = float(timestamps[0])
            while True:
                state = replay.get_state_at(step)
                mirror.update(state)
                viser_viewer.sync(mirror.data)

                step += 1
                if step >= n:
                    if args.loop:
                        step = 0
                        wall_start = time.monotonic()
                        sim_start = float(timestamps[0])
                    else:
                        break

                sim_elapsed = (float(timestamps[step]) - sim_start) / speed
                wall_elapsed = time.monotonic() - wall_start
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            viser_viewer.close()
