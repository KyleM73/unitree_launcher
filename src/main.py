"""MuJoCo viewer and headless runner for the Metal (native macOS) plan.

Provides:
    run_with_viewer() — interactive MuJoCo passive viewer with GLFW key callbacks
    run_headless()    — headless simulation for server/batch evaluation
"""
from __future__ import annotations

import time
from typing import Optional

import mujoco.viewer

from src.control.controller import Controller
from src.robot.sim_robot import SimRobot

# GLFW key constants — printable ASCII chars match their ASCII values.
# See: https://www.glfw.org/docs/latest/group__keys.html
GLFW_KEY_MAP = {
    32: "space",   # GLFW_KEY_SPACE
    65: "a",       # GLFW_KEY_A
    67: "c",       # GLFW_KEY_C
    68: "d",       # GLFW_KEY_D
    69: "e",       # GLFW_KEY_E
    78: "n",       # GLFW_KEY_N
    80: "p",       # GLFW_KEY_P
    81: "q",       # GLFW_KEY_Q
    82: "r",       # GLFW_KEY_R
    83: "s",       # GLFW_KEY_S
    87: "w",       # GLFW_KEY_W
    88: "x",       # GLFW_KEY_X
    90: "z",       # GLFW_KEY_Z
}


def run_with_viewer(sim_robot: SimRobot, controller: Controller) -> None:
    """Run simulation with interactive MuJoCo viewer.

    The viewer runs in the main thread.  The control loop runs in a
    background thread (started by controller.start()).  MuJoCo's
    launch_passive handles its own thread-safety for rendering.

    Key callback fires on key PRESS only (not repeat or release).
    """

    def key_callback(keycode: int) -> None:
        key = GLFW_KEY_MAP.get(keycode)
        if key:
            controller.handle_key(key)

    with mujoco.viewer.launch_passive(
        sim_robot.mj_model,
        sim_robot.mj_data,
        key_callback=key_callback,
    ) as viewer:
        controller.start()
        try:
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 60.0)  # ~60 FPS viewer update
        except KeyboardInterrupt:
            pass
        finally:
            controller.stop()


def run_headless(
    sim_robot: SimRobot,
    controller: Controller,
    duration: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> None:
    """Run simulation without viewer (for server evals).

    Termination conditions (first one wins):
        1. Ctrl+C (KeyboardInterrupt)
        2. *duration* seconds elapsed
        3. *max_steps* policy steps completed
        4. BeyondMimic trajectory ends (controller auto-stops)

    Args:
        duration: Auto-terminate after this many seconds (None = no limit).
        max_steps: Auto-terminate after this many policy steps (None = no limit).
    """
    if max_steps is not None:
        controller._max_steps = max_steps

    controller.start()
    # Auto-start the policy (no viewer to press Space).
    controller.safety.start()

    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)

            if duration is not None and (time.time() - start_time) >= duration:
                print(f"[headless] Duration limit reached ({duration}s). Stopping.")
                break

            if not controller.is_running:
                print("[headless] Controller stopped (trajectory end or error).")
                break

    except KeyboardInterrupt:
        print("\n[headless] Ctrl+C received. Stopping.")
    finally:
        controller.stop()
