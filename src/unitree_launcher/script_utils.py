"""Shared helpers for test scripts.

Provides:
    build_script_parser   -- argparse with unified sim/real subcommands
    create_robot          -- robot factory from mode + config
    ScriptContext         -- context manager for viewer, gamepad, recorder, shutdown
    phase_settle          -- shared gantry settle phase (3 scripts use this)
"""
from __future__ import annotations

# Patch SDK before any unitree_sdk2py imports.
from unitree_launcher.compat import patch_unitree_b2_import, patch_unitree_threading
patch_unitree_b2_import()
patch_unitree_threading()

import argparse
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from unitree_launcher.config import load_config
from unitree_launcher.robot.base import RobotCommand


# ============================================================================
# Argparse
# ============================================================================

def build_script_parser(
    description: str,
    *,
    sim_only: bool = False,
    extra_args_fn=None,
) -> argparse.ArgumentParser:
    """Build a unified argparse parser for test scripts.

    All scripts get: --config, --gui, --viser, --port, --gamepad, --record.
    Scripts with ``sim_only=False`` get sim/real subcommands with --interface
    and --backend on the real parser.

    Args:
        description: Script description for help text.
        sim_only: If True, no sim/real subcommands (always sim).
        extra_args_fn: Optional callback ``fn(parser)`` to add script-specific
            args. For sim/real scripts, called once per subparser.
    """
    parser = argparse.ArgumentParser(description=description)

    def _add_common(p):
        p.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML configuration file")
        p.add_argument("--gui", action="store_true",
                        help="Launch MuJoCo GLFW viewer (use mjpython on macOS)")
        p.add_argument("--viser", action="store_true",
                        help="Launch viser web viewer")
        p.add_argument("--port", type=int, default=8080,
                        help="Viser web viewer port (default: 8080)")
        p.add_argument("--gamepad", action="store_true",
                        help="Enable gamepad e-stop (Logitech F310)")
        p.add_argument("--record", nargs="?", const="sim.mp4", default=None,
                        metavar="PATH",
                        help="Record video to MP4 (default: sim.mp4)")

    if sim_only:
        _add_common(parser)
        if extra_args_fn:
            extra_args_fn(parser)
    else:
        subparsers = parser.add_subparsers(dest="mode")

        sim_parser = subparsers.add_parser("sim", help="Simulation mode (default)")
        _add_common(sim_parser)
        # Real-only args with defaults so args.interface etc. always exist.
        sim_parser.set_defaults(interface=None, backend="python")
        if extra_args_fn:
            extra_args_fn(sim_parser)

        real_parser = subparsers.add_parser("real", help="Real robot mode")
        _add_common(real_parser)
        real_parser.add_argument("--interface", default="en8",
                                 help="Network interface (default: en8)")
        real_parser.add_argument("--backend", choices=["python", "cpp"],
                                 default="python",
                                 help="Real robot SDK backend")
        if extra_args_fn:
            extra_args_fn(real_parser)

    return parser


# ============================================================================
# Robot factory
# ============================================================================

def create_robot(mode: str, config, interface: str | None = None,
                 backend: str = "python"):
    """Create and configure a SimRobot or RealRobot from mode string.

    Returns the robot instance (not yet connected).
    """
    if mode == "sim":
        from unitree_launcher.robot.sim_robot import SimRobot
        config.network.domain_id = 1
        return SimRobot(config)
    else:
        config.network.domain_id = 0
        config.network.interface = interface
        if backend == "cpp":
            from unitree_launcher.robot.cpp_real_robot import CppRealRobot
            return CppRealRobot(config)
        else:
            from unitree_launcher.robot.real_robot import RealRobot
            return RealRobot(config)


# ============================================================================
# Script context manager
# ============================================================================

class ScriptContext:
    """Manages viewer, gamepad, recorder, and shutdown for test scripts.

    Usage::

        with ScriptContext(robot, config, gui=True, viser=True) as ctx:
            phase_settle(robot, ..., ctx.sync, ctx.realtime)
            ...
        # Automatic cleanup: stop gamepad, close recorder, shutdown robot,
        # close viewers.
    """

    def __init__(
        self,
        robot,
        config,
        *,
        is_sim: bool = True,
        gui: bool = False,
        viser: bool = False,
        port: int = 8080,
        gamepad: bool = False,
        record_path: str | None = None,
        mirror=None,
    ):
        self.robot = robot
        self._config = config
        self._is_sim = is_sim

        # ---- Viewer ----
        self._glfw_viewer = None
        self._viser_viewer = None
        self._mirror = mirror

        show = gui or viser or (not is_sim and not gui and not viser and mirror is not None)
        self.realtime = show or not is_sim

        if viser and is_sim:
            from unitree_launcher.viz.viser_viewer import ViserViewer
            self._viser_viewer = ViserViewer(robot.mj_model, port=port)
            self._viser_viewer.setup()

        if gui:
            import mujoco.viewer
            if mirror is not None:
                self._glfw_viewer = mujoco.viewer.launch_passive(
                    mirror.model, mirror.data
                )
            elif is_sim:
                self._glfw_viewer = mujoco.viewer.launch_passive(
                    robot.mj_model, robot.mj_data
                )

        # ---- Recorder ----
        self._recorder = None
        if record_path and is_sim:
            from unitree_launcher.recording import VideoRecorder, normalize_record_path
            self._recorder = VideoRecorder(
                normalize_record_path(record_path),
                robot.mj_model, robot.mj_data,
            )

        # ---- Gamepad ----
        self._gamepad_monitor = None
        self._safety = None
        if gamepad:
            from unitree_launcher.control.safety import SafetyController
            from unitree_launcher.control.gamepad import start_gamepad
            self._safety = SafetyController(config, n_dof=robot.n_dof)
            self._safety.start()
            self._gamepad_monitor = start_gamepad(self._safety)

    @property
    def safety(self):
        """The SafetyController (created if gamepad is enabled, else None)."""
        return self._safety

    def check_estop(self) -> bool:
        """Return True if e-stopped (callers should abort phase)."""
        if self._safety is None:
            return False
        from unitree_launcher.control.safety import SystemState
        return self._safety.state == SystemState.ESTOP

    def sync(self):
        """Per-step callback: update all active viewers and recorder."""
        if self._mirror is not None:
            state = self.robot.get_state()
            self._mirror.update(state)
        if self._viser_viewer is not None:
            self._viser_viewer.sync(self.robot.mj_data)
        if self._glfw_viewer is not None:
            self._glfw_viewer.sync()
        if self._recorder is not None:
            self._recorder.capture()

    def close(self):
        """Clean up all resources."""
        if self._gamepad_monitor is not None:
            self._gamepad_monitor.stop()
        if self._recorder is not None:
            self._recorder.close()
        self.robot.graceful_shutdown()
        if self._viser_viewer is not None:
            self._viser_viewer.close()
        if self._glfw_viewer is not None:
            self._glfw_viewer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ============================================================================
# Shared phases (used by gantry, right_arm, wrist scripts)
# ============================================================================

def phase_settle(
    robot, band, torso_id, is_sim,
    sync=lambda: None, realtime: bool = False,
    check_estop=lambda: False,
    duration: float = 5.0,
) -> None:
    """Sim only: find gravity equilibrium for the hanging robot.

    Zeros all gains and lets the robot hang under gravity + elastic band
    until velocities converge. Used by gantry, arm, and wrist scripts.
    """
    from unitree_launcher.gantry import apply_band

    if not is_sim:
        return

    model = robot.mj_model
    data = robot.mj_data
    dt = model.opt.timestep
    n_iters = int(duration / dt)

    print(f"[settle] Finding gravity equilibrium ({duration}s, "
          f"{n_iters} physics steps)...")

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
        if check_estop():
            print("[settle] E-STOP. Aborting.")
            break
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

    model.actuator_gainprm[:] = orig_gainprm
    model.actuator_biasprm[:] = orig_biasprm
    model.dof_damping[:] = orig_dof_damping
    print("[settle] Done.")
    sync()
