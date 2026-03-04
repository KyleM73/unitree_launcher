"""Web-based 3D viewer bridging MuJoCo state to viser.

Streams the robot scene to any browser via WebSockets.  One-time mesh
upload at setup, then per-frame transform updates from mj_data.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import viser

from unitree_launcher.viz.conversions import build_body_meshes, mat_to_quat_wxyz


class ViserViewer:
    """Web-based 3D viewer bridging MuJoCo state to viser.

    Usage::

        viewer = ViserViewer(sim_robot.mj_model, port=8080,
                             policy_paths=[...], active_policy="walk")
        viewer.setup()
        # In main loop:
        viewer.update(sim_robot.mj_data, sim_robot.lock)
        keys = viewer.drain_key_queue()
        policy_path = viewer.drain_policy_selection()
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        port: int = 8080,
        policy_paths: Optional[list[str]] = None,
        active_policy: Optional[str] = None,
        gantry: bool = False,
    ):
        self._mj_model = mj_model
        self._port = port

        # Build display-name -> full-path mapping (strip .onnx suffix).
        self._policy_map: dict[str, str] = {}
        if policy_paths:
            for p in policy_paths:
                name = Path(p).stem
                self._policy_map[name] = p
        self._active_policy_name = active_policy or (
            next(iter(self._policy_map)) if self._policy_map else "unknown"
        )
        self._gantry = gantry

        self._server: viser.ViserServer | None = None
        self._body_handles: dict[int, viser._scene_handles.GlbHandle] = {}
        self._key_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._policy_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._status_handle = None
        self._vx_slider = None
        self._vy_slider = None
        self._yaw_slider = None
        self._base_velocity = None
        self._sim_time = 0.0

        # Follow camera: track pelvis body.
        self._pelvis_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if self._pelvis_id == -1:
            self._pelvis_id = 1
        self._follow_mode = False
        self._follow_origin_xy: Optional[np.ndarray] = None

    def setup(self) -> None:
        """One-time scene upload: meshes, ground grid, and GUI controls."""
        self._server = viser.ViserServer(port=self._port)

        # Scene orientation and lighting.
        self._server.scene.set_up_direction("+z")
        self._server.scene.configure_default_lights()

        # Ground grid (replaces the MuJoCo floor plane geom).
        self._server.scene.add_grid(
            "/ground",
            width=20.0,
            height=20.0,
            cell_size=0.5,
            plane="xy",
            infinite_grid=True,
        )

        # Build per-body trimeshes from MuJoCo model.
        body_meshes = build_body_meshes(self._mj_model)

        # Upload each body mesh as a scene node.
        for body_id, mesh in body_meshes.items():
            body_name = mujoco.mj_id2name(
                self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or f"body_{body_id}"
            handle = self._server.scene.add_mesh_trimesh(
                f"/bodies/{body_name}",
                mesh,
            )
            self._body_handles[body_id] = handle

        # GUI controls.
        self._setup_gui()

        print(f"[viser] Web viewer at http://localhost:{self._port}")

    def _setup_gui(self) -> None:
        """Create GUI buttons and sliders in the viser sidebar."""
        assert self._server is not None

        # ---- Controls folder ----
        self._controls_folder = self._server.gui.add_folder("Controls")
        with self._controls_folder:
            btn_start = self._server.gui.add_button(
                "Start / Stop", color="green"
            )
            btn_start.on_click(lambda _: self._key_queue.put("space"))

            btn_estop = self._server.gui.add_button("E-STOP", color="red")
            btn_estop.on_click(lambda _: self._key_queue.put("backspace"))

            btn_clear = self._server.gui.add_button("Clear E-stop")
            btn_clear.on_click(lambda _: self._key_queue.put("enter"))

            btn_reset = self._server.gui.add_button("Reset")
            btn_reset.on_click(lambda _: self._key_queue.put("delete"))

            self._follow_btn = self._server.gui.add_button("Camera: Free")
            self._follow_btn.on_click(lambda _: self._toggle_follow())

        # ---- Policy folder (hidden in gantry mode) ----
        if not self._gantry:
            with self._server.gui.add_folder("Policy"):
                policy_names = list(self._policy_map.keys()) or [self._active_policy_name]
                for name in policy_names:
                    btn = self._server.gui.add_button(name)
                    btn.on_click(lambda _ev, n=name: self._on_policy_button(n))

        # ---- Status panel ----
        with self._server.gui.add_folder("Status"):
            self._status_handle = self._server.gui.add_markdown(
                "**State:** IDLE | **Time:** 0.0s"
            )

        # ---- Velocity sliders (hidden in gantry mode) ----
        if not self._gantry:
            with self._server.gui.add_folder("Velocity Sliders"):
                self._vx_slider = self._server.gui.add_slider(
                    "vx", min=-1.0, max=1.0, step=0.05, initial_value=0.0
                )
                self._vy_slider = self._server.gui.add_slider(
                    "vy", min=-1.0, max=1.0, step=0.05, initial_value=0.0
                )
                self._yaw_slider = self._server.gui.add_slider(
                    "yaw", min=-1.0, max=1.0, step=0.05, initial_value=0.0
                )
                btn_zero = self._server.gui.add_button("Zero Velocity")
                btn_zero.on_click(lambda _: self._zero_velocity())

    def _toggle_follow(self) -> None:
        """Toggle between follow and free camera modes."""
        self._follow_mode = not self._follow_mode
        self._follow_origin_xy = None  # Re-capture on next frame.
        # Replace button with updated label inside the Controls folder.
        self._follow_btn.remove()
        label = "Camera: Follow" if self._follow_mode else "Camera: Free"
        with self._controls_folder:
            self._follow_btn = self._server.gui.add_button(label)
            self._follow_btn.on_click(lambda _: self._toggle_follow())

    def _on_policy_button(self, name: str) -> None:
        """Policy button clicked -- enqueue the full path for the main loop."""
        path = self._policy_map.get(name)
        if path is not None:
            self._policy_queue.put(path)

    def _zero_velocity(self) -> None:
        """Reset velocity sliders and enqueue the zero-velocity key."""
        if self._vx_slider is not None:
            self._vx_slider.value = 0.0
        if self._vy_slider is not None:
            self._vy_slider.value = 0.0
        if self._yaw_slider is not None:
            self._yaw_slider.value = 0.0
        self._key_queue.put("slash")

    def sync(self, mj_data: mujoco.MjData) -> None:
        """Single-threaded convenience: update scene without a lock.

        Use this from scripts that run physics and rendering on the same
        thread (no lock needed).
        """
        if self._server is None:
            return
        body_xpos = mj_data.xpos.copy()
        body_xmat = mj_data.xmat.copy()
        self._base_velocity = mj_data.qvel[:3].copy()
        self._sim_time = mj_data.time
        self._update_scene(body_xpos, body_xmat)

    def update(self, mj_data: mujoco.MjData, lock: threading.Lock) -> None:
        """Per-frame transform sync from MuJoCo to viser scene nodes.

        Snapshots body positions/orientations under the lock (~50us), then
        updates viser handles outside the lock (WebSocket sends, ~1-5ms).
        """
        if self._server is None:
            return

        # 1. Snapshot under lock (minimal hold time).
        with lock:
            body_xpos = mj_data.xpos.copy()
            body_xmat = mj_data.xmat.copy()
            self._base_velocity = mj_data.qvel[:3].copy()
            self._sim_time = mj_data.time

        self._update_scene(body_xpos, body_xmat)

    def _update_scene(self, body_xpos: np.ndarray, body_xmat: np.ndarray) -> None:
        """Push body transforms to viser scene handles."""
        # Follow mode: offset XY so pelvis stays at the captured origin.
        offset = np.zeros(3)
        if self._follow_mode:
            if self._follow_origin_xy is None:
                # First frame in follow mode — capture origin.
                self._follow_origin_xy = body_xpos[self._pelvis_id][:2].copy()
            offset[:2] = self._follow_origin_xy - body_xpos[self._pelvis_id][:2]

        with self._server.atomic():
            for body_id, handle in self._body_handles.items():
                R = body_xmat[body_id].reshape(3, 3)
                handle.position = body_xpos[body_id] + offset
                handle.wxyz = mat_to_quat_wxyz(R)
        self._server.flush()

    def drain_key_queue(self) -> list[str]:
        """Drain all pending key names from the GUI button queue (FIFO)."""
        keys = []
        while True:
            try:
                keys.append(self._key_queue.get_nowait())
            except queue.Empty:
                break
        return keys

    def drain_policy_selection(self) -> Optional[str]:
        """Return the most recent policy path selected, or None."""
        path = None
        while True:
            try:
                path = self._policy_queue.get_nowait()
            except queue.Empty:
                break
        return path

    def get_velocity_commands(self) -> tuple[float, float, float] | None:
        """Read current velocity slider values.

        Returns:
            (vx, vy, yaw) tuple, or None if sliders not available.
        """
        if self._vx_slider is None:
            return None
        return (
            self._vx_slider.value,
            self._vy_slider.value,
            self._yaw_slider.value,
        )

    def set_status(
        self,
        state: str,
        policy_name: str,
        velocity_command: object = None,
        telemetry: dict | None = None,
    ) -> None:
        """Update the status panel in the GUI sidebar."""
        if self._status_handle is None:
            return
        vel = self._base_velocity
        sim_time = self._sim_time
        if vel is not None:
            vel_lines = (
                f"&nbsp;&nbsp;&nbsp;&nbsp;vx = {vel[0]:+.2f}\n\n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;vy = {vel[1]:+.2f}\n\n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;vz = {vel[2]:+.2f}"
            )
        else:
            vel_lines = "&nbsp;&nbsp;&nbsp;&nbsp;---"
        if velocity_command is not None:
            cmd_lines = (
                f"&nbsp;&nbsp;&nbsp;&nbsp;vx = {velocity_command[0]:+.2f}\n\n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;vy = {velocity_command[1]:+.2f}\n\n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;yaw = {velocity_command[2]:+.2f}"
            )
        else:
            cmd_lines = "&nbsp;&nbsp;&nbsp;&nbsp;---"

        # Telemetry line
        if telemetry:
            hz = telemetry.get("loop_hz", 0.0)
            inf = telemetry.get("inference_ms", 0.0)
            height = telemetry.get("base_height", 0.0)
            steps = telemetry.get("step_count", 0)
            telem_line = f"{hz:.0f} Hz | {inf:.1f}ms inf | {height:.3f}m | step {steps}"
        else:
            telem_line = "---"

        cam_mode = "Follow" if self._follow_mode else "Free"
        self._status_handle.content = (
            f"**State:** {state} | **Policy:** {policy_name}\n\n"
            f"**Telemetry:** {telem_line}\n\n"
            f"**Time:** {sim_time:.1f}s | **Camera:** {cam_mode}\n\n"
            f"**Velocity Cmd:**\n\n{cmd_lines}\n\n"
            f"**Base Velocity:**\n\n{vel_lines}"
        )

    def is_running(self) -> bool:
        """Always True -- the web server never 'closes' like a GLFW window."""
        return True

    def close(self) -> None:
        """Shut down the viser server."""
        if self._server is not None:
            self._server.stop()
            self._server = None
