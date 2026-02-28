"""Offscreen video recording for MuJoCo simulations.

Captures frames via ``mujoco.Renderer`` and writes them to an MP4 file
using ``cv2.VideoWriter``.  Works in both GUI and headless modes.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import cv2
import mujoco
import numpy as np


def normalize_record_path(name: str, directory: str | None = None) -> str:
    """Build a full recording path from a name and optional directory.

    - Appends ``.mp4`` if not already present.
    - If *directory* is given and *name* has no directory component,
      the file is placed inside *directory*.
    """
    from pathlib import Path as _Path

    p = _Path(name)
    if p.suffix.lower() != ".mp4":
        p = p.with_suffix(".mp4")
    # Only prepend directory when the name is a bare filename (no separators).
    if directory and p.parent == _Path("."):
        p = _Path(directory) / p
    return str(p)


class VideoRecorder:
    """Record offscreen-rendered MuJoCo frames to an MP4 file.

    Frame pacing is driven entirely by a **sim-step counter** — no wall
    time is used.  The counter can come from:

    * ``controller.sim_step_count`` (threaded main.py), or
    * an external integer that the caller increments once per step
      (single-threaded scripts).

    When ``capture()`` is called, it reads the counter and writes
    exactly enough frames to stay in sync.  If the counter advanced
    by more than one since the last call (sim faster than the call
    rate), duplicate frames fill the gap.  If the counter hasn't
    changed, the call is a no-op.  The result: one video frame per
    sim step, every time.

    Args:
        path: Output file path (e.g. ``"output.mp4"``).
        model: ``mujoco.MjModel`` instance.
        data: ``mujoco.MjData`` instance.
        fps: Playback frame rate written to the MP4 header (default 50).
        width: Frame width in pixels (default 1920).
        height: Frame height in pixels (default 1080).
        step_fn: Callable returning the current sim-step count (int).
            If *None*, every call to ``capture()`` writes exactly one
            frame (suitable for single-threaded loops that already
            call ``capture()`` once per step).
    """

    def __init__(
        self,
        path: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        fps: int = 50,
        width: int = 1920,
        height: int = 1080,
        track_body: str = "pelvis",
        distance: float = 4.0,
        azimuth: float = 135.0,
        elevation: float = -20.0,
        step_fn: Callable[[], int] | None = None,
    ) -> None:
        # Clamp to the model's offscreen framebuffer limits.
        max_w = model.vis.global_.offwidth
        max_h = model.vis.global_.offheight
        width = min(width, max_w)
        height = min(height, max_h)

        self._path = str(path)
        self._data = data
        self._fps = fps
        self._width = width
        self._height = height
        self._frame_count = 0

        self._step_fn = step_fn
        self._last_step: int | None = None  # set on first capture

        # Camera that tracks the robot's base body.
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        try:
            self._camera.trackbodyid = model.body(track_body).id
        except KeyError:
            self._camera.trackbodyid = 1  # fallback: first non-world body
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation

        self._renderer = mujoco.Renderer(model, height, width)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self._writer = cv2.VideoWriter(self._path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self._path}")

        self._start_time = time.time()

    def capture(self, data: mujoco.MjData | None = None) -> None:
        """Render and write frame(s) to the video.

        When *step_fn* was provided, frames are written to match the
        sim-step counter (duplicating frames if steps were skipped
        between calls, no-op if the counter hasn't changed).

        When *step_fn* is ``None``, exactly one frame is written per
        call — the caller is responsible for calling once per sim step.

        Args:
            data: Optional override for ``MjData``.  If *None*, uses
                the instance passed at construction time.
        """
        d = data if data is not None else self._data

        if self._step_fn is None:
            # No step counter — one frame per call.
            n_needed = 1
        else:
            current_step = self._step_fn()
            if self._last_step is None:
                self._last_step = current_step
            n_needed = current_step - self._last_step
            if n_needed <= 0:
                return  # counter hasn't advanced — skip
            self._last_step = current_step

        # Render once, write for each missing frame.
        self._renderer.update_scene(d, self._camera)
        frame = self._renderer.render()  # RGB uint8 (H, W, 3)
        bgr = np.ascontiguousarray(frame[:, :, ::-1])  # cv2 expects BGR
        for _ in range(n_needed):
            self._writer.write(bgr)
        self._frame_count += n_needed

    def close(self) -> str:
        """Finalize the video file and print summary.

        Returns:
            The output file path.
        """
        self._writer.release()
        self._renderer.close()

        size_mb = os.path.getsize(self._path) / (1024 * 1024)
        duration = self._frame_count / self._fps if self._fps else 0

        print(
            f"[record] Saved {self._path}: "
            f"{self._frame_count} frames, {duration:.1f}s @ {self._fps} fps, "
            f"{size_mb:.1f} MB"
        )
        return self._path
