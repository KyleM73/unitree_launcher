"""Gamepad input controller (USB HID, e.g. Logitech F310).

Extends the HID polling with stick axes for velocity commands
and consistent button mapping (A → SHUTDOWN, matching wireless).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from unitree_launcher.controller.input import InputController

logger = logging.getLogger(__name__)

# Logitech F310 D-Input mode
F310_VENDOR_ID = 0x046D
F310_PRODUCT_ID = 0xC216

# HID report layout (Logitech F310, D-Input mode):
#   Byte 0: Left stick X  (0=left, 128=center, 255=right)
#   Byte 1: Left stick Y  (0=up, 127=center, 255=down)
#   Byte 2: Right stick X (0=left, 128=center, 255=right)
#   Byte 3: Right stick Y (0=up, 127=center, 255=down)
#   Byte 4: D-pad (low nibble, value-based) + face buttons (high nibble, bitmask)
#           D-pad: 0=Up, 2=Right, 4=Down, 6=Left, 8=center
#           Face:  0x10=X, 0x20=A, 0x40=B, 0x80=Y
#   Byte 5: Shoulder/start/back/stick clicks (bitmask)
#           0x01=LB, 0x02=RB, 0x10=Back, 0x20=Start, 0x40=L3, 0x80=R3
#   Byte 6: Mode button (0x08=rest, 0x00=pressed)
#   Byte 7: 0xFF at rest
#
# Verified with scripts/tests/gamepad_mapping.py on actual hardware.
F310_A_BUTTON = (4, 0x20)
F310_B_BUTTON = (4, 0x40)
F310_X_BUTTON = (4, 0x10)
F310_Y_BUTTON = (4, 0x80)
F310_START_BUTTON = (5, 0x20)
F310_BACK_BUTTON = (5, 0x10)
F310_RB_BUTTON = (5, 0x02)


@dataclass
class _ButtonSpec:
    byte_idx: int
    mask: int

    def pressed(self, report: list) -> bool:
        if self.byte_idx >= len(report):
            return False
        return bool(report[self.byte_idx] & self.mask)


class GamepadInput(InputController):
    """USB gamepad input via HID polling (Logitech F310, D-Input mode).

    Runs a daemon thread that polls the gamepad at 50 Hz.
    Stick axes map to velocity commands. Buttons map to discrete commands.

    Button mapping (consistent with Unitree wireless controller):

        Button  | Byte | Mask | Command
        --------|------|------|------------------
        A       |  4   | 0x20 | [SHUTDOWN]
        B       |  4   | 0x40 | [MOTION_FADE_OUT]
        X       |  4   | 0x10 | [MOTION_FADE_IN]
        Y       |  4   | 0x80 | [MOTION_RESET]
        Start   |  5   | 0x20 | [ESTOP_CLEAR]
        Back    |  5   | 0x10 | [STOP]
        RB      |  5   | 0x02 | [START]

    Stick mapping:

        Stick        | Byte | Center | Output
        -------------|------|--------|--------
        Left stick X |  0   |  128   | vy (strafe)
        Left stick Y |  1   |  127   | vx (forward/back, inverted)
        Right stick X|  2   |  128   | yaw rate
    """

    def __init__(
        self,
        vendor_id: int = F310_VENDOR_ID,
        product_id: int = F310_PRODUCT_ID,
        poll_hz: float = 50.0,
        deadzone: float = 0.1,
    ):
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._poll_hz = poll_hz
        self._deadzone = deadzone

        self._velocity = np.zeros(3)
        self._commands: set = set()
        self._lock = threading.Lock()

        # Button specs
        self._buttons = {
            "[SHUTDOWN]": _ButtonSpec(*F310_A_BUTTON),
            "[MOTION_FADE_OUT]": _ButtonSpec(*F310_B_BUTTON),
            "[MOTION_FADE_IN]": _ButtonSpec(*F310_X_BUTTON),
            "[MOTION_RESET]": _ButtonSpec(*F310_Y_BUTTON),
            "[ESTOP_CLEAR]": _ButtonSpec(*F310_START_BUTTON),
            "[STOP]": _ButtonSpec(*F310_BACK_BUTTON),
            "[START]": _ButtonSpec(*F310_RB_BUTTON),
        }
        self._button_held = {cmd: False for cmd in self._buttons}

        self._device = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the polling daemon thread."""
        if self._running:
            return
        self._try_connect()
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="gamepad", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._device is not None:
            self._device.close()
            self._device = None

    def update(self) -> None:
        """No-op — polling runs in daemon thread."""
        pass

    def get_velocity(self) -> np.ndarray:
        with self._lock:
            return self._velocity.copy()

    def get_commands(self) -> set:
        with self._lock:
            cmds = self._commands.copy()
            self._commands.clear()
            return cmds

    def _try_connect(self) -> bool:
        """Try to connect to the gamepad via HID.

        Uses open_path() instead of open(vid, pid) because macOS
        blocks game controller reads via the VID/PID open method.
        """
        try:
            import hid
            devices = hid.enumerate(self._vendor_id, self._product_id)
            if not devices:
                return False
            device = hid.device()
            device.open_path(devices[0]["path"])
            device.set_nonblocking(True)
            self._device = device
            logger.info(
                "Gamepad connected: %s (VID=%04X PID=%04X)",
                devices[0].get("product_string", "unknown"),
                self._vendor_id,
                self._product_id,
            )
            return True
        except Exception as exc:
            logger.debug("Gamepad not found: %s", exc)
            return False

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            if self._device is None:
                if not self._try_connect():
                    time.sleep(1.0)
                    continue

            try:
                report = self._device.read(64)
            except OSError:
                logger.warning("Gamepad disconnected")
                with self._lock:
                    self._commands.add("[SHUTDOWN]")
                self._device = None
                continue

            if not report:
                time.sleep(1.0 / self._poll_hz)
                continue

            self._process_report(report)
            time.sleep(1.0 / self._poll_hz)

    def _process_report(self, report: list) -> None:
        """Parse HID report into velocity + commands."""
        with self._lock:
            # Buttons (edge-triggered)
            for cmd, spec in self._buttons.items():
                if spec.pressed(report):
                    if not self._button_held[cmd]:
                        self._button_held[cmd] = True
                        self._commands.add(cmd)
                else:
                    self._button_held[cmd] = False

            # Stick axes (F310 D-Input: bytes 0-3 are axes, center ~0x80)
            if len(report) >= 4:
                lx = (report[0] - 128) / 128.0  # Left stick X -> vy
                ly = (report[1] - 128) / 128.0  # Left stick Y -> vx (inverted)
                rx = (report[2] - 128) / 128.0  # Right stick X -> yaw

                # Apply deadzone
                if abs(lx) < self._deadzone:
                    lx = 0.0
                if abs(ly) < self._deadzone:
                    ly = 0.0
                if abs(rx) < self._deadzone:
                    rx = 0.0

                self._velocity[0] = -ly  # Forward = stick up = negative Y
                self._velocity[1] = -lx  # Left = stick left = negative X
                self._velocity[2] = rx   # Yaw right = positive


def start_gamepad_input(**kwargs) -> Optional[GamepadInput]:
    """Create and start a GamepadInput, or return None on failure."""
    try:
        gamepad = GamepadInput(**kwargs)
        gamepad.start()
        return gamepad
    except Exception as exc:
        logger.warning("Gamepad init failed: %s", exc)
        return None
