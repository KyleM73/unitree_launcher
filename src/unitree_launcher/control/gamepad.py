"""Gamepad e-stop monitor using hidapi (direct USB HID).

Polls a USB gamepad (Logitech F310 in D-Input mode) in a daemon thread
and maps buttons to SafetyController.estop() / clear_estop().

Uses hidapi for direct HID access — no display system interaction,
no conflicts with GLFW (MuJoCo viewer).

Logitech F310 D-Input HID report (8 bytes):
    [0-3] Axes (left stick X/Y, right stick X/Y), center ~0x80
    [4]   Low nibble: hat switch; high nibble: buttons (bit6=B, bit4=X, etc.)
    [5]   Buttons (bit5=Start, bit4=Back, etc.)
    [6]   Unused
    [7]   0xFF at rest
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import hid

from unitree_launcher.control.safety import SafetyController

# Logitech F310 USB IDs (D-Input mode).
F310_VENDOR_ID = 0x046D
F310_PRODUCT_ID = 0xC216

# F310 D-Input button masks (byte index, bit mask).
F310_B_BUTTON = (4, 0x40)       # byte[4] bit 6
F310_START_BUTTON = (5, 0x20)   # byte[5] bit 5


@dataclass
class ButtonSpec:
    """A button identified by its byte offset and bit mask in the HID report."""
    byte: int
    mask: int

    def pressed(self, report: list[int]) -> bool:
        """Return True if this button is pressed in the given report."""
        if self.byte >= len(report):
            return False
        return bool(report[self.byte] & self.mask)


class GamepadMonitor:
    """Polls a USB gamepad via hidapi and triggers safety actions on button press.

    Args:
        safety: SafetyController instance (thread-safe estop/clear_estop).
        vendor_id: USB vendor ID (default: Logitech 0x046D).
        product_id: USB product ID (default: F310 D-Input 0xC216).
        estop_button: (byte_index, bit_mask) for e-stop button.
        clear_button: (byte_index, bit_mask) for clear-estop button.
        poll_hz: Polling rate in Hz (default: 50).
        debug: If True, log raw HID reports on any change.
    """

    def __init__(
        self,
        safety: SafetyController,
        vendor_id: int = F310_VENDOR_ID,
        product_id: int = F310_PRODUCT_ID,
        estop_button: tuple[int, int] = F310_B_BUTTON,
        clear_button: tuple[int, int] = F310_START_BUTTON,
        poll_hz: float = 50.0,
        debug: bool = False,
    ) -> None:
        self._safety = safety
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._estop = ButtonSpec(*estop_button)
        self._clear = ButtonSpec(*clear_button)
        self._poll_hz = poll_hz
        self._debug = debug

        self._device: hid.device | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._estop_held = False
        self._clear_held = False

    def start(self) -> None:
        """Start the polling daemon thread. Idempotent."""
        if self._running:
            return

        self._try_connect()

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="gamepad-estop", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the polling thread and close the HID device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._device is not None:
            self._device.close()
            self._device = None

    @property
    def connected(self) -> bool:
        """Whether the gamepad HID device is currently open."""
        return self._device is not None

    def _try_connect(self) -> bool:
        """Try to open the HID device. Returns True on success."""
        # Close any stale handle first.
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None

        try:
            h = hid.device()
            h.open(self._vendor_id, self._product_id)
            h.set_nonblocking(True)
            self._device = h
        except OSError:
            print("[gamepad] No gamepad detected. Waiting for connection...")
            return False

        name = h.get_product_string() or "Unknown"
        print(f"[gamepad] Connected: {name}")
        print(
            f"[gamepad] E-stop: byte[{self._estop.byte}] & 0x{self._estop.mask:02X} | "
            f"Clear: byte[{self._clear.byte}] & 0x{self._clear.mask:02X}"
        )
        if self._debug:
            print("[gamepad] Debug mode ON — HID report changes will be logged.")
        return True

    def _poll_loop(self) -> None:
        """Background polling loop (runs in daemon thread)."""
        prev_report = None

        while self._running:
            # Reconnect if device is gone.
            if self._device is None:
                if self._try_connect():
                    self._estop_held = False
                    self._clear_held = False
                    prev_report = None
                else:
                    time.sleep(1.0)
                    continue

            try:
                report = self._device.read(64)
            except OSError:
                # Device disconnected.
                print("[gamepad] Gamepad disconnected. Triggering e-stop.")
                self._safety.estop()
                self._device = None
                continue

            if not report:
                time.sleep(1.0 / self._poll_hz)
                continue

            # Debug: log any report change.
            if self._debug and report != prev_report:
                print(f"[gamepad] HID: {[hex(b) for b in report]}")
            prev_report = report

            # E-stop (edge-triggered).
            if self._estop.pressed(report):
                if not self._estop_held:
                    self._estop_held = True
                    self._safety.estop()
                    print("[gamepad] E-STOP triggered")
            else:
                self._estop_held = False

            # Clear e-stop (edge-triggered).
            if self._clear.pressed(report):
                if not self._clear_held:
                    self._clear_held = True
                    if self._safety.clear_estop():
                        print("[gamepad] E-stop cleared")
            else:
                self._clear_held = False

            time.sleep(1.0 / self._poll_hz)


def start_gamepad(safety, debug: bool = False):
    """Create and start a GamepadMonitor, or return None on failure.

    Convenience function used by main.py and scripts. Catches import
    errors (hidapi not installed) and device-not-found gracefully.
    """
    try:
        monitor = GamepadMonitor(safety, debug=debug)
        monitor.start()
        return monitor
    except Exception as exc:
        print(f"[gamepad] WARNING: init failed: {exc}. Continuing without gamepad.")
        return None
