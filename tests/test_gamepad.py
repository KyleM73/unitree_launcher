"""Tests for GamepadMonitor (gamepad e-stop via hidapi).

All tests mock the hid module so no actual gamepad is needed.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# F310 D-Input defaults (must match gamepad.py).
ESTOP_BYTE, ESTOP_MASK = 4, 0x40   # B button
CLEAR_BYTE, CLEAR_MASK = 5, 0x20   # Start button

# Resting report: axes centered, no buttons, hat neutral.
RESTING = [0x80, 0x7F, 0x80, 0x7F, 0x08, 0x00, 0x00, 0xFF]


def _report_with(base=None, byte=None, mask=None):
    """Return a copy of base with the given bit set."""
    r = list(base or RESTING)
    if byte is not None and mask is not None:
        r[byte] |= mask
    return r


def _make_mock_hid_device(connected: bool = True):
    """Build a mock hid.device."""
    dev = MagicMock()
    dev.get_product_string.return_value = "Logitech Dual Action"
    # Default: return resting report.
    dev.read.return_value = list(RESTING)
    return dev


def _make_mock_safety():
    """Build a mock SafetyController."""
    safety = MagicMock()
    safety.estop.return_value = None
    safety.clear_estop.return_value = True
    return safety


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_gamepad_module():
    """Ensure the gamepad module is re-imported fresh each test."""
    sys.modules.pop("unitree_launcher.control.gamepad", None)
    yield
    sys.modules.pop("unitree_launcher.control.gamepad", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestButtonSpec:
    """Unit tests for ButtonSpec."""

    def test_pressed_true(self):
        from unitree_launcher.control.gamepad import ButtonSpec
        btn = ButtonSpec(byte=4, mask=0x40)
        assert btn.pressed(_report_with(byte=4, mask=0x40))

    def test_pressed_false(self):
        from unitree_launcher.control.gamepad import ButtonSpec
        btn = ButtonSpec(byte=4, mask=0x40)
        assert not btn.pressed(RESTING)

    def test_short_report(self):
        from unitree_launcher.control.gamepad import ButtonSpec
        btn = ButtonSpec(byte=10, mask=0x01)
        assert not btn.pressed(RESTING)  # report too short


class TestGamepadMonitor:
    """Unit tests for GamepadMonitor."""

    @patch("hid.device")
    def test_estop_button_triggers_safety_estop(self, MockDevice):
        """Pressing B should call safety.estop() once."""
        dev = _make_mock_hid_device()
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        monitor.start()

        # Simulate B button press.
        dev.read.return_value = _report_with(byte=ESTOP_BYTE, mask=ESTOP_MASK)
        time.sleep(0.1)

        monitor.stop()
        safety.estop.assert_called()

    @patch("hid.device")
    def test_clear_button_calls_clear_estop(self, MockDevice):
        """Pressing Start should call safety.clear_estop()."""
        dev = _make_mock_hid_device()
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        monitor.start()

        # Simulate Start button press.
        dev.read.return_value = _report_with(byte=CLEAR_BYTE, mask=CLEAR_MASK)
        time.sleep(0.1)

        monitor.stop()
        safety.clear_estop.assert_called()

    @patch("hid.device")
    def test_edge_detection_no_repeat(self, MockDevice):
        """Holding B should trigger estop only once."""
        dev = _make_mock_hid_device()
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=500)
        monitor.start()

        # Hold B for several poll cycles.
        dev.read.return_value = _report_with(byte=ESTOP_BYTE, mask=ESTOP_MASK)
        time.sleep(0.1)

        monitor.stop()
        assert safety.estop.call_count == 1

    @patch("hid.device")
    def test_disconnect_triggers_estop(self, MockDevice):
        """Gamepad disconnect (OSError on read) should trigger e-stop."""
        dev = _make_mock_hid_device()
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        monitor.start()
        time.sleep(0.05)

        # Simulate disconnect.
        dev.read.side_effect = OSError("disconnected")
        time.sleep(0.1)

        monitor.stop()
        safety.estop.assert_called()

    @patch("hid.device")
    def test_no_device_at_startup(self, MockDevice):
        """No crash when gamepad not connected at startup."""
        dev = _make_mock_hid_device()
        dev.open.side_effect = OSError("not found")
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        monitor.start()
        time.sleep(0.15)
        monitor.stop()

        # No estop — missing device at startup is not a disconnect.
        safety.estop.assert_not_called()

    @patch("hid.device")
    def test_stop_terminates_thread(self, MockDevice):
        """stop() should terminate the polling thread."""
        dev = _make_mock_hid_device()
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        monitor.stop()
        assert monitor._thread is None

    @patch("hid.device")
    def test_start_is_idempotent(self, MockDevice):
        """Calling start() twice should not create a second thread."""
        dev = _make_mock_hid_device()
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        monitor.start()
        thread1 = monitor._thread
        monitor.start()
        thread2 = monitor._thread

        assert thread1 is thread2
        monitor.stop()

    @patch("hid.device")
    def test_connected_property(self, MockDevice):
        """connected property should reflect device state."""
        dev = _make_mock_hid_device()
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        assert not monitor.connected

        monitor.start()
        time.sleep(0.05)
        assert monitor.connected

        monitor.stop()

    @patch("hid.device")
    def test_empty_read_no_action(self, MockDevice):
        """Empty read (no data) should not trigger any action."""
        dev = _make_mock_hid_device()
        dev.read.return_value = []
        MockDevice.return_value = dev
        safety = _make_mock_safety()

        from unitree_launcher.control.gamepad import GamepadMonitor
        monitor = GamepadMonitor(safety, poll_hz=200)
        monitor.start()
        time.sleep(0.1)
        monitor.stop()

        safety.estop.assert_not_called()
        safety.clear_estop.assert_not_called()
