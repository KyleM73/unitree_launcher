"""Tests for compat.py — cross-platform utilities and SDK patches."""
import threading
import time
from unittest import mock

import pytest

from unitree_launcher.compat import (
    RecurrentThread,
    get_loopback_interface,
    patch_unitree_threading,
    resolve_network_interface,
)


# ---------------------------------------------------------------------------
# RecurrentThread tests
# ---------------------------------------------------------------------------


class TestRecurrentThread:
    def test_recurrent_thread_runs(self):
        """Target function is called multiple times within 0.2s at 10ms interval."""
        counter = {"n": 0}

        def target():
            counter["n"] += 1

        t = RecurrentThread(interval=0.01, target=target, name="test_runs")
        t.Start()
        time.sleep(0.2)
        t.Shutdown()
        assert counter["n"] >= 5  # conservative lower bound

    def test_recurrent_thread_interval(self):
        """Calls are spaced approximately by the interval (within 50% tolerance)."""
        timestamps = []

        def target():
            timestamps.append(time.monotonic())

        interval = 0.02
        t = RecurrentThread(interval=interval, target=target, name="test_interval")
        t.Start()
        time.sleep(0.25)
        t.Shutdown()

        # Check gaps between consecutive timestamps
        assert len(timestamps) >= 3
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        median_gap = sorted(gaps)[len(gaps) // 2]
        # 50% tolerance for CI jitter
        assert interval * 0.5 <= median_gap <= interval * 2.0

    def test_recurrent_thread_shutdown(self):
        """Thread stops within 1s after Shutdown()."""
        t = RecurrentThread(interval=0.01, target=lambda: None, name="test_shutdown")
        t.Start()
        time.sleep(0.05)
        t.Shutdown()
        assert t._thread is not None
        assert not t._thread.is_alive()

    def test_recurrent_thread_daemon(self):
        """Thread is a daemon thread."""
        t = RecurrentThread(interval=0.01, target=lambda: None, name="test_daemon")
        t.Start()
        assert t._thread.daemon is True
        t.Shutdown()

    def test_recurrent_thread_double_start(self):
        """Calling Start() twice doesn't crash."""
        counter = {"n": 0}

        def target():
            counter["n"] += 1

        t = RecurrentThread(interval=0.01, target=target, name="test_double")
        t.Start()
        t.Start()  # Should not raise
        time.sleep(0.05)
        t.Shutdown()

    def test_recurrent_thread_shutdown_before_start(self):
        """Calling Shutdown() before Start() doesn't crash or raise."""
        t = RecurrentThread(interval=0.01, target=lambda: None, name="test_pre_shutdown")
        t.Shutdown()  # Should not raise

    def test_recurrent_thread_target_exception(self):
        """If target raises, thread logs error and continues running."""
        call_count = {"n": 0}

        def bad_target():
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise ValueError("boom")

        t = RecurrentThread(interval=0.01, target=bad_target, name="test_exc")
        t.Start()
        time.sleep(0.15)
        t.Shutdown()
        # Thread survived the exceptions and kept calling target
        assert call_count["n"] >= 5

    def test_recurrent_thread_slow_target(self):
        """When target takes longer than interval, thread still runs."""
        call_count = {"n": 0}

        def slow_target():
            call_count["n"] += 1
            if call_count["n"] == 1:
                time.sleep(0.05)  # Slower than interval

        t = RecurrentThread(interval=0.01, target=slow_target, name="test_slow")
        t.Start()
        time.sleep(0.2)
        t.Shutdown()
        # Should have called multiple times despite the slow first call
        assert call_count["n"] >= 3


# ---------------------------------------------------------------------------
# Network interface tests
# ---------------------------------------------------------------------------


class TestNetworkInterface:
    def test_get_loopback_interface_macos(self):
        """Returns lo0 on macOS."""
        with mock.patch("unitree_launcher.compat.platform") as mock_plat:
            mock_plat.system.return_value = "Darwin"
            assert get_loopback_interface() == "lo0"

    def test_get_loopback_interface_linux(self):
        """Returns lo on Linux."""
        with mock.patch("unitree_launcher.compat.platform") as mock_plat:
            mock_plat.system.return_value = "Linux"
            assert get_loopback_interface() == "lo"

    def test_resolve_auto(self):
        """resolve_network_interface('auto') returns platform-appropriate value."""
        result = resolve_network_interface("auto")
        assert result in ("lo0", "lo")

    def test_resolve_explicit(self):
        """resolve_network_interface('eth0') returns 'eth0' unchanged."""
        assert resolve_network_interface("eth0") == "eth0"


# ---------------------------------------------------------------------------
# patch_unitree_threading tests
# ---------------------------------------------------------------------------


class TestPatchUnitreeThreading:
    def test_patch_unitree_threading_idempotent(self):
        """Calling patch_unitree_threading() twice is safe."""
        patch_unitree_threading()
        patch_unitree_threading()  # Should not raise

    def test_patch_unitree_threading_linux_noop(self):
        """On Linux, patch_unitree_threading() is a no-op."""
        with mock.patch("unitree_launcher.compat.platform") as mock_plat:
            mock_plat.system.return_value = "Linux"
            # Should return immediately without touching SDK
            patch_unitree_threading()
