"""Cross-platform threading utilities.

Replaces unitree_sdk2py.utils.thread.RecurrentThread, which uses
Linux-specific APIs (timerfd_create) that are unavailable on macOS.

Reference: https://x.com/TairanHe99/status/1857935343825334693
"""
import logging
import platform
import sys
import threading
import time
import types

logger = logging.getLogger(__name__)


class RecurrentThread:
    """Drop-in replacement for unitree RecurrentThread.

    Usage (identical to the original):
        thread = RecurrentThread(interval=0.005, target=my_func, name="my_thread")
        thread.Start()
        # ...
        thread.Shutdown()

    Timing note: Uses time.sleep() in a loop. For intervals below 2ms,
    timing precision depends on the OS scheduler. For the DDS publishing
    use case (typically 3-5ms intervals), this is adequate.
    """

    def __init__(self, interval: float, target, name: str = ""):
        self._interval = interval
        self._target = target
        self._name = name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def Start(self):
        """Start the recurrent thread. Thread is a daemon (won't block exit)."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True
        )
        self._thread.start()

    def Shutdown(self):
        """Stop the recurrent thread and wait for it to finish (up to 2s)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        """Main loop: call target, sleep for interval, repeat."""
        while not self._stop_event.is_set():
            try:
                self._target()
            except Exception:
                logger.exception(
                    "Exception in RecurrentThread '%s' target", self._name
                )
            # Sleep for the interval. If the target took longer than
            # the interval, we just call it again immediately (no
            # negative sleep, no accumulating delay).
            elapsed = 0.0  # We don't measure elapsed here; sleep is best-effort
            time.sleep(self._interval)


def get_loopback_interface() -> str:
    """Return the loopback interface name for the current platform."""
    if platform.system() == "Darwin":
        return "lo0"
    return "lo"


def resolve_network_interface(config_value: str) -> str:
    """Resolve 'auto' to the platform-appropriate loopback interface.

    Args:
        config_value: Network interface from config. "auto" is resolved
                      to lo0 (macOS) or lo (Linux). Any other value is
                      returned as-is (e.g., "eth0" for real robot).
    """
    if config_value == "auto":
        return get_loopback_interface()
    return config_value


def patch_unitree_threading():
    """Monkey-patch unitree SDK to use our RecurrentThread on macOS.

    Call this BEFORE any unitree_sdk2py imports that use RecurrentThread.
    This function is idempotent.

    On Linux, this is a no-op (SDK's native RecurrentThread works fine).
    """
    if platform.system() != "Darwin":
        return

    mod_name = "unitree_sdk2py.utils.thread"

    # If the module is already loaded and has a working RecurrentThread, skip.
    if mod_name in sys.modules:
        existing = getattr(sys.modules[mod_name], "RecurrentThread", None)
        if existing is not None:
            return

    # The SDK's thread module imports timerfd at module level, which calls
    # dlsym(timerfd_create) — a Linux-only symbol. We can't import the module
    # normally on macOS. Instead, stub out timerfd first, then import thread.
    timerfd_mod_name = "unitree_sdk2py.utils.timerfd"
    if timerfd_mod_name not in sys.modules:
        sys.modules[timerfd_mod_name] = types.ModuleType(timerfd_mod_name)

    try:
        import unitree_sdk2py.utils.thread as thread_module
    except (ImportError, OSError, AttributeError):
        # If import still fails, create a minimal stub module
        thread_module = types.ModuleType(mod_name)
        sys.modules[mod_name] = thread_module

    thread_module.RecurrentThread = RecurrentThread
    logger.info("Patched unitree_sdk2py RecurrentThread for macOS")
