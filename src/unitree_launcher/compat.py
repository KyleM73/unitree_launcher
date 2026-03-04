"""Cross-platform utilities and unitree_sdk2py patches.

Provides:
- RecurrentThread: drop-in for unitree SDK's Linux-only RecurrentThread
- patch_unitree_*: fixes for unitree_sdk2py bugs (b2 import, CRC, threading)
- resolve_network_interface: "auto" -> platform-appropriate loopback

Only needed by MirrorRobot (Python DDS) and diagnostic test scripts.
SimRobot and RealRobot do not use these patches.
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
        """Main loop: call target, wait for interval, repeat.

        Uses a hybrid sleep strategy: sleep for most of the interval
        then busy-wait for the remainder.  This avoids the macOS
        scheduler rounding ``time.sleep(0.002)`` up to ~2.5 ms while
        keeping CPU usage low for longer intervals.
        """
        next_tick = time.perf_counter() + self._interval
        while not self._stop_event.is_set():
            try:
                self._target()
            except Exception:
                logger.exception(
                    "Exception in RecurrentThread '%s' target", self._name
                )
            # Hybrid wait: coarse sleep then busy-wait for precision
            now = time.perf_counter()
            remaining = next_tick - now
            if remaining > 0.001:
                time.sleep(remaining - 0.001)
            while time.perf_counter() < next_tick:
                pass
            # Advance next_tick to avoid drift accumulation
            now = time.perf_counter()
            if now - next_tick > self._interval:
                # We fell behind — reset to avoid burst of catch-up calls
                next_tick = now + self._interval
            else:
                next_tick += self._interval


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


def patch_unitree_b2_import():
    """Prevent unitree_sdk2py from importing nonexistent 'b2' submodule.

    The upstream unitree_sdk2py __init__.py imports ``b2``, which doesn't
    exist and causes an ImportError.  We block the import by injecting
    an empty stub module before unitree_sdk2py is imported.

    This function is idempotent — safe to call multiple times.
    """
    b2_name = "unitree_sdk2py.b2"
    if b2_name not in sys.modules:
        sys.modules[b2_name] = types.ModuleType(b2_name)


def patch_unitree_crc():
    """Patch CRC class to fall back to pure-Python when native .so is missing.

    The upstream unitree_sdk2py ships crc_amd64.so / crc_aarch64.so in a
    ``lib/`` subdirectory, but git-based installs (via uv) don't include
    them because the package lacks proper package_data configuration.

    On Linux the CRC class crashes in __init__ trying to load the .so.
    This patch makes it fall back to the pure-Python implementation,
    which is fine at 50 Hz command rates.
    """
    if platform.system() != "Linux":
        return

    try:
        from unitree_sdk2py.utils.crc import CRC
    except Exception:
        return  # Can't import yet, will be patched later

    _orig_init = CRC.__init__

    def _patched_init(self):
        try:
            _orig_init(self)
        except OSError:
            import ctypes
            # Set up struct pack formats (same as original __init__)
            self._CRC__packFmtLowCmd = '<4B4IH2x' + 'B3x5f3I' * 20 + '4B' + '55Bx2I'
            self._CRC__packFmtLowState = '<4B4IH2x' + '13fb3x' + 'B3x7fb3x3I' * 20 + '4BiH4b15H' + '8hI41B3xf2b2x2f4h2I'
            self._CRC__packFmtHGLowCmd = '<2B2x' + 'B3x5fI' * 35 + '5I'
            self._CRC__packFmtHGLowState = '<2I2B2xI' + '13fh2x' + 'B3x4f2hf7I' * 35 + '40B5I'
            self.platform = "PythonFallback"
            self.crc_lib = None
            logger.warning(
                "CRC native library (crc_amd64.so) not found — using pure-Python fallback"
            )

    CRC.__init__ = _patched_init


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
