# LOG.md — Build Progress

> **DO NOT DELETE this file.** Running log of implementation progress.

---

## Pass 1, Step 1: Phase 0 — Environment Validation [Metal-specific]

**Date:** 2026-02-11
**Status:** COMPLETE

### Task 0.1: Validation Script

Created `scripts/validate_macos.py` per PLAN_METAL spec. Results (all 8/8 pass):

```
  [PASS] Python version: 3.10.18 (OK)
  [PASS] Platform: Darwin arm64
  [PASS] MuJoCo import: 3.4.0
  [PASS] ONNX Runtime import: 1.23.2
  [PASS] CycloneDDS import: 0.10.2
  [PASS] unitree_sdk2py core imports: ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize OK
  [PASS] DDS loopback init: DDS init on lo0 OK
  [PASS] MuJoCo G1 model load: G1 model loaded: 29 actuators, 36 qpos
  [FAIL] RecurrentThread import: dlsym(RTLD_DEFAULT, timerfd_create): symbol not found
         ^ EXPECTED on macOS. Will be replaced by src/compat.py in Phase 3.
VERDICT: macOS bare metal is VIABLE.
```

### Task 0.2: uv Environment Setup

- `uv venv --python 3.10 .venv` — used system Python 3.10.18 from Homebrew
- Core deps installed via `uv pip install`: mujoco 3.4.0, onnxruntime 1.23.2, numpy 2.2.6, pyyaml 6.0.3, h5py 3.15.1, cyclonedds 0.10.2
- SDK installed from GitHub: `uv pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git"`

### Issues Encountered & Fixes

1. **unitree-sdk2py not on PyPI** — The package name `unitree_sdk2_python` and `unitree-sdk2py` both fail on PyPI. Must install from GitHub source.

2. **SDK `__init__.py` imports missing `b2` submodule** — The latest SDK HEAD (a035ade) has `from . import idl, utils, core, rpc, go2, b2` in `__init__.py` but the `b2/` directory doesn't exist in the installed package. This breaks all imports of `unitree_sdk2py`. **Fix:** Patched `.venv/.../unitree_sdk2py/__init__.py` to wrap the import in try/except. This is a venv-local fix — will need to be automated (e.g., post-install hook or pinned commit) for reproducibility.

3. **CycloneDDS 0.10.2 has no `__version__` attribute** — `cyclonedds.__version__` raises AttributeError. **Fix:** Updated `validate_macos.py` to use `importlib.metadata.version('cyclonedds')` instead.

4. **RecurrentThread fails on macOS** — Expected. `timerfd_create` is Linux-only. Will be replaced by `src/compat.py` (Phase 3).

### Task 0.3: MuJoCo Viewer

- Created `scripts/test_viewer.py` for manual verification
- Headless sim verified: 29 actuators, 36 qpos, 35 dof. Sim steps correctly.
- `mujoco.viewer.launch_passive` is importable
- **Manual step required:** Run `source .venv/bin/activate && python scripts/test_viewer.py` to confirm the native window opens with mouse orbit/pan/zoom

### Environment Summary

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.10.18 | OK |
| macOS | Darwin arm64 | OK |
| uv | 0.8.4 | OK |
| MuJoCo | 3.4.0 | OK |
| ONNX Runtime | 1.23.2 | OK |
| NumPy | 2.2.6 | OK |
| CycloneDDS | 0.10.2 | OK |
| unitree_sdk2py | 1.0.1 (git HEAD) | OK (with b2 patch) |
| RecurrentThread | N/A | Expected fail, compat.py pending |
| G1 model | 29 actuators, 36 qpos | OK |

### Known Tech Debt

- SDK `b2` patch is in `.venv/` and will be lost on venv rebuild. Need to either:
  - Pin to a specific SDK commit that doesn't have `b2` in `__init__.py`, or
  - Add a post-install script, or
  - Carry a local patch file
- This should be resolved in Phase 1 when `pyproject.toml` is created

### Verdict

**macOS bare metal is VIABLE. Proceed with PLAN_METAL.md Phase 1.**
