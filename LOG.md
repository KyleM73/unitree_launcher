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
- **Manual step verified (2026-02-11):** Native MuJoCo viewer opens with G1 model, mouse orbit/pan/zoom works, window closes cleanly.

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

---

## Pass 1, Step 2: Phase 1 — Project Scaffolding

**Date:** 2026-02-11
**Status:** COMPLETE

### Task 1.1: Directory Structure

Created full directory tree per PLAN_METAL. All `__init__.py` files in place. `.python-version` contains `3.10`.

### Task 1.2: MuJoCo Model Files

Copied from `reference/unitree_mujoco/unitree_robots/g1/`:
- `g1_29dof.xml` → `assets/robots/g1/`
- `g1_23dof.xml` → `assets/robots/g1/`
- `meshes/` (60 STL files) → `assets/robots/g1/meshes/`

Both models load successfully from the new location. Mesh paths resolve correctly (XML uses `<compiler meshdir="meshes"/>` relative).

Scene files (`scene_29dof.xml`, `scene_23dof.xml`) are Phase 7 work — not yet created.

### Task 1.3: pyproject.toml

- SDK dependency uses git URL: `unitree-sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python.git`
- Dev deps: pytest, pytest-cov, onnx
- `uv pip install -e ".[dev]"` succeeds
- `pytest --co` runs without import errors

### Task 1.4: conftest.py

Fixtures created:
- `g1_29dof_joint_names` — 29 config names in robot-native order
- `g1_23dof_joint_names` — 23 config names in robot-native order
- `isaaclab_29dof_joint_names` — 29 MuJoCo names in IsaacLab order
- `sample_robot_state_dict` — plausible standing state (raw dict, will become `RobotState` in Phase 2)
- `mujoco_model_path_29dof` / `mujoco_model_path_23dof` — absolute paths to MJCF files
- `tmp_log_dir` — temp directory for log output

Helper functions: `create_isaaclab_onnx()`, `create_beyondmimic_onnx()` — for creating test ONNX fixtures.

Note: `sample_config` fixture deferred to Phase 2 (depends on `Config` dataclass). `patch_unitree_threading()` call deferred to Phase 3.

### Tests

16 tests in `tests/test_scaffolding.py` — **all passing**.

```
tests/test_scaffolding.py — 16 passed in 3.01s
```

### Pending for Future Phases

- `sample_robot_state` fixture (proper `RobotState` dataclass) — Phase 2
- `sample_config` fixture (proper `Config` dataclass) — Phase 2
- `patch_unitree_threading()` in conftest.py — Phase 3
- Scene XML files (`scene_29dof.xml`, `scene_23dof.xml`) — Phase 7
- Config YAML files (`configs/default.yaml`, etc.) — Phase 2
