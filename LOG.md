# LOG.md — Build Progress

> **DO NOT DELETE this file.** Running log of implementation progress.
>
> This is a running log of progress, roadblocks, issues, etc. Not verbose — just enough detail to allow future engineers to pick up where we left off and understand the state of the codebase.

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

- ~~`sample_robot_state` fixture — Phase 2~~ DONE
- ~~`sample_config` fixture — Phase 2~~ DONE
- `patch_unitree_threading()` in conftest.py — Phase 3
- Scene XML files (`scene_29dof.xml`, `scene_23dof.xml`) — Phase 7
- ~~Config YAML files (`configs/default.yaml`, etc.) — Phase 2~~ DONE

---

## Pass 1, Step 3: Phase 2 — Core Data Structures and Configuration [Shared]

**Date:** 2026-02-11
**Status:** COMPLETE

### Task 2.1: Robot Constants (`src/config.py` - Part 1)

All constants implemented:
- `G1_29DOF_JOINTS` (29 entries), `G1_23DOF_JOINTS` (23 entries)
- `G1_29DOF_MUJOCO_JOINTS`, `G1_23DOF_MUJOCO_JOINTS` — config name to MuJoCo name
- `ISAACLAB_G1_29DOF_JOINTS` — 29 MuJoCo names in IsaacLab order
- `ISAACLAB_TO_NATIVE_INDICES` — 29-element index mapping
- `Q_HOME_29DOF`, `Q_HOME_23DOF` — home positions
- `JOINT_LIMITS_29DOF`, `JOINT_LIMITS_23DOF` — position limits (min, max) tuples
- `TORQUE_LIMITS_29DOF`, `TORQUE_LIMITS_23DOF` — max torque per joint
- DDS/IDL name mappings (`_DDS_TO_CONFIG_29DOF`, `_DDS_TO_CONFIG_23DOF`)
- `resolve_joint_name()` — accepts config, MuJoCo, or DDS names

**Note on 23-DOF MuJoCo names:** The 23-DOF config names differ from MuJoCo joint names in the XML:
- `torso` -> `waist_yaw_joint`
- `left_elbow_pitch` -> `left_elbow_joint`
- `left_elbow_roll` -> `left_wrist_roll_joint`
- (same pattern for right arm)

### Task 2.2: Dataclasses (`src/robot/base.py`)

- `RobotState` with `zeros(n_dof)` factory and `copy()` method
- `RobotCommand` with `damping(n_dof, kd)` factory

### Task 2.3: RobotInterface ABC (`src/robot/base.py`)

- `connect()`, `disconnect()`, `get_state()`, `send_command()`, `step()`, `reset()`, `n_dof`

### Task 2.4: PolicyInterface ABC (`src/policy/base.py`)

- `load()`, `reset()`, `get_action()`, `observation_dim`, `action_dim`

### Task 2.5: Config Dataclasses and Loading (`src/config.py` - Part 2)

- 7 config dataclasses: `RobotConfig`, `PolicyConfig`, `ControlConfig`, `SafetyConfig`, `NetworkConfig`, `ViewerConfig`, `LoggingConfig`, `Config`
- `load_config(path)` — YAML loading with validation
- `merge_configs(base, override)` — non-None override values win
- Validation: variant, idl_mode, joint names, gain list lengths, frequency divisibility, logging format

### Task 2.6: YAML Config Files

- `configs/default.yaml` — full default config
- `configs/g1_29dof.yaml` — 29-DOF variant (with BeyondMimic reference gains as comments)
- `configs/g1_23dof.yaml` — 23-DOF variant

### Fixtures Added to conftest.py

- `sample_robot_state` — proper `RobotState` at home position
- `sample_config` — `Config` loaded from `configs/default.yaml`

### Tests

38 tests in `tests/test_config.py` — **all passing**.
54 total tests (Phase 1 + Phase 2) — **all passing**.

```
tests/test_config.py — 38 passed in 0.54s
Full suite — 54 passed in 0.99s
```

### Pending for Future Phases

- ~~`patch_unitree_threading()` in conftest.py — Phase 3~~ DONE
- Scene XML files (`scene_29dof.xml`, `scene_23dof.xml`) — Phase 7

---

## Pass 1, Step 4: Phase 3 — Cross-Platform Compatibility Layer [Shared]

**Date:** 2026-02-11
**Status:** COMPLETE

### Task 3.1: `src/compat.py`

Implemented:
- `RecurrentThread` — drop-in replacement using `threading.Thread` + `time.sleep()` loop
- `get_loopback_interface()` — returns `lo0` (macOS) / `lo` (Linux)
- `resolve_network_interface()` — resolves `"auto"` to platform loopback, passes through explicit values
- `patch_unitree_threading()` — monkey-patches SDK on macOS, no-op on Linux, idempotent

### Task 3.2: conftest.py Updated

`patch_unitree_threading()` now called at top of `tests/conftest.py` before any SDK imports.

### Issues Encountered & Fixes

1. **SDK timerfd import fails at module level** — `unitree_sdk2py.utils.thread` does `from .timerfd import *` at the top level, and `timerfd.py` calls `clib["timerfd_create"]` (a Linux syscall) during module init. This means you can't even `import unitree_sdk2py.utils.thread` on macOS. **Fix:** `patch_unitree_threading()` stubs out `unitree_sdk2py.utils.timerfd` in `sys.modules` with an empty module *before* importing `thread`, then injects our `RecurrentThread` onto the thread module. Also catches `AttributeError` (not just `ImportError`/`OSError`) since the ctypes dlsym lookup raises `AttributeError`.

### Tests

14 tests in `tests/test_compat.py` — **all passing**.
68 total tests (Phase 1 + 2 + 3) — **all passing**.

```
tests/test_compat.py — 14 passed
Full suite — 68 passed in 2.03s
```

### Pending for Future Phases

- Scene XML files (`scene_29dof.xml`, `scene_23dof.xml`) — Phase 7
