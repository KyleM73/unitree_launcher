# Unitree G1 Deployment Stack - Bare Metal Implementation Plan

## Document Purpose

This is a **fully self-contained** implementation plan for the Unitree G1 deployment stack, targeting **native macOS (Apple Silicon)** as the primary development and simulation platform with support for **headless Linux servers** for batch evaluations.

This document can be executed by an engineer without reference to PLAN_DOCKER.md (the Docker/Viser plan). Both plans share the same SPEC.md and the same core codebase. Sections marked **[Shared]** are identical between plans. Sections marked **[Metal-specific]** diverge from the Docker/Viser plan.

**Project documentation files (`SPEC.md`, `PLAN_DOCKER.md`, `PLAN_METAL.md`, `WORK.md`) must never be deleted.** They are the authoritative record of requirements, design decisions, and implementation plans. They should be updated in place as the project evolves.

---

## Why This Plan Exists

Research from the previous planning phase revealed:

1. **CycloneDDS has macOS ARM64 wheels** for Python 3.8-3.10 (`cyclonedds==0.10.2`).
2. **Tairan He (HybridRobotics/BeyondMimic) confirmed** the Unitree SDK works on Mac Mini M4. The only incompatibility is `RecurrentThread` — a Linux-specific threading wrapper trivially replaced with `threading.Thread` + `time.sleep()`.
3. **MuJoCo has a native macOS viewer** (`mujoco.viewer`) that provides interactive 3D visualization with zero additional code.

This eliminates the need for Docker, Viser, EGL/OSMesa, and headless rendering workarounds during development and local simulation.

---

## What Differs from PLAN_DOCKER.md (Docker/Viser Plan)

| This Plan (Metal) | Docker/Viser Plan (PLAN_DOCKER.md) |
|---|---|
| Native macOS + `uv` virtualenv | Docker container (Ubuntu 20.04) |
| MuJoCo native viewer | Viser browser-based visualization |
| Keyboard controls (WASD + hotkeys) | Viser UI panel (buttons, sliders, dropdowns) |
| `--headless` flag for server evals | Docker with `MUJOCO_GL=egl` |
| Python 3.10 (CycloneDDS ceiling) | Python 3.8+ in Docker |
| `RecurrentThread` shim in `src/compat.py` | Uses SDK's native `RecurrentThread` (Linux) |
| No Docker, no Viser dependencies | Full Docker + Viser stack |

**Code shared between both plans:** All modules in `src/` except `src/viz/` (Docker-only) are shared. The `src/compat.py` module is used by both plans (on Linux it simply delegates to the SDK's native implementation when available).

---

## Guiding Principles

1. **Test as you go.** Every module gets unit tests written alongside the implementation code. Tests are run after every change. Do not move to the next task until the current task's tests pass.
2. **Build bottom-up.** Start with foundational data structures and pure-logic modules that have zero external dependencies. These are the easiest to test and the hardest to change later.
3. **One concern per module.** Each file does one thing. If a module is doing two things, split it.
4. **Sim-first, real-second.** The entire sim path must work end-to-end before real robot code is tested. Real robot code is too dangerous to iterate on casually.
5. **Fail loudly at startup, not at runtime.** All configuration validation, dimension checks, and joint name resolution happen at initialization. Runtime errors trigger E-stop.
6. **Native first.** Run directly on the developer's machine. Docker is for CI/deployment, not daily development.
7. **Python 3.10 is the ceiling.** CycloneDDS ARM64 wheels stop at 3.10. Do not use 3.11+ features.
8. **uv for environment management.** Fast, deterministic, can install Python 3.10 automatically.

---

## Assumptions

| ID | Assumption | Rationale |
|----|-----------|-----------|
| A1 | **Pure Python SDK** (`unitree_sdk2_python`). No ROS/ROS2. | Simpler dependencies, matches reference implementation. |
| A2 | **29-DOF is the primary target.** 23-DOF support is architected but lightly tested. | User has only a 29-DOF robot. |
| A3 | **Real ONNX policy files are available** for integration testing. Mock models for unit tests. | User confirmed. |
| A4 | **macOS (Apple Silicon) is the primary dev platform.** | Developer has an M-series Mac. |
| A5 | **Python 3.10** is used (highest version with CycloneDDS ARM64 wheels). | PyPI shows `cp310-macosx_11_0_arm64` is the latest ARM64 wheel. |
| A6 | **DDS loopback uses `lo0` on macOS**, `lo` on Linux. | macOS names its loopback interface `lo0`. |
| A7 | **MuJoCo viewer is used for interactive visualization on macOS.** Headless evals on Linux use `mj_step` only. | Native viewer is zero-cost and full-featured. |
| A8 | **RecurrentThread is the only macOS incompatibility** in `unitree_sdk2_python`. Phase 0 validates this. | Confirmed by Tairan He's Mac Mini M4 work. |
| A9 | **No GPU inference.** ONNX Runtime CPU provider is sufficient for the small policy networks. | Avoids CUDA dependency. |
| A10 | **Real robot deployment still requires Linux** + Ethernet. | DDS multicast to robot hardware needs a real network stack, not macOS. |
| A11 | **The control loop must never be blocked** by visualization, logging, or UI. On real hardware, a missed cycle could cause a fall. | Safety-critical requirement. |
| A12 | **Joint name mapping accepts all three naming conventions** and resolves to a canonical form at startup. | SPEC sections 2.2, 4.6. |

---

## Design Decisions and Trade-offs

### D1: Threading Model - Threads vs AsyncIO **[Shared]**

**Decision:** Use `threading` with locks, not `asyncio`.

**Rationale:** The reference implementation uses threads. MuJoCo is not async-compatible. The DDS SDK uses callback threads internally. Mixing asyncio with these would add complexity for no benefit.

**Trade-off:** Threads have GIL contention, but the control loop does mostly NumPy/ONNX operations which release the GIL.

### D2: DDS Architecture - Bridge vs Direct **[Shared]**

**Decision:** In simulation mode, the DDS bridge runs in-process. In real robot mode, the controller publishes/subscribes directly to the robot's DDS topics.

**Rationale:** Keeping it in-process avoids inter-process DDS discovery issues. The SimRobot class encapsulates MuJoCo + DDS bridge as a single unit.

### D3: Configuration Hierarchy **[Shared]**

**Decision:** CLI args override YAML config. YAML config has a default file that can be extended by robot-specific configs.

**Resolution order:** CLI > YAML (robot-specific merges over default) > Code defaults > ONNX metadata (BeyondMimic only, when `use_onnx_metadata: true`).

### D4: Observation Normalization **[Shared]**

**Decision:** No observation normalization applied by default. If required, it must be embedded in the ONNX model or configured separately.

### D5: Torque Commands vs Position Commands **[Shared]**

**Decision:** Send position targets with PD gains via `LowCmd` message. The motor controller computes actual torque via: `tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff`.

### D6: No Docker for Primary Workflow **[Metal-specific]**

**Decision:** Use `uv` virtual environment on bare metal. Docker is available as an optional CI/deployment target (see PLAN_DOCKER.md).

**Rationale:** Native macOS runs MuJoCo viewer, CycloneDDS, and the full SDK. Docker adds latency (Linux VM on macOS), prevents native viewer, and forces EGL/Viser workarounds.

### D7: MuJoCo Viewer for Visualization **[Metal-specific]**

**Decision:** Use `mujoco.viewer.launch_passive()` for interactive visualization. Keyboard callbacks on the viewer window handle start/stop/e-stop/reset/velocity commands.

**Rationale:** The MuJoCo viewer provides 3D rendering, mouse orbit/pan/zoom, contact force visualization, body selection, and runs at display refresh rate — all for free.

### D8: Joint Name Mapping Architecture **[Shared]**

**Decision:** Three name spaces: (1) Config names (`left_hip_pitch`), (2) IsaacLab/MuJoCo joint names (`left_hip_pitch_joint`), (3) DDS/IDL names (`L_LEG_HIP_PITCH`). A single `JointMapper` class handles all translations.

### D9: Headless Mode for Server Evals **[Metal-specific]**

**Decision:** `--headless` flag disables the MuJoCo viewer. `--duration` and `--steps` flags provide automatic termination. BeyondMimic policies auto-terminate at trajectory end.

### D10: Loopback Interface Auto-Detection **[Shared]**

**Decision:** Auto-detect the loopback interface name (`lo0` on macOS, `lo` on Linux) when `interface` is set to `"auto"` in config.

### D11: Cross-Platform RecurrentThread **[Shared]**

**Decision:** `src/compat.py` provides a cross-platform `RecurrentThread` that works on both macOS and Linux. On Linux, if the SDK's native implementation is available, the compat module can optionally delegate to it, but the pure-Python version is the default for consistency.

**Rationale:** Both plans benefit from a single threading implementation that works everywhere. The compat module is shared code.

---

## Execution Guide

This section provides a high-level roadmap for executing this plan. Use it to understand the critical path, identify parallelization opportunities, and track progress.

### Dependency Graph

```
Phase 0: Environment Validation [Metal]
  │
  v
Phase 1: Project Scaffolding
  │
  ├──────────────────────────────────┐
  v                                  v
Phase 2: Core Data Structures    Phase 3: Compat Layer
  │         [Shared]                [Shared]
  │                                  │
  ├──────────┬──────────┐            │
  v          v          v            v
Phase 4    Phase 5    Phase 6    Phase 7: MuJoCo Sim
Joint Map  Safety     Policies    & DDS Bridge
[Shared]   [Shared]   [Shared]      │
  │          │          │            │
  │          └──────────┴────────────┤
  │                                  │
  │          ┌───────────────────────┤
  v          v                       │
Phase 10   Phase 8: Control Loop ◄──┘
Logging       │
[Shared]      ├─────────────────────┐
  │           v                     v
  │        Phase 9: Viewer     Phase 11: Real Robot
  │        [Metal]             [Shared]
  │           │                     │
  └───────────┴─────────────────────┤
                                    v
                          Phase 12: CLI [Metal]
                                    │
                                    v
                          Phase 13: Integration Tests
```

### Critical Path

The longest dependency chain determines the minimum sequential work:

**Phase 0 → 1 → 2 → 4 → 6 → 7 → 8 → 9 → 12 → 13**

All other phases can be parallelized alongside this chain.

### Phase Execution Summary

| Phase | Description | Depends On | Can Parallelize With | Est. Tasks |
|-------|-------------|------------|---------------------|------------|
| **0** | Environment Validation [Metal] | — | — | 3 |
| **1** | Project Scaffolding | Phase 0 | — | 4 |
| **2** | Core Data Structures & Config [Shared] | Phase 1 | Phase 3 | 6 |
| **3** | Cross-Platform Compat [Shared] | Phase 1 | Phase 2 | 2 |
| **4** | Joint Mapping & Observations [Shared] | Phase 2 | Phases 5, 6, 7, 10 | 2 |
| **5** | Safety System [Shared] | Phase 2 | Phases 4, 6, 7, 10 | 1 |
| **6** | Policy Interfaces [Shared] | Phases 2, 4 | Phases 5, 7, 10 | 4 |
| **7** | MuJoCo Sim & DDS Bridge | Phases 2, 3 | Phases 4, 5, 6, 10 | 3 |
| **8** | Control Loop | Phases 5, 6, 7 | Phase 10 | 1 |
| **9** | MuJoCo Viewer Integration [Metal] | Phases 7, 8 | Phases 10, 11 | 3 |
| **10** | Logging System [Shared] | Phase 2 | Phases 4-9 | 2 |
| **11** | Real Robot Interface [Shared] | Phases 2, 3 | Phases 9, 10 | 1 |
| **12** | CLI Entry Point [Metal] | Phases 8, 9, 10, 11 | — | 2 |
| **13** | Integration Testing | Phase 12 | — | 6 |

### Parallelization Opportunities

**After Phase 1 completes**, two independent tracks can proceed simultaneously:
- **Track A**: Phase 2 (data structures + config)
- **Track B**: Phase 3 (compat layer)

**After Phase 2 completes**, four independent tracks can proceed:
- **Track A**: Phase 4 (joint mapping) → Phase 6 (policies)
- **Track B**: Phase 5 (safety)
- **Track C**: Phase 7 (MuJoCo sim + DDS) — also needs Phase 3
- **Track D**: Phase 10 (logging)

**After Phase 8 (control loop) completes**, two more independent tracks:
- **Track A**: Phase 9 (viewer)
- **Track B**: Phase 11 (real robot) — can start after Phases 2+3

### Cross-Plan Reference (PLAN_DOCKER.md)

Phases marked **[Shared]** produce identical code in both plans. If implementing both plans, shared phases only need to be done once. The Docker plan differs only in:
- No Phase 0 (environment validation is Docker-specific)
- No Phase 9 (Viser visualization replaces MuJoCo viewer)
- Phase 11 is Docker configuration instead of CLI scripts
- Phase 12 adds Docker platform testing

---

## Phase 0: Environment Validation **[Metal-specific]**

**Purpose:** Prove that all core dependencies work on macOS bare metal before writing any application code. If this phase fails, fall back to PLAN_DOCKER.md.

### Task 0.1: Run Validation Script

Create and run `scripts/validate_macos.py`. This script must be runnable standalone (no project imports).

```python
#!/usr/bin/env python3
"""Validate that all core dependencies work on macOS bare metal."""
import sys
import platform

def check(name, fn):
    try:
        result = fn()
        print(f"  [PASS] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

results = []

# 1. Python version
results.append(check("Python version",
    lambda: f"{sys.version} ({'OK' if sys.version_info[:2] == (3, 10) else 'WANT 3.10'})"))

# 2. Platform
results.append(check("Platform",
    lambda: f"{platform.system()} {platform.machine()}"))

# 3. MuJoCo
results.append(check("MuJoCo import",
    lambda: __import__('mujoco').__version__))

# 4. ONNX Runtime
results.append(check("ONNX Runtime import",
    lambda: __import__('onnxruntime').__version__))

# 5. CycloneDDS
results.append(check("CycloneDDS import",
    lambda: __import__('cyclonedds').__version__))

# 6. unitree_sdk2py core imports
def check_sdk():
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    return "ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize OK"
results.append(check("unitree_sdk2py core imports", check_sdk))

# 7. DDS loopback init
def check_dds():
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    iface = "lo0" if platform.system() == "Darwin" else "lo"
    ChannelFactoryInitialize(1, iface)
    return f"DDS init on {iface} OK"
results.append(check("DDS loopback init", check_dds))

# 8. MuJoCo model load
def check_model():
    import mujoco
    model = mujoco.MjModel.from_xml_path(
        "reference/unitree_mujoco/unitree_robots/g1/scene.xml")
    return f"G1 model loaded: {model.nu} actuators, {model.nq} qpos"
results.append(check("MuJoCo G1 model load", check_model))

# 9. RecurrentThread import (expected to fail on macOS)
def check_recurrent():
    from unitree_sdk2py.utils.thread import RecurrentThread
    return "RecurrentThread available (unexpected on macOS — good, means SDK is fully native)"
recurrent_ok = check("RecurrentThread import", check_recurrent)
if not recurrent_ok:
    print("    ^ This is EXPECTED on macOS. We will replace it with src/compat.py.")

# Summary
passed = sum(results)
total = len(results)
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
if not recurrent_ok:
    print("RecurrentThread failure is expected and will be patched.")
if passed >= total - 1:  # Allow RecurrentThread to fail
    print("VERDICT: macOS bare metal is VIABLE. Proceed with PLAN_METAL.md.")
else:
    print("VERDICT: Too many failures. Fall back to PLAN_DOCKER.md (Docker).")
```

**Acceptance criteria:**
- All checks pass except (optionally) RecurrentThread
- If RecurrentThread fails, the error is `ImportError` or `OSError` (not a deeper SDK issue)
- MuJoCo model loads with the expected actuator count

**What to do if validation partially fails:**
- If only RecurrentThread fails: proceed, this is expected and handled by `src/compat.py`
- If SDK core imports fail (check #6) but RecurrentThread also fails: the SDK may fail to import because RecurrentThread is imported at module init time. Try the monkey-patch approach in Task 3.2 and re-run
- If DDS init fails (check #7): verify `lo0` interface exists (`ifconfig lo0`). If it does, this may indicate a CycloneDDS issue — check CycloneDDS version
- If 3+ checks fail: fall back to PLAN_DOCKER.md (Docker)

### Task 0.2: Set Up uv Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv pinned to Python 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install core dependencies
uv pip install mujoco onnxruntime numpy pyyaml h5py cyclonedds==0.10.2

# Install unitree SDK
uv pip install unitree_sdk2_python

# Verify
python scripts/validate_macos.py
```

**Acceptance criteria:**
- `uv venv --python 3.10` succeeds (uv downloads Python 3.10 if needed)
- `uv pip install` completes without build errors
- `scripts/validate_macos.py` passes

### Task 0.3: Validate MuJoCo Viewer Opens

Manual check: run a script that opens the MuJoCo viewer with the G1 model.

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "reference/unitree_mujoco/unitree_robots/g1/scene.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
```

**Acceptance criteria:**
- A native window opens showing the G1 robot
- Mouse orbit/pan/zoom works
- Closing the window exits cleanly

---

## Phase 1: Project Scaffolding

### Task 1.1: Create Directory Structure

```
unitree_launcher/
├── configs/
│   ├── default.yaml
│   ├── g1_29dof.yaml
│   └── g1_23dof.yaml
├── assets/
│   └── robots/
│       └── g1/
├── policies/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── compat.py               # [Shared] Cross-platform shims
│   ├── robot/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sim_robot.py
│   │   └── real_robot.py
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── isaaclab_policy.py
│   │   ├── beyondmimic_policy.py
│   │   ├── joint_mapper.py
│   │   └── observations.py
│   ├── control/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   └── safety.py
│   └── logging/
│       ├── __init__.py
│       ├── logger.py
│       └── replay.py
├── scripts/
│   ├── validate_macos.py       # [Metal-specific]
│   ├── run_sim.sh
│   ├── run_real.sh
│   ├── run_eval.sh             # [Metal-specific] Headless batch eval
│   └── replay_log.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_joint_mapper.py
│   ├── test_observations.py
│   ├── test_safety.py
│   ├── test_isaaclab_policy.py
│   ├── test_beyondmimic_policy.py
│   ├── test_logger.py
│   ├── test_sim_robot.py
│   ├── test_controller.py
│   └── test_compat.py
├── pyproject.toml
└── .python-version             # Contains "3.10" for uv
```

**Note:** No `docker/` directory and no `src/viz/` directory. Those exist only in the Docker/Viser plan (PLAN_DOCKER.md). The Docker plan adds `docker/Dockerfile`, `docker/docker-compose.yml`, and `src/viz/{server.py, robot_viz.py, ui.py}`.

**Acceptance criteria:**
- All directories exist
- All `__init__.py` files exist (can be empty)
- `.python-version` contains `3.10`
- `git status` shows the new structure

### Task 1.2: Copy MuJoCo Model Files

Copy from `reference/unitree_mujoco/unitree_robots/g1/` into `assets/robots/g1/`:
- `g1_29dof.xml`
- `g1_23dof.xml`
- Entire `meshes/` directory

**Do not copy `scene.xml` or `scene_terrain.xml`.** Those include environment elements we will build ourselves.

**Acceptance criteria:**
- `assets/robots/g1/g1_29dof.xml` exists and is valid XML
- `assets/robots/g1/g1_23dof.xml` exists and is valid XML
- `assets/robots/g1/meshes/` contains all STL files
- MuJoCo can load both models: `mujoco.MjModel.from_xml_path("assets/robots/g1/g1_29dof.xml")` succeeds

**Important note:** The MJCF files reference mesh paths relative to their location. Verify that mesh paths resolve correctly from `assets/robots/g1/`. If the XML uses `<compiler meshdir="meshes"/>` (relative), this should work. If it uses absolute paths, they must be patched.

### Task 1.3: Set Up `pyproject.toml`

```toml
[project]
name = "unitree-launcher"
version = "0.1.0"
requires-python = ">=3.8,<3.11"
dependencies = [
    "mujoco>=3.0",
    "onnxruntime>=1.16",
    "numpy",
    "pyyaml",
    "h5py",
    "cyclonedds==0.10.2",
    "unitree_sdk2_python",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "onnx>=1.14",      # Required for creating test ONNX model fixtures
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Key notes:**
- `requires-python = ">=3.8,<3.11"` enforces the Python 3.10 ceiling (CycloneDDS ARM64 wheels stop at 3.10)
- No `viser` dependency (that's Docker-plan only)
- `onnx` is a dev dependency needed for creating synthetic ONNX models in test fixtures
- `unitree_sdk2_python` is installed from PyPI/GitHub; it depends on `cyclonedds`
- Pin `cyclonedds==0.10.2` explicitly because `unitree_sdk2_python` pins this version

**Acceptance criteria:**
- `uv pip install -e ".[dev]"` succeeds in the .venv
- `pytest` runs (even with 0 tests) without import errors

### Task 1.4: Create `conftest.py` with Shared Fixtures

**File:** `tests/conftest.py`

Define pytest fixtures that will be reused across test files:

```python
# Fixtures to create:
# - g1_29dof_joint_names: list of 29 joint config names in robot-native order
# - g1_23dof_joint_names: list of 23 joint config names in robot-native order
# - isaaclab_29dof_joint_names: list of 29 joint names in IsaacLab order
# - sample_robot_state: a RobotState with plausible values for a standing robot
# - sample_config: a Config object with default values
# - mujoco_model_path_29dof: path to the 29-DOF MJCF file
# - mujoco_model_path_23dof: path to the 23-DOF MJCF file
# - tmp_log_dir: a temporary directory for log output (cleaned up after test)
```

**Acceptance criteria:**
- `conftest.py` is importable
- Fixtures produce valid data
- `pytest --co` lists the fixtures

---

## Phase 2: Core Data Structures and Configuration **[Shared]**

These modules have **zero external dependencies** beyond Python stdlib and NumPy. They are the foundation everything else builds on. Every function must be unit tested.

### Task 2.1: Implement Robot Constants (`src/config.py` - Part 1: Constants)

Define the robot joint configuration constants as Python data. These are the source of truth for joint names, limits, and home positions.

**Contents:**
- `G1_29DOF_JOINTS`: List of 29 config-name strings in robot-native order (from SPEC section 2.2)
- `G1_23DOF_JOINTS`: List of 23 config-name strings in robot-native order (from SPEC section 2.3)
- `G1_29DOF_MUJOCO_JOINTS`: Mapping from config name to MuJoCo joint name (e.g., `"left_hip_pitch"` -> `"left_hip_pitch_joint"`)
- `ISAACLAB_G1_29DOF_JOINTS`: List of 29 MuJoCo joint names in IsaacLab order (from SPEC section 2.2.1)
- `ISAACLAB_TO_NATIVE_INDICES`: Precomputed index mapping array (from SPEC section 2.2.1 mapping table)
- `Q_HOME_29DOF`: Dict of config-name -> home position (from SPEC section 2.4)
- `Q_HOME_23DOF`: Dict of config-name -> home position (from SPEC section 2.4)
- `JOINT_LIMITS_29DOF`: Dict of config-name -> (min, max) tuple (from SPEC section 2.2)
- `TORQUE_LIMITS_29DOF`: Dict of config-name -> max torque (from SPEC section 2.2)

**Tests (`tests/test_config.py` - Part 1):**
- `test_29dof_joint_count`: `len(G1_29DOF_JOINTS) == 29`
- `test_23dof_joint_count`: `len(G1_23DOF_JOINTS) == 23`
- `test_29dof_home_position_keys_match`: All keys in `Q_HOME_29DOF` match `G1_29DOF_JOINTS`
- `test_23dof_home_position_keys_match`: All keys in `Q_HOME_23DOF` match `G1_23DOF_JOINTS`
- `test_joint_limits_keys_match`: All keys in `JOINT_LIMITS_29DOF` match `G1_29DOF_JOINTS`
- `test_home_position_within_limits`: Every value in `Q_HOME_29DOF` is within the corresponding joint limit
- `test_torque_limits_keys_match`: All keys in `TORQUE_LIMITS_29DOF` match `G1_29DOF_JOINTS`
- `test_23dof_joint_limits_keys_match`: All keys in `JOINT_LIMITS_23DOF` match `G1_23DOF_JOINTS`
- `test_23dof_home_within_limits`: Every value in `Q_HOME_23DOF` is within corresponding 23-DOF joint limits
- `test_isaaclab_index_mapping_length`: `len(ISAACLAB_TO_NATIVE_INDICES) == 29`
- `test_isaaclab_index_mapping_bijective`: All 29 native indices appear exactly once
- `test_isaaclab_index_mapping_known_values`: Verify specific index values against SPEC section 2.2.1 mapping table (e.g., IsaacLab index 0 maps to native index 0, index 1 maps to native index 6, etc.). A bijective-but-wrong mapping must not pass.
- `test_mujoco_joint_name_mapping_complete`: Every config name has a MuJoCo name

**Run tests:** `pytest tests/test_config.py -v`

### Task 2.2: Implement State and Command Dataclasses (`src/robot/base.py` - Part 1) **[Shared]**

Define `RobotState` and `RobotCommand` as Python dataclasses (from SPEC section 5.2).

```python
@dataclass
class RobotState:
    timestamp: float
    joint_positions: np.ndarray      # (N_DOF,)
    joint_velocities: np.ndarray     # (N_DOF,)
    joint_torques: np.ndarray        # (N_DOF,) estimated
    imu_quaternion: np.ndarray       # (4,) wxyz
    imu_angular_velocity: np.ndarray # (3,)
    imu_linear_acceleration: np.ndarray # (3,)
    base_position: np.ndarray        # (3,) world frame (sim only, NaN for real)
    base_velocity: np.ndarray        # (3,) world frame (sim only, NaN for real)

@dataclass
class RobotCommand:
    joint_positions: np.ndarray   # (N_DOF,) target positions
    joint_velocities: np.ndarray  # (N_DOF,) target velocities
    joint_torques: np.ndarray     # (N_DOF,) feedforward torques
    kp: np.ndarray                # (N_DOF,) position gains
    kd: np.ndarray                # (N_DOF,) velocity gains
```

Also define factory methods:
- `RobotState.zeros(n_dof: int) -> RobotState`: Create a zero-initialized state
- `RobotCommand.damping(n_dof: int, kd: float) -> RobotCommand`: Create a damping-mode command

**Tests (`tests/test_config.py` - Part 2):**
- `test_robot_state_zeros`: Verify shapes and values
- `test_robot_command_damping`: Verify kp is zero, kd is set, positions are zero
- `test_robot_state_copy`: Verify deep copy semantics (modifying copy doesn't affect original)
- `test_robot_state_nan_base`: Create `RobotState` with `base_position=NaN, base_velocity=NaN` (real robot case). Verify no exceptions on creation, copy, or attribute access.

### Task 2.3: Implement Abstract `RobotInterface` (`src/robot/base.py` - Part 2) **[Shared]**

```python
class RobotInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def get_state(self) -> RobotState: ...
    @abstractmethod
    def send_command(self, cmd: RobotCommand) -> None: ...
    @abstractmethod
    def step(self) -> None: ...
    @abstractmethod
    def reset(self, initial_state: Optional[RobotState] = None) -> None: ...
    @property
    @abstractmethod
    def n_dof(self) -> int: ...
```

**Tests:** No tests needed for a pure ABC.

### Task 2.4: Implement Abstract `PolicyInterface` (`src/policy/base.py`) **[Shared]**

```python
class PolicyInterface(ABC):
    @abstractmethod
    def load(self, path: str) -> None: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray: ...
    @property
    @abstractmethod
    def observation_dim(self) -> int: ...
    @property
    @abstractmethod
    def action_dim(self) -> int: ...
```

**Note:** `get_action` uses `**kwargs` to accommodate BeyondMimic's `time_step` input.

**Tests:** No tests needed for a pure ABC.

### Task 2.5: Implement Configuration Loading (`src/config.py` - Part 2) **[Metal-specific config]**

Define a `Config` dataclass and YAML loading with validation.

```python
@dataclass
class RobotConfig:
    variant: str = "g1_29dof"
    idl_mode: int = 0

@dataclass
class PolicyConfig:
    format: Optional[str] = None  # "isaaclab", "beyondmimic", or None (auto-detect)
    observed_joints: Optional[List[str]] = None
    controlled_joints: Optional[List[str]] = None
    use_onnx_metadata: bool = True
    use_estimator: bool = True  # False to omit base_lin_vel from obs (--no-est overrides to False)

@dataclass
class ControlConfig:
    policy_frequency: int = 50
    sim_frequency: int = 200
    kp: Union[float, List[float]] = 100.0
    kd: Union[float, List[float]] = 10.0
    ka: Union[float, List[float]] = 0.5
    kd_damp: float = 5.0
    q_home: Optional[Dict[str, float]] = None

@dataclass
class SafetyConfig:
    joint_position_limits: bool = True
    joint_velocity_limits: bool = True
    torque_limits: bool = True

@dataclass
class NetworkConfig:
    interface: str = "auto"       # "auto" detects lo0 (macOS) / lo (Linux)
    domain_id: int = 1

@dataclass
class ViewerConfig:
    enabled: bool = True          # False for --headless
    sync: bool = True             # Sync viewer to sim time

@dataclass
class LoggingConfig:
    enabled: bool = True
    format: str = "hdf5"
    compression: str = "gzip"
    log_frequency: int = 50

@dataclass
class Config:
    robot: RobotConfig
    policy: PolicyConfig
    control: ControlConfig
    safety: SafetyConfig
    network: NetworkConfig
    viewer: ViewerConfig          # Metal: ViewerConfig. Docker plan uses ViserConfig instead.
    logging: LoggingConfig

def load_config(path: str) -> Config:
    """Load and validate config from YAML file."""
    ...

def merge_configs(base: Config, override: Config) -> Config:
    """Merge override config into base config (non-None values win)."""
    ...
```

**Note on code sharing:** The Docker/Viser plan uses `ViserConfig` (with `host`, `port`, `default_camera`, `default_render`) instead of `ViewerConfig`. Both plans share the same `Config` class but with different visualization config fields. The recommended approach for sharing: define a base `Config` with all shared fields and use a `viz` field whose type is either `ViewerConfig` or `ViserConfig` depending on the plan. Alternatively, keep `ViewerConfig` as a superset that includes Viser fields with sensible defaults — this is the simpler approach and is recommended.

**Joint name resolution in configs:**
Implement `resolve_joint_name(name: str, variant: str) -> str` that normalizes any accepted name to the canonical config-name form:
- Config names: `left_hip_pitch` -> `left_hip_pitch`
- IsaacLab/MuJoCo names: `left_hip_pitch_joint` -> `left_hip_pitch`
- DDS/IDL names: `L_LEG_HIP_PITCH` -> `left_hip_pitch`

This function is called during config loading for every joint name in `observed_joints` and `controlled_joints`. If a name doesn't match any known joint, raise `ValueError` with the unrecognized name and a list of valid names.

**Validation at load time:**
- `variant` must be `"g1_29dof"` or `"g1_23dof"`
- If `controlled_joints` is specified, every name must resolve to a valid joint for the variant
- If `observed_joints` is specified, every name must resolve to a valid joint for the variant
- If `kp`/`kd`/`ka` are lists, their length must match the number of controlled joints
- `policy_frequency` must evenly divide `sim_frequency`
- `idl_mode` must be 0 or 1

**Tests (`tests/test_config.py` - Part 3):**
- `test_load_default_config`: Load `configs/default.yaml`, verify all fields populated
- `test_load_robot_specific_config`: Load `configs/g1_29dof.yaml` merged over default
- `test_invalid_variant_rejected`: Variant `"g1_99dof"` raises `ValueError`
- `test_invalid_joint_name_rejected`: Unknown joint in `controlled_joints` raises `ValueError`
- `test_kp_list_wrong_length_rejected`: `kp` list length != number of controlled joints raises `ValueError`
- `test_frequency_divisibility`: `sim_frequency=200, policy_frequency=50` OK; `sim_frequency=200, policy_frequency=60` raises
- `test_cli_override_merges`: CLI args override YAML values
- `test_default_gains_are_scalar`: When `kp` is a scalar, it's broadcast to all joints
- `test_joint_name_resolution_config_name`: `left_hip_pitch` resolves correctly
- `test_joint_name_resolution_mujoco_name`: `left_hip_pitch_joint` resolves to `left_hip_pitch`
- `test_joint_name_resolution_dds_name`: `L_LEG_HIP_PITCH` resolves to `left_hip_pitch`
- `test_joint_name_resolution_unknown_raises`: `nonexistent_joint` raises `ValueError`
- `test_viewer_config_defaults`: `ViewerConfig` has `enabled=True, sync=True`
- `test_idl_mode_validation`: `idl_mode=2` raises `ValueError` (only 0 or 1 allowed)
- `test_logging_format_validation`: `format="invalid"` raises `ValueError` (only "hdf5" or "npz" allowed)
- `test_merge_configs_none_preserves_base`: `merge_configs(base, override)` where override field is `None` does not overwrite the base value
- `test_ka_list_wrong_length_rejected`: `ka` list length != number of controlled joints raises `ValueError`

### Task 2.6: Write Default YAML Configuration Files

**File:** `configs/default.yaml`
```yaml
robot:
  variant: g1_29dof
  idl_mode: 0

policy:
  observed_joints: null
  controlled_joints: null
  use_onnx_metadata: true

control:
  policy_frequency: 50
  sim_frequency: 200
  kp: 100.0
  kd: 10.0
  ka: 0.5
  kd_damp: 5.0
  q_home: null

safety:
  joint_position_limits: true
  joint_velocity_limits: true
  torque_limits: true

network:
  interface: "auto"
  domain_id: 1

viewer:
  enabled: true
  sync: true

logging:
  enabled: true
  format: hdf5
  compression: gzip
  log_frequency: 50
```

**File:** `configs/g1_29dof.yaml` — 29-DOF specific config (inherits from default). Include BeyondMimic reference gains as comments.

**File:** `configs/g1_23dof.yaml` — 23-DOF specific config (sets `variant: g1_23dof`).

**Acceptance criteria:**
- `load_config("configs/default.yaml")` succeeds
- `load_config("configs/g1_29dof.yaml")` succeeds
- `load_config("configs/g1_23dof.yaml")` succeeds
- All three pass validation

---

## Phase 3: Cross-Platform Compatibility Layer **[Shared]**

This is the critical macOS enablement work. The `compat.py` module is shared between both plans.

### Task 3.1: Implement `RecurrentThread` Replacement (`src/compat.py`)

```python
"""Cross-platform threading utilities.

Replaces unitree_sdk2py.utils.thread.RecurrentThread, which uses
Linux-specific APIs (POSIX real-time threads) that are unavailable on macOS.

Reference: https://x.com/TairanHe99/status/1857935343825334693
"""
import threading
import time
import platform


class RecurrentThread:
    """Drop-in replacement for unitree RecurrentThread.

    Usage (identical to the original):
        thread = RecurrentThread(interval=0.005, target=my_func, name="my_thread")
        thread.Start()
        # ...
        thread.Shutdown()

    Timing note: Uses time.sleep() in a loop. For intervals below 2ms,
    timing precision depends on the OS scheduler. For the DDS publishing
    use case (typically 3-5ms intervals), this is adequate. If sub-millisecond
    precision is needed in the future, consider a busy-wait approach.
    """

    def __init__(self, interval: float, target, name: str = ""):
        self._interval = interval
        self._target = target
        self._name = name
        self._stop_event = threading.Event()
        self._thread = None

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
            self._target()
            # Use sleep instead of Event.wait for more consistent timing.
            # Event.wait can return early when the event is set, which is
            # fine for shutdown but sleep gives better interval consistency.
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

    Call this BEFORE any unitree_sdk2py imports if the SDK fails to import
    due to RecurrentThread at module init time. This function is idempotent.

    On Linux, this is a no-op (SDK's native RecurrentThread works fine).
    """
    if platform.system() != "Darwin":
        return

    try:
        # Test if the SDK's RecurrentThread already works
        from unitree_sdk2py.utils.thread import RecurrentThread as _
        return  # It works, no patch needed
    except (ImportError, OSError):
        pass

    # Inject our RecurrentThread into the SDK's module
    import importlib
    import unitree_sdk2py.utils.thread as thread_module
    thread_module.RecurrentThread = RecurrentThread
```

**Tests (`tests/test_compat.py`):**
- `test_recurrent_thread_runs`: Target function is called multiple times within 0.2s at 10ms interval
- `test_recurrent_thread_interval`: Calls are spaced approximately by the interval (within 50% tolerance for CI)
- `test_recurrent_thread_shutdown`: Thread stops within 1s after Shutdown()
- `test_recurrent_thread_daemon`: Thread is a daemon thread
- `test_recurrent_thread_double_start`: Calling Start() twice doesn't crash
- `test_recurrent_thread_shutdown_before_start`: Calling Shutdown() before Start() doesn't crash or raise
- `test_recurrent_thread_target_exception`: If target raises, thread logs error and continues running (does not silently die). Verify thread stays alive after an exception in one iteration.
- `test_recurrent_thread_slow_target`: When target takes longer than interval, thread still runs (no accumulating delay crash). Verify it calls target again immediately without negative sleep.
- `test_get_loopback_interface`: Returns `lo0` on macOS, `lo` on Linux
- `test_resolve_auto`: `resolve_network_interface("auto")` returns platform-appropriate value
- `test_resolve_explicit`: `resolve_network_interface("eth0")` returns `"eth0"` unchanged
- `test_patch_unitree_threading_idempotent`: Calling `patch_unitree_threading()` twice is safe (no error, no double-patch)
- `test_patch_unitree_threading_linux_noop`: On Linux (or mock `platform.system()="Linux"`), `patch_unitree_threading()` is a no-op — does not modify the SDK module

### Task 3.2: Validate SDK Imports with Patch

After Task 3.1, verify that the full import chain works on macOS:

```python
from src.compat import patch_unitree_threading
patch_unitree_threading()  # Must be called before SDK imports

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from src.compat import resolve_network_interface

iface = resolve_network_interface("auto")
ChannelFactoryInitialize(1, iface)
print(f"DDS initialized on {iface}")
```

**Acceptance criteria:**
- All unitree SDK imports succeed on macOS
- `ChannelFactoryInitialize(1, "lo0")` succeeds on macOS
- No errors or warnings from CycloneDDS

**Where to call `patch_unitree_threading()`:** It must be called at the very top of `src/main.py` (before any other imports from `src/` that might transitively import the SDK) and at the top of `tests/conftest.py`.

---

## Phase 4: Joint Mapping and Observation Building **[Shared]**

These modules depend only on Phase 2 (data structures + config). They are pure NumPy operations with no I/O. Every function must be unit tested thoroughly because **incorrect joint mapping can damage the physical robot.**

### Task 4.1: Implement `JointMapper` (`src/policy/joint_mapper.py`)

**Key behaviors:**
1. Accept `robot_joints` (all joints in native order), optional `observed_joints`, optional `controlled_joints`
2. Resolution order when args are None:
   - If both None: observe and control all joints in robot-native order
   - If only `controlled_joints` specified: `observed_joints` = `controlled_joints`
   - If only `observed_joints` specified: `controlled_joints` = all joints
   - If both specified: use as given
3. Build index arrays mapping policy order <-> robot-native order
4. `robot_to_observation()`: Extract observed joints from full state, in policy order
5. `robot_to_action()`: Extract controlled joints from full state, in policy order
6. `action_to_robot()`: Map policy action (n_controlled,) to full robot array (n_total,), with default for uncontrolled
7. Validate all joint names at init time, raise `ValueError` for unknowns

**Edge cases:**
- Duplicate joint names -> raise `ValueError`
- Empty `controlled_joints` -> raise `ValueError`
- `observed_joints` == `controlled_joints` -> valid, common case

**Properties to implement:**
- `observed_indices: np.ndarray` — indices into robot state array for observed joints, in policy order
- `controlled_indices: np.ndarray` — indices into robot state array for controlled joints, in policy order
- `non_controlled_indices: np.ndarray` — indices for non-controlled joints
- `n_observed: int`, `n_controlled: int`, `n_total: int`

**Tests (`tests/test_joint_mapper.py`):**
- `test_default_all_joints`: Both None -> observe and control all 29 in native order
- `test_controlled_only`: Only `controlled_joints` specified -> observed defaults to controlled
- `test_observed_only`: Only `observed_joints` specified -> controlled defaults to all
- `test_both_specified`: Both specified with different sets
- `test_identity_mapping`: When policy order == robot order, indices are identity
- `test_isaaclab_reordering`: Use IsaacLab 29-DOF order, verify indices match SPEC section 2.2.1 mapping table
- `test_robot_to_observation_values`: Create known robot state, verify observation output matches expected reordering
- `test_robot_to_action_values`: Create known robot state, verify action output matches expected subset/reorder
- `test_action_to_robot_values`: Create known action, verify full array has actions at correct indices and defaults elsewhere
- `test_action_to_robot_roundtrip`: `action_to_robot(robot_to_action(full))` reconstructs controlled joints correctly
- `test_partial_control_12_legs`: 12 leg joints controlled, 17 in damping. Verify `non_controlled_indices` has exactly 17 entries.
- `test_partial_control_7_arm`: 7 right arm joints controlled
- `test_invalid_joint_raises`: Unknown joint name raises `ValueError`
- `test_duplicate_joint_raises`: Duplicate joint in controlled list raises `ValueError`
- `test_empty_controlled_raises`: Empty controlled list raises `ValueError`
- `test_n_observed_property`: Verify `n_observed` matches len(observed_joints)
- `test_n_controlled_property`: Verify `n_controlled` matches len(controlled_joints)
- `test_n_total_property`: Always equals len(robot_joints)
- `test_23dof_joint_mapper_basic`: Create JointMapper with `G1_23DOF_JOINTS`, verify `n_total == 23` and basic observation/action extraction works
- `test_isaaclab_reordering_known_state`: Create state `[0.0, 0.1, ..., 2.8]`, apply IsaacLab reordering, verify specific numerical output values (not just that indices are valid)
- `test_action_to_robot_default_value`: When `action_to_robot()` fills uncontrolled joints, verify the default value (current position or zero) and document which

**Run tests:** `pytest tests/test_joint_mapper.py -v`

### Task 4.2: Implement `ObservationBuilder` (`src/policy/observations.py`) **[Shared]**

```python
class ObservationBuilder:
    def __init__(self, joint_mapper: JointMapper, config: ControlConfig,
                 use_estimator: bool = True): ...

    @property
    def observation_dim(self) -> int:
        """2*n_observed + n_controlled + (12 if use_estimator else 9)"""
        ...

    def build(self, robot_state: RobotState, last_action: np.ndarray,
              velocity_command: np.ndarray) -> np.ndarray:
        """Build observation vector (IsaacLab PolicyCfg order):
        [base_lin_vel (3)?, base_ang_vel (3), projected_gravity (3), velocity_commands (3),
         joint_pos (n_observed), joint_vel (n_observed), actions (n_controlled)]
        base_lin_vel omitted when use_estimator=False (--no-est).
        """
        ...

    def compute_projected_gravity(self, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Rotate [0, 0, -1] by inverse of IMU quaternion. Returns (3,) vector."""
        ...

    def compute_body_velocity_in_body_frame(self, world_velocity: np.ndarray,
                                             quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Transform world-frame velocity to body frame using IMU quaternion."""
        ...
```

**Observation vector layout (from SPEC section 4.2, IsaacLab PolicyCfg order):**
```
[base_lin_vel (3)?, base_ang_vel (3), projected_gravity (3), velocity_commands (3),
 joint_pos (n_observed), joint_vel (n_observed), actions (n_controlled)]
```
`base_lin_vel` is omitted when `use_estimator=False` (set via `policy.use_estimator: false` in config, or `--no-est` CLI override).

**Key implementation details:**
- `joint_pos` values are **relative to home pose**: `q - q_home` (from `joint_mapper.robot_to_observation()` output minus corresponding `q_home` entries)
- `joint_vel` comes from `joint_mapper.robot_to_observation()` (raw velocities)
- `actions` (last_action) has dimension `n_controlled` (not `n_observed`)
- Linear velocity must be transformed from world frame to body frame using IMU quaternion
- Projected gravity: rotate `[0, 0, -1]` by the inverse of the IMU quaternion
- Angular velocity is already in body frame (from IMU gyroscope)
- When `use_estimator=False`, `base_lin_vel` is **omitted entirely** (not zeroed), reducing obs_dim by 3
- MuJoCo uses wxyz quaternion convention

**Tests (`tests/test_observations.py`):**

*Dimension tests:*
- `test_observation_dim_full_body`: 12 + 29 + 29 + 29 = 99 (with estimator)
- `test_observation_dim_full_body_no_est`: 9 + 29 + 29 + 29 = 96 (without estimator)
- `test_observation_dim_partial_control`: 12 + 29 + 29 + 12 = 82 (with estimator)
- `test_observation_dim_partial_control_no_est`: 9 + 29 + 29 + 12 = 79 (without estimator)
- `test_observation_dim_isolated`: 12 + 12 + 12 + 12 = 48 (with estimator)
- `test_observation_dim_isolated_no_est`: 9 + 12 + 12 + 12 = 45 (without estimator)

*Gravity and velocity transforms:*
- `test_projected_gravity_upright`: Quaternion [1,0,0,0] -> gravity = [0, 0, -1]
- `test_projected_gravity_tilted_forward`: Known tilt -> verify gravity vector
- `test_projected_gravity_inverted`: Upside down -> gravity = [0, 0, 1]
- `test_projected_gravity_90_degree_roll`: Quaternion for 90° roll -> gravity in horizontal plane `[0, ±1, 0]`
- `test_body_velocity_transform_identity`: Upright robot, world vel = body vel
- `test_body_velocity_transform_rotated`: Known rotation -> verify transform
- `test_body_velocity_transform_180_yaw`: 180° yaw rotation, world vel `[1,0,0]` -> body vel `[-1,0,0]`

*Build output (IsaacLab PolicyCfg order):*
- `test_build_output_shape`: Output shape matches observation_dim
- `test_build_output_shape_no_est`: Output shape matches observation_dim when `use_estimator=False`
- `test_build_joint_positions_are_relative`: Known state with known `q_home` -> verify `joint_pos` segment equals `q - q_home` (not raw positions)
- `test_build_joint_velocities_correct`: Known state -> verify joint_vel segment (raw velocities, not relative)
- `test_build_actions_correct`: Known action -> verify actions segment at correct offset
- `test_build_velocity_command_correct`: Known command -> verify velocity_commands segment

*Observation vector ordering:*
- `test_build_lin_vel_segment_position`: Verify `base_lin_vel` appears at offset 0 in the concatenated observation vector (first 3 elements)
- `test_build_ang_vel_segment_position`: Verify `base_ang_vel` appears at offset 3 (after `base_lin_vel`, before `projected_gravity`)
- `test_build_gravity_segment_position`: Verify `projected_gravity` appears at offset 6
- `test_build_vel_cmd_segment_position`: Verify `velocity_commands` appears at offset 9
- `test_build_joint_pos_segment_position`: Verify `joint_pos` appears at offset 12
- `test_build_no_est_ang_vel_at_offset_0`: When `use_estimator=False`, verify `base_ang_vel` starts at offset 0 (no lin_vel prefix)
- `test_build_no_est_joint_pos_at_offset_9`: When `use_estimator=False`, verify `joint_pos` starts at offset 9

*Multi-config:*
- `test_23dof_observation_builder`: ObservationBuilder with 23-DOF JointMapper, verify `observation_dim` and `build()` output shape
- `test_build_with_mismatched_n_observed_n_controlled`: When `n_observed=29, n_controlled=12`, verify `actions` segment is size 12 and positioned correctly after `joint_vel` (size 29)

**Run tests:** `pytest tests/test_observations.py -v`

---

## Phase 5: Safety System **[Shared]**

### Task 5.1: Implement Safety State Machine (`src/control/safety.py`)

```python
class SystemState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ESTOP = "estop"

class SafetyController:
    def __init__(self, config: ControlConfig, n_dof: int): ...

    @property
    def state(self) -> SystemState: ...

    def start(self) -> bool:
        """IDLE -> RUNNING. Returns False if transition invalid."""
    def stop(self) -> bool:
        """RUNNING -> STOPPED. Returns False if transition invalid."""
    def estop(self) -> None:
        """Trigger E-stop. Always succeeds from non-IDLE states. Latching."""
    def clear_estop(self) -> bool:
        """ESTOP -> STOPPED. Returns False if not in ESTOP state."""

    def get_damping_command(self, current_state: RobotState) -> RobotCommand:
        """Generate damping command: target_pos = current_pos, kp=0, kd=kd_damp."""

    def check_orientation(self, imu_quaternion: np.ndarray) -> tuple:
        """Check if robot orientation is safe (real robot startup check).
        Returns (is_safe: bool, message: str).
        Safe = projected gravity Z component > 0.8 (roughly < 35 deg from vertical)."""
```

**State transition rules:**
- IDLE -> RUNNING (via `start()`)
- RUNNING -> STOPPED (via `stop()`)
- RUNNING -> ESTOP (via `estop()`)
- STOPPED -> ESTOP (via `estop()`)
- ESTOP -> STOPPED (via `clear_estop()`)
- ESTOP from any state except IDLE
- IDLE is not reachable from ESTOP (must clear first, then reset)

**Damping command:** `target_pos = current_pos, kp = 0, kd = kd_damp, tau = 0`

**Tests (`tests/test_safety.py`):**
- `test_initial_state_idle`
- `test_idle_to_running`
- `test_running_to_stopped`
- `test_running_to_estop`
- `test_estop_to_stopped`
- `test_estop_latching`: After estop(), state remains ESTOP until cleared
- `test_cannot_start_from_estop`
- `test_cannot_start_from_running`
- `test_estop_from_stopped`
- `test_estop_from_idle_rejected`
- `test_damping_command_shape`
- `test_damping_command_zero_kp`
- `test_damping_command_kd_set`
- `test_damping_command_position_is_current`
- `test_orientation_check_upright`
- `test_orientation_check_tilted_safe`
- `test_orientation_check_tilted_unsafe`
- `test_orientation_check_inverted`
- `test_stopped_to_idle_transition`: Verify the path from STOPPED back to IDLE (via reset or explicit transition). Document and test whether `start()` from STOPPED is valid or requires going through IDLE first.
- `test_stop_from_idle_noop`: `stop()` from IDLE returns False (invalid transition)
- `test_clear_estop_from_non_estop`: `clear_estop()` from RUNNING/IDLE/STOPPED returns False
- `test_estop_idempotent`: Calling `estop()` 10 times from RUNNING stays in ESTOP, no error
- `test_damping_command_zero_velocity_target`: Verify `joint_velocities` is all zeros (so velocity error = current velocity)
- `test_damping_command_zero_torque`: Verify `joint_torques` (feedforward) is all zeros
- `test_orientation_boundary_angle`: Test at exactly the boundary (projected gravity Z ≈ 0.8). One side passes, the other fails.
- `test_safety_thread_safe_concurrent_transitions`: Spawn 10 threads calling `estop()` and `start()` concurrently. Verify no exceptions, state is always valid (no corrupt intermediate state). Use `threading.Barrier` to synchronize start.
- `test_safety_clamp_joint_position`: When `joint_position_limits=True`, verify `clamp_command()` clips target positions to `JOINT_LIMITS`. Input a command with positions exceeding limits, verify output is clamped.
- `test_safety_clamp_joint_velocity`: When `joint_velocity_limits=True`, verify target velocities are clipped to configured max velocity
- `test_safety_clamp_torque`: When `torque_limits=True`, verify feedforward torques are clipped to `TORQUE_LIMITS`
- `test_safety_limits_disabled`: When all limit booleans are `False`, `clamp_command()` passes through unchanged

**Note on safety limit enforcement:** The `SafetyController` must implement a `clamp_command(cmd: RobotCommand) -> RobotCommand` method that enforces joint position, velocity, and torque limits when the corresponding `SafetyConfig` booleans are enabled. This method should be called by the Controller before `robot.send_command()`. Without this, the limit config booleans are dead code.

**Run tests:** `pytest tests/test_safety.py -v`

---

## Phase 6: Policy Interfaces **[Shared]**

### Task 6.1: Create Test ONNX Model Fixtures (`tests/conftest.py` - additions)

Create helper functions that generate minimal ONNX models for testing. Requires `onnx` dev dependency.

```python
def create_isaaclab_onnx(obs_dim: int, action_dim: int, path: str) -> None:
    """Create minimal ONNX: Input 'obs' [1, obs_dim] -> Output 'action' [1, action_dim]."""

def create_beyondmimic_onnx(obs_dim: int, action_dim: int, n_joints: int,
                             path: str, metadata: dict) -> None:
    """Create minimal ONNX with 'obs' + 'time_step' inputs,
    multiple outputs, and embedded metadata."""
```

Use `onnx.helper.make_graph()` and `onnx.helper.make_model()`. The model body can be a simple constant output. The goal is to verify loading, dimensions, and metadata extraction.

### Task 6.2: Implement Policy Format Detection (`src/policy/base.py` - addition)

```python
def detect_policy_format(onnx_path: str) -> str:
    """Auto-detect: if ONNX has 'time_step' input -> 'beyondmimic', else 'isaaclab'."""
    session = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]
    if "time_step" in input_names:
        return "beyondmimic"
    return "isaaclab"
```

**Tests:**
- `test_detect_isaaclab`: IsaacLab model detected correctly
- `test_detect_beyondmimic`: BeyondMimic model detected correctly

### Task 6.3: Implement IsaacLab Policy (`src/policy/isaaclab_policy.py`) **[Shared]**

```python
class IsaacLabPolicy(PolicyInterface):
    def __init__(self, joint_mapper: JointMapper, config: ControlConfig): ...
    def load(self, path: str) -> None:
        """Create ONNX InferenceSession, validate dimensions."""
    def reset(self) -> None: ...
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Run ONNX inference, return (n_controlled,) action array."""
    @property
    def observation_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
```

**Dimension validation at load time:**
- `obs_dim` from ONNX must match `ObservationBuilder.observation_dim`
- `action_dim` from ONNX must match `joint_mapper.n_controlled`

**Tests (`tests/test_isaaclab_policy.py`):**
- `test_load_valid_policy`
- `test_load_invalid_path_raises`
- `test_load_dimension_mismatch_raises`
- `test_get_action_output_shape`
- `test_get_action_output_dtype`
- `test_reset_clears_state`
- `test_observation_dim_matches_builder`
- `test_get_action_deterministic`: Same observation input twice produces identical output (ONNX CPU inference must be deterministic)
- `test_load_corrupt_onnx_raises`: Truncated or corrupt file raises a clear error (not a segfault)
- `test_load_twice_replaces_session`: Calling `load()` a second time replaces the ONNX session cleanly (no resource leak)

### Task 6.4: Implement BeyondMimic Policy (`src/policy/beyondmimic_policy.py`) **[Shared]**

```python
class BeyondMimicPolicy(PolicyInterface):
    def __init__(self, joint_mapper: JointMapper, config: ControlConfig): ...
    def load(self, path: str) -> None:
        """Load ONNX, extract metadata (joint_names, stiffness, damping, etc.)."""
    def reset(self) -> None: ...
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Expects time_step kwarg. Returns action, stores target_q/target_dq."""
    @property
    def target_q(self) -> np.ndarray: ...
    @property
    def target_dq(self) -> np.ndarray: ...
    def load_metadata(self, onnx_path: str) -> dict: ...
```

**Key behaviors:**
- Extracts metadata via `ast.literal_eval()` (NOT `eval()`)
- When `use_onnx_metadata: true`, overrides config gains with ONNX metadata gains
- Stores `target_q`, `target_dq` from ONNX output for the control law

**Observation construction (see SPEC section 4.3 for full specification):**
Unlike IsaacLab (which uses the shared `ObservationBuilder`), BeyondMimic builds observations *inside* the policy class via a `build_observation()` method. This is because the observation structure depends on policy-specific metadata (`obs_terms`, `controlled_bodies`) only known after loading the ONNX model. Key steps:
1. Read `obs_terms` and `controlled_bodies` from ONNX metadata at load time
2. Use MuJoCo's `xpos`/`xquat` for body state (sim) or forward kinematics (real)
3. Convert all rotations to 6D representation (first 2 columns of rotation matrix)
4. Compute body-relative positions/orientations relative to anchor body
5. Cache previous ONNX outputs for motion target construction
6. Stack history frames if `obs_history_lengths` requires it
7. Concatenate into observation vector: `obs_dim = 21 + (9 × N_bodies)` (without history)

**Control law difference from IsaacLab:**
```
tau = Kp * (target_q + Ka * action - q) - Kd * (qdot - target_dq)
```
Note: uses velocity **error** `(qdot - target_dq)` rather than absolute velocity `qdot`.

**Tests (`tests/test_beyondmimic_policy.py`):**
- `test_load_valid_policy`
- `test_metadata_extraction`
- `test_get_action_with_time_step`
- `test_get_action_stores_targets`
- `test_get_action_output_shape`
- `test_metadata_overrides_config`
- `test_metadata_not_used_when_disabled`
- `test_build_observation_shape`: Observation vector has correct dimensions
- `test_build_observation_6d_rotation`: Rotation conversion produces correct 6D output
- `test_build_observation_6d_rotation_known_values`: Test with identity quaternion `[1,0,0,0]` -> first 6D column = `[1,0,0]`, second = `[0,1,0]`. Test with 90° yaw -> verify against hand-computed rotation matrix columns.
- `test_build_observation_body_relative_position`: Set known `xpos` values for anchor and two bodies, verify body positions are correctly subtracted from anchor position and rotated into anchor frame
- `test_build_observation_body_relative_orientation`: Set known `xquat` values, verify body orientations are relative to anchor orientation (quat_diff) and converted to 6D
- `test_build_observation_motion_targets`: After calling `get_action()` once (which caches `target_q`/`target_dq`), verify that the next `build_observation()` includes motion targets from the cached outputs
- `test_build_observation_history_stacking`: Set `obs_history_lengths=3`, verify observation dim is multiplied by 3 and history frames are correctly stacked (most recent first)
- `test_build_observation_full_vector_known_state`: Create a known MuJoCo state (specific joint positions, body positions, orientations), build the full observation vector, and verify every segment against hand-computed expected values
- `test_beyondmimic_reset_clears_targets`: After `get_action()`, calling `reset()` clears `target_q`, `target_dq`, and cached history
- `test_metadata_missing_required_field`: ONNX metadata missing `joint_names` raises a clear `ValueError` with field name
- `test_metadata_malformed_string`: ONNX metadata with malformed `ast.literal_eval()` input raises `ValueError` (not `SyntaxError`)
- `test_detect_format_corrupt_file`: `detect_policy_format()` with a corrupt file raises `ValueError` or `RuntimeError`

**Run tests:** `pytest tests/test_isaaclab_policy.py tests/test_beyondmimic_policy.py -v`

---

## Phase 7: MuJoCo Simulation and DDS Bridge

### Task 7.1: Create MuJoCo Scene File (`assets/robots/g1/scene.xml`) **[Shared]**

Create a minimal scene file for simulation:
- Includes the robot model via `<include file="g1_29dof.xml"/>`
- Adds a flat ground plane with appropriate friction
- Adds lighting (directional + ambient)
- Sets simulation timestep to 0.005s (200 Hz)
- Sets gravity to [0, 0, -9.81]

**Note on variant selection:** Create two scene files: `scene_29dof.xml` and `scene_23dof.xml`, each including the appropriate robot model. The `SimRobot` class selects the scene file based on `config.robot.variant`.

**Acceptance criteria:**
- `mujoco.MjModel.from_xml_path("assets/robots/g1/scene_29dof.xml")` succeeds
- Model has 29 actuators
- Stepping 1000 times without commands: robot falls (gravity works)

### Task 7.2: Implement `SimRobot` (`src/robot/sim_robot.py`) **[Shared with Metal-specific additions]**

```python
class SimRobot(RobotInterface):
    def __init__(self, config: Config):
        # Load MuJoCo model
        # Initialize DDS via compat module
        # Create DDS publishers/subscribers

    def connect(self) -> None:
        """Start DDS state publishing thread."""

    def disconnect(self) -> None:
        """Stop DDS threads, clean up."""

    def get_state(self) -> RobotState:
        """Read current state from MuJoCo sensor data."""

    def send_command(self, cmd: RobotCommand) -> None:
        """Apply command to MuJoCo ctrl array (impedance control law)."""

    def step(self) -> None:
        """Advance simulation by one policy timestep
        (sim_frequency / policy_frequency physics steps, e.g. 4).
        Uses self._lock to protect mj_data from concurrent viewer/DDS reads."""

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Reset MuJoCo to initial keyframe or specified state."""

    @property
    def n_dof(self) -> int: ...

    # [Metal-specific] Expose for viewer integration:
    @property
    def mj_model(self) -> mujoco.MjModel:
        """Access MuJoCo model (needed by viewer)."""
        return self._model

    @property
    def mj_data(self) -> mujoco.MjData:
        """Access MuJoCo data (needed by viewer)."""
        return self._data

    @property
    def lock(self) -> threading.Lock:
        """Access the data lock (needed by viewer for thread-safe sync)."""
        return self._lock
```

**Threading model:**

The control loop calls `get_state()`, `send_command()`, and `step()` synchronously from a single thread. There is no separate physics thread. The only background thread is DDS state publishing.

```
Control loop thread:  get_state() → policy → send_command() → step() → sleep
DDS publishing thread: periodically reads mj_data and publishes LowState_ (read-only)
[Metal] MuJoCo viewer: runs in main thread, calls viewer.sync() which reads mj_data
```

**Thread safety:** A `threading.Lock` (`self._lock`) protects `mj_data`:
- The control loop holds the lock during `step()` (which calls `mj_step()`)
- The DDS thread briefly acquires the lock to snapshot sensor data
- **[Metal]** The viewer's `sync()` call reads `mj_data`. The `launch_passive` context manager handles its own internal locking for the MuJoCo viewer. However, the control loop should call `step()` inside a `with self._lock:` block, and DDS publishing should also use `with self._lock:`. The viewer's `sync()` is called from the main thread between control loop ticks and is safe because `launch_passive` uses its own synchronization.

**Use `src.compat.RecurrentThread`** instead of the SDK's native version:
```python
from src.compat import RecurrentThread, resolve_network_interface, patch_unitree_threading
```

**Resolve network interface at init:**
```python
iface = resolve_network_interface(config.network.interface)
ChannelFactoryInitialize(config.network.domain_id, iface)
```

**Sensor data mapping** (compute offsets dynamically from model):
```python
num_motor = mj_model.nu                    # e.g. 29
# Motor sensors (3 per joint: position, velocity, torque)
# Positions:  sensordata[0 : num_motor]
# Velocities: sensordata[num_motor : 2 * num_motor]
# Torques:    sensordata[2 * num_motor : 3 * num_motor]
dim_motor_sensor = 3 * num_motor
# IMU (starts at dim_motor_sensor):
# Quaternion (wxyz): sensordata[dim_motor_sensor + 0 : dim_motor_sensor + 4]
# Gyroscope:         sensordata[dim_motor_sensor + 4 : dim_motor_sensor + 7]
# Accelerometer:     sensordata[dim_motor_sensor + 7 : dim_motor_sensor + 10]
```

**Command application** (impedance control):
```python
mj_data.ctrl[i] = cmd.joint_torques[i] + \
    cmd.kp[i] * (cmd.joint_positions[i] - sensor_pos[i]) + \
    cmd.kd[i] * (cmd.joint_velocities[i] - sensor_vel[i])
```

**Tests (`tests/test_sim_robot.py`):**
- `test_sim_robot_init`
- `test_sim_robot_n_dof_29`
- `test_sim_robot_get_state_shape`
- `test_sim_robot_reset`
- `test_sim_robot_step`
- `test_sim_robot_gravity`
- `test_sim_robot_damping_holds`
- `test_sim_robot_send_command_shape`
- `test_sim_robot_imu_upright`
- `test_sim_robot_connect_disconnect`
- `test_sim_robot_exposes_mj_model` **[Metal-specific test]**
- `test_sim_robot_exposes_lock` **[Metal-specific test]**
- `test_sim_robot_impedance_control_values`: Set known sensor positions/velocities (via `mj_data.sensordata`), send a command with known `kp, kd, joint_positions, joint_velocities, joint_torques`, then verify `mj_data.ctrl[i]` matches hand-computed `tau_ff + kp*(q_des - q) + kd*(dq_des - dq)` for each joint
- `test_sim_robot_sensor_mapping_correctness`: After reset, verify `get_state().joint_positions` matches `mj_data.sensordata[0:29]`, `get_state().joint_velocities` matches `sensordata[29:58]`, and IMU quaternion matches `sensordata[87:91]`
- `test_sim_robot_23dof`: Load 23-DOF scene, verify `n_dof == 23`, `get_state()` shapes are `(23,)`, and `step()` works
- `test_sim_robot_substep_count`: With `sim_frequency=200, policy_frequency=50`, verify `step()` advances MuJoCo time by exactly `4 * 0.005 = 0.02` seconds
- `test_sim_robot_base_position`: After reset, verify `get_state().base_position` matches expected initial position from MuJoCo model. Step with gravity, verify base_position Z decreases.
- `test_sim_robot_reset_custom_state`: Call `reset(initial_state=custom)`, verify joint positions match the custom state
- `test_sim_robot_dds_publish_mock`: Mock the DDS `ChannelPublisher`, call `connect()`, wait briefly, verify `publish()` was called with a valid `LowState_` message containing the correct number of motor states

**Run tests:** `pytest tests/test_sim_robot.py -v`

### Task 7.3: Validate DDS Communication (Manual Test)

1. Start `SimRobot` in one terminal
2. In another terminal, subscribe to `rt/lowstate` and print IMU data
3. Verify state messages are received at expected frequency
4. Send a `LowCmd` message and verify the robot responds
5. **Test on macOS with `lo0`** to confirm DDS loopback works

**Document results** in comments in `sim_robot.py`.

---

## Phase 8: Control Loop

### Task 8.1: Implement `Controller` (`src/control/controller.py`) **[Shared core, Metal-specific key handling]**

```python
class Controller:
    def __init__(self, robot: RobotInterface, policy: PolicyInterface,
                 safety: SafetyController, joint_mapper: JointMapper,
                 obs_builder: Optional[ObservationBuilder], config: Config,
                 logger: Optional['DataLogger'] = None):
        """
        Args:
            obs_builder: None for BeyondMimic (policy builds its own observations)
            logger: Optional data logger. If None, no logging occurs.
        """
        ...

    def start(self) -> None:
        """Start the control loop in a new thread."""

    def stop(self) -> None:
        """Stop the control loop."""

    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        """Set velocity command (thread-safe)."""

    def get_velocity_command(self) -> np.ndarray:
        """Get current velocity command (thread-safe). Returns [vx, vy, yaw_rate]."""

    def get_telemetry(self) -> dict:
        """Get latest telemetry (thread-safe, non-blocking).
        Returns dict with keys: policy_hz, sim_hz, inference_ms, loop_ms,
        base_height, base_vel, system_state, step_count."""

    def reload_policy(self, policy_path: str) -> None:
        """Load a new ONNX policy while the control loop is stopped.
        Thread-safe. Stops control loop if running, loads new policy, resets state."""

    def handle_key(self, key: str) -> None:
        """Handle keyboard input from MuJoCo viewer or CLI.
        Called from the main thread (viewer callback). Thread-safe.

        Key mappings:
            space  - Toggle start/stop
            e      - E-stop (latching)
            c      - Clear E-stop
            r      - Reset simulation
            w      - Increase Vx by 0.1 (clamped to [-1.0, 1.0])
            s      - Decrease Vx by 0.1
            a      - Increase Vy by 0.1 (clamped to [-0.5, 0.5])
            d      - Decrease Vy by 0.1
            q      - Increase yaw_rate by 0.1 (clamped to [-1.0, 1.0])
            z      - Decrease yaw_rate by 0.1
            x      - Zero all velocity commands
            n      - Load next policy from --policy-dir (if provided)
            p      - Load previous policy from --policy-dir (if provided)
        """
        if key == "space":
            if self.safety.state == SystemState.IDLE:
                self.safety.start()
                self.start()
            elif self.safety.state == SystemState.RUNNING:
                self.safety.stop()
                self.stop()
        elif key == "e":
            self.safety.estop()
        elif key == "c":
            self.safety.clear_estop()
        elif key == "r":
            self.robot.reset()
        elif key == "w":
            vc = self._velocity_command.copy()
            vc[0] = min(vc[0] + 0.1, 1.0)
            self._velocity_command = vc
        elif key == "s":
            vc = self._velocity_command.copy()
            vc[0] = max(vc[0] - 0.1, -1.0)
            self._velocity_command = vc
        elif key == "a":
            vc = self._velocity_command.copy()
            vc[1] = min(vc[1] + 0.1, 0.5)
            self._velocity_command = vc
        elif key == "d":
            vc = self._velocity_command.copy()
            vc[1] = max(vc[1] - 0.1, -0.5)
            self._velocity_command = vc
        elif key == "q":
            vc = self._velocity_command.copy()
            vc[2] = min(vc[2] + 0.1, 1.0)
            self._velocity_command = vc
        elif key == "z":
            vc = self._velocity_command.copy()
            vc[2] = max(vc[2] - 0.1, -1.0)
            self._velocity_command = vc
        elif key == "x":
            self._velocity_command = np.zeros(3)
```

**Velocity command ranges (from SPEC section 6.5):**

| Command | Range | Step Size | Key Up / Key Down |
|---------|-------|-----------|-------------------|
| Vx (forward/back) | [-1.0, 1.0] m/s | 0.1 | W / S |
| Vy (left/right) | [-0.5, 0.5] m/s | 0.1 | A / D |
| Yaw rate | [-1.0, 1.0] rad/s | 0.1 | Q / Z |
| Zero all | — | — | X |

**Control loop pseudocode:**

```python
def _control_loop(self):
    last_action = np.zeros(self.joint_mapper.n_controlled)
    step_count = 0
    while self._running:
        loop_start = time.perf_counter()

        # 1. Check E-stop
        if self.safety.state == SystemState.ESTOP:
            state = self.robot.get_state()
            cmd = self.safety.get_damping_command(state)
            self.robot.send_command(cmd)
            self.robot.step()
            self._sleep_until_next_tick(loop_start)
            continue

        # 2. Skip if not RUNNING
        if self.safety.state != SystemState.RUNNING:
            self.robot.step()  # Keep sim advancing even when stopped
            self._sleep_until_next_tick(loop_start)
            continue

        # 3. Get robot state
        state = self.robot.get_state()

        # 4. Build observation and run policy inference
        inference_start = time.perf_counter()
        if self.obs_builder is not None:
            # IsaacLab: external observation builder
            obs = self.obs_builder.build(state, last_action, self._velocity_command)
            action = self.policy.get_action(obs)
        else:
            # BeyondMimic: policy builds its own observations internally
            action = self.policy.get_action(state, time_step=self._time_step)
        inference_time = time.perf_counter() - inference_start

        # 5. Build command (PD control law)
        cmd = self._build_command(state, action)

        # 6. Send command
        self.robot.send_command(cmd)

        # 7. Step simulation (no-op for real robot)
        self.robot.step()

        # 8. Store for next iteration
        last_action = action.copy()
        step_count += 1

        # 9. Log (if logger provided)
        if self._logger is not None:
            self._logger.log_step(...)

        # 10. Update telemetry
        self._update_telemetry(state, inference_time, loop_start, step_count)

        # 11. Check auto-termination (headless evals)
        if self._max_steps and step_count >= self._max_steps:
            self.stop()
            break

        # 12. Sleep to maintain frequency
        self._sleep_until_next_tick(loop_start)
```

**Command building (`_build_command`):**

For controlled joints (IsaacLab):
```
target_pos[i] = q_home[i] + Ka[i] * action[i]
kp[i] = config.kp[i]
kd[i] = config.kd[i]
dq_target[i] = 0
tau[i] = 0
```

For controlled joints (BeyondMimic):
```
target_pos[i] = target_q[i] + Ka[i] * action[i]
kp[i] = config.kp[i] (or metadata kp if use_onnx_metadata)
kd[i] = config.kd[i] (or metadata kd)
dq_target[i] = target_dq[i]
tau[i] = 0
```

For non-controlled joints (damping mode):
```
target_pos[i] = current_pos[i]
kp[i] = 0
kd[i] = config.kd_damp
dq_target[i] = 0
tau[i] = 0
```

**BeyondMimic end-of-trajectory handling:**
When `time_step` exceeds trajectory length:
1. Capture final joint positions
2. Linearly interpolate from final positions to `q_home` over 2 seconds (100 steps at 50 Hz)
3. Enter STOPPED state after interpolation completes

**Stdout output:** The controller should print periodic status lines to stdout at 1 Hz:
```
[controller] state=RUNNING step=150 policy_hz=49.8 vel_cmd=[0.3, 0.0, 0.0]
```
This is important for headless mode where there's no viewer.

**Tests (`tests/test_controller.py`):**

*Command building (value-level):*
- `test_controller_init`
- `test_build_command_isaaclab_values`: With `q_home=0.5, Ka=0.3, action=1.0`, verify `target_pos = 0.5 + 0.3 * 1.0 = 0.8`, `kp = config.kp`, `kd = config.kd`, `dq_target = 0`, `tau = 0`
- `test_build_command_beyondmimic_values`: With `target_q=0.2, Ka=0.3, action=1.0, target_dq=0.5`, verify `target_pos = 0.2 + 0.3 * 1.0 = 0.5`, `dq_target = 0.5`, `kp = metadata_kp`, `kd = metadata_kd`
- `test_build_command_damping`: Non-controlled joints get `target_pos=current_pos, kp=0, kd=kd_damp, dq_target=0, tau=0`

*Safety integration:*
- `test_estop_sends_damping`: In ESTOP state, verify damping command is sent
- `test_control_loop_exception_triggers_estop`: Mock policy to raise `RuntimeError` during `get_action()`. Verify controller enters ESTOP state (not crash).

*Velocity command:*
- `test_velocity_command_thread_safe`: Set velocity from one thread, read from another, no corruption
- `test_telemetry_updates`: After running, telemetry dict contains expected keys with reasonable values

*Key handling (complete coverage):*
- `test_handle_key_space_toggles`: Space from IDLE -> RUNNING, Space from RUNNING -> STOPPED
- `test_handle_key_estop`: 'e' triggers E-stop, verify state == ESTOP
- `test_handle_key_clear_estop`: 'c' from ESTOP -> STOPPED
- `test_handle_key_reset`: 'r' calls `robot.reset()` (mock and verify)
- `test_handle_key_velocity_wasd`: W increases vx by 0.1, S decreases, A increases vy, D decreases
- `test_handle_key_velocity_clamps`: After 20 W presses, vx is clamped to 1.0 (not 2.0)
- `test_handle_key_qz_yaw`: Q increases yaw by 0.1, Z decreases, clamped to [-1.0, 1.0]
- `test_handle_key_x_zeros_velocity`: X zeros all velocity components
- `test_handle_key_next_policy`: 'n' calls `reload_policy()` with next policy from `--policy-dir` (mock and verify)
- `test_handle_key_prev_policy`: 'p' calls `reload_policy()` with previous policy
- `test_handle_key_unknown_noop`: Unknown key (e.g., 'j') does nothing, no error

*Policy reloading:*
- `test_reload_policy_while_stopped`: `reload_policy("new.onnx")` loads successfully, resets state
- `test_reload_policy_while_running`: If RUNNING, `reload_policy()` stops first, loads, resets (verify stop -> load -> reset sequence)
- `test_reload_policy_invalid_path`: Invalid path raises error, original policy is preserved

*BeyondMimic trajectory:*
- `test_beyondmimic_end_of_trajectory`: When `time_step` exceeds trajectory length, verify: (1) final positions captured, (2) linear interpolation to `q_home` over 2s/100 steps, (3) state transitions to STOPPED after interpolation
- `test_auto_termination_max_steps`: Set `_max_steps=10`, run loop, verify it stops after exactly 10 steps

*Integration:*
- `test_control_loop_lifecycle`: Create controller with mock robot/policy, `start()` -> verify running -> `stop()` -> verify stopped. No exceptions.
- `test_control_loop_stopped_still_steps`: In STOPPED state, verify `robot.step()` is still called (sim keeps advancing) but no command is sent

**Run tests:** `pytest tests/test_controller.py -v`

---

## Phase 9: MuJoCo Viewer Integration **[Metal-specific]**

### Task 9.1: Implement Viewer Launch (`src/main.py`)

Use `mujoco.viewer.launch_passive()` to create a non-blocking viewer in the main thread.

```python
import mujoco.viewer

# GLFW key constants (subset). Full list: https://www.glfw.org/docs/latest/group__keys.html
# These match ASCII for letter keys and common keys.
GLFW_KEY_MAP = {
    32: "space",     # GLFW_KEY_SPACE
    67: "c",         # GLFW_KEY_C
    69: "e",         # GLFW_KEY_E
    82: "r",         # GLFW_KEY_R
    87: "w",         # GLFW_KEY_W
    83: "s",         # GLFW_KEY_S
    65: "a",         # GLFW_KEY_A
    68: "d",         # GLFW_KEY_D
    81: "q",         # GLFW_KEY_Q
    90: "z",         # GLFW_KEY_Z
    88: "x",         # GLFW_KEY_X
}


def run_with_viewer(sim_robot: SimRobot, controller: Controller):
    """Run simulation with interactive MuJoCo viewer.

    The viewer runs in the main thread. The control loop runs in a
    background thread (started by controller.start()). MuJoCo's
    launch_passive handles its own thread-safety for rendering.

    Key callback fires on key PRESS only (not repeat or release).
    """

    def key_callback(keycode):
        """Called by MuJoCo viewer on key press. Runs in main thread."""
        key = GLFW_KEY_MAP.get(keycode)
        if key:
            controller.handle_key(key)

    with mujoco.viewer.launch_passive(
        sim_robot.mj_model,
        sim_robot.mj_data,
        key_callback=key_callback,
    ) as viewer:
        controller.start()
        try:
            while viewer.is_running():
                # sync() copies mj_data into the viewer's internal buffer
                # for rendering. This is thread-safe with launch_passive.
                viewer.sync()
                time.sleep(1.0 / 60.0)  # ~60 FPS viewer update
        except KeyboardInterrupt:
            pass
        finally:
            controller.stop()
```

**Note on GLFW key callbacks:** `mujoco.viewer.launch_passive()` accepts a `key_callback` parameter (since MuJoCo 3.0+). The callback receives a single `int` argument which is a GLFW keycode. GLFW keycodes for printable ASCII characters match their ASCII values (A=65, B=66, etc.). The callback fires once per key press (not on repeat or release). Shift state is ignored — 'e' and 'E' both fire keycode 69.

**Note on WASD conflict with MuJoCo viewer:** The MuJoCo viewer does NOT use WASD for camera control (it uses mouse for orbit/pan/zoom). WASD keys are free for application use.

**Tests (`tests/test_viewer.py`):**
- `test_glfw_key_map_values`: Verify `GLFW_KEY_MAP[32] == "space"`, `[69] == "e"`, `[65] == "a"`, etc. Catches typos in keycode constants.
- `test_key_callback_dispatches_to_controller`: Mock `controller.handle_key`, simulate `key_callback(32)` (space), verify `controller.handle_key("space")` was called
- `test_key_callback_unmapped_ignored`: Simulate `key_callback(999)` (unmapped code), verify `controller.handle_key` is NOT called

**Acceptance criteria (manual):**
- Viewer opens and shows the G1 robot
- Space bar starts/stops the policy
- E key triggers E-stop
- R key resets the simulation
- WASD/QZ adjust velocity commands
- X zeros velocity commands
- Closing the viewer window cleanly shuts down the control loop

### Task 9.2: Implement Headless Runner (`src/main.py`) **[Metal-specific]**

```python
def run_headless(sim_robot: SimRobot, controller: Controller,
                 duration: Optional[float] = None, max_steps: Optional[int] = None):
    """Run simulation without viewer (for server evals).

    Termination conditions (first one wins):
        1. Ctrl+C (KeyboardInterrupt)
        2. --duration seconds elapsed
        3. --steps policy steps completed
        4. BeyondMimic trajectory ends (controller auto-stops)

    Prints periodic status to stdout at 1 Hz for monitoring.

    Args:
        duration: Auto-terminate after this many seconds (None = no limit)
        max_steps: Auto-terminate after this many policy steps (None = no limit)
    """
    if max_steps is not None:
        controller._max_steps = max_steps

    controller.start()
    # Auto-start the policy (no viewer to press Space)
    controller.safety.start()

    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)

            # Check duration limit
            if duration is not None and (time.time() - start_time) >= duration:
                print(f"[headless] Duration limit reached ({duration}s). Stopping.")
                break

            # Check if controller self-stopped (e.g., BeyondMimic trajectory end)
            if not controller.is_running:
                print("[headless] Controller stopped (trajectory end or error).")
                break

    except KeyboardInterrupt:
        print("\n[headless] Ctrl+C received. Stopping.")
    finally:
        controller.stop()
```

**Tests (`tests/test_viewer.py` - additions):**
- `test_run_headless_starts_policy`: `run_headless()` calls `controller.start()` and `controller.safety.start()` (auto-starts the policy)
- `test_run_headless_duration_termination`: With `duration=0.5`, verify run completes within 0.5-1.0 seconds (use mock robot/policy for speed)
- `test_run_headless_step_termination`: With `max_steps=10`, verify controller ran exactly 10 steps
- `test_run_headless_trajectory_end`: Mock `controller.is_running` to return False after a delay, verify `run_headless()` exits cleanly

**Acceptance criteria (manual):**
- `--headless` runs without any display or viewer
- `--headless --duration 10` auto-terminates after 10 seconds
- `--headless --steps 500` auto-terminates after 500 policy steps
- Periodic stdout status lines appear at ~1 Hz
- Logs are generated normally

### Task 9.3: Display Status Overlay (Optional)

MuJoCo viewer supports text overlays. Display system state and velocity command.

This task is optional and can be deferred. If implemented, use the `mjvFigure` or viewport overlay API to show:
- System state (IDLE/RUNNING/STOPPED/ESTOP)
- Current velocity command [vx, vy, yaw]
- Policy Hz

**Note:** The exact MuJoCo Python overlay API varies by version. The engineer should consult the MuJoCo Python bindings documentation for the `viewer.user_scn` API or the `mjv_` overlay functions. If overlay proves too complex, printing to stdout (which the controller already does) is sufficient.

---

## Phase 10: Logging System **[Shared]**

### Task 10.1: Implement Data Logger (`src/logging/logger.py`)

```python
class DataLogger:
    def __init__(self, config: LoggingConfig, run_name: str, log_dir: str): ...

    def start(self) -> None:
        """Create log directory and open files."""

    def log_step(self, timestamp: float, robot_state: RobotState,
                 observation: np.ndarray, action: np.ndarray,
                 command: RobotCommand, system_state: SystemState,
                 velocity_command: np.ndarray,
                 timing: dict) -> None:
        """Log one timestep. Called from control loop."""

    def log_event(self, event_type: str, data: dict) -> None:
        """Log discrete event (start, stop, e-stop, etc.)."""

    def stop(self) -> None:
        """Flush and close files. Print summary statistics."""
```

**File structure:**
```
logs/{timestamp}_{mode}_{policy_name}/
├── metadata.yaml      # Run configuration snapshot
├── data.hdf5          # Compressed time-series data (or data.npz)
└── events.json        # Discrete events
```

**Dual format support:** Both HDF5 (`.hdf5`) and compressed NumPy (`.npz`) formats must be supported, selectable via `logging.format` in config. HDF5 is the default. The `DataLogger` should use a backend abstraction (or simple if/else) so both formats share the same `log_step()` / `stop()` interface. The `LogReplay` class must auto-detect the format from the file extension.

**Dataset structure (same schema for both formats):**
```
/timestamps     float64  (N,)
/joint_pos      float32  (N, n_dof)
/joint_vel      float32  (N, n_dof)
/joint_torques  float32  (N, n_dof)
/imu_quat       float32  (N, 4)
/imu_gyro       float32  (N, 3)
/imu_accel      float32  (N, 3)
/base_pos       float32  (N, 3)
/base_vel       float32  (N, 3)
/observations   float32  (N, obs_dim)
/actions        float32  (N, action_dim)
/cmd_pos        float32  (N, n_dof)
/cmd_kp         float32  (N, n_dof)
/cmd_kd         float32  (N, n_dof)
/system_state   int32    (N,)
/vel_cmd        float32  (N, 3)
/inference_ms   float32  (N,)
/loop_ms        float32  (N,)
```

For HDF5: datasets are gzip-compressed, support partial reads. For NPZ: `np.savez_compressed()` with the same key names.

**Performance:** Buffer 100 steps, then flush. Alternatively, use a separate logging thread with a queue to prevent I/O from blocking the control loop.

**Tests (`tests/test_logger.py`):**
- `test_logger_creates_directory`
- `test_logger_writes_metadata`
- `test_logger_writes_data_hdf5`: Log 100 steps, verify HDF5 shapes
- `test_logger_writes_data_npz`: Log 100 steps, verify NPZ shapes
- `test_logger_writes_events`
- `test_logger_compression`: HDF5 is gzip-compressed
- `test_logger_roundtrip_hdf5`: Write and read back via HDF5, verify values
- `test_logger_roundtrip_npz`: Write and read back via NPZ, verify values
- `test_logger_stop_prints_summary`
- `test_logger_handles_empty_run`
- `test_logger_log_event`: Call `log_event("estop", {"reason": "orientation"})`, verify event appears in `events.json` with timestamp
- `test_logger_metadata_contains_config`: Verify `metadata.yaml` contains the full config snapshot (robot variant, policy format, control gains, etc.)
- `test_logger_nonblocking`: Log 100 steps, measure wall time. Verify logging overhead is <1ms per step (no I/O blocking on the control loop thread).

### Task 10.2: Implement Log Replay (`src/logging/replay.py` and `scripts/replay_log.py`)

```python
class LogReplay:
    def __init__(self, log_dir: str): ...
    def load(self) -> None: ...
    @property
    def metadata(self) -> dict: ...
    @property
    def duration(self) -> float: ...
    @property
    def n_steps(self) -> int: ...
    def get_state_at(self, step: int) -> RobotState: ...
    def get_observation_at(self, step: int) -> np.ndarray: ...
    def get_action_at(self, step: int) -> np.ndarray: ...
    def to_csv(self, output_path: str) -> None: ...
    def summary(self) -> str: ...
```

**`scripts/replay_log.py`:**
```bash
# Standalone script (no Docker required)
python scripts/replay_log.py logs/<run>/ [--format csv] [--output file]
```

**Note:** `--visualize` flag for Viser replay is Docker-plan only. The Metal plan can add MuJoCo viewer replay as future work.

**Tests (`tests/test_logger.py` - additions):**
- `test_replay_load`
- `test_replay_metadata`
- `test_replay_state_at`
- `test_replay_observation_at`: `get_observation_at(step)` returns correct observation array matching what was logged
- `test_replay_action_at`: `get_action_at(step)` returns correct action array
- `test_replay_to_csv`
- `test_replay_summary`
- `test_replay_auto_detect_format_hdf5`: `LogReplay` with an HDF5 log directory auto-detects and loads HDF5 format
- `test_replay_auto_detect_format_npz`: `LogReplay` with an NPZ log directory auto-detects and loads NPZ format

**Run tests:** `pytest tests/test_logger.py -v`

---

## Phase 11: Real Robot Interface **[Shared]**

**SAFETY WARNING:** All development should happen in simulation first. Real robot testing should start with the robot hanging from a support harness.

### Task 11.1: Implement `RealRobot` (`src/robot/real_robot.py`)

```python
class RealRobot(RobotInterface):
    def __init__(self, config: Config): ...

    def connect(self) -> None:
        """Initialize DDS on the specified network interface.
        Subscribe to rt/lowstate. Prepare to publish rt/lowcmd.
        Verify connection by waiting for first state message (timeout: 5s)."""

    def disconnect(self) -> None:
        """Stop publishing, clean up DDS."""

    def get_state(self) -> RobotState:
        """Return latest state from DDS subscription. Thread-safe."""

    def send_command(self, cmd: RobotCommand) -> None:
        """Publish LowCmd to rt/lowcmd via DDS.
        Sets motor_cmd[i].mode = 0x01 (PMSM servo mode).
        Computes CRC32 before publishing (required by robot firmware)."""

    def step(self) -> None:
        """No-op for real robot."""

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Log warning. Cannot reset physical robot."""

    @property
    def n_dof(self) -> int: ...
```

**Key implementation details:**

1. **DDS initialization:**
   ```python
   ChannelFactoryInitialize(domain_id=0, interface=config.network.interface)
   ```
   Note: real robot uses `domain_id=0` (Unitree default), NOT the sim default of 1.

2. **IDL types:** Use `unitree_hg` IDL types (required for G1). Import from:
   ```python
   from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
   from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
   ```

3. **CRC calculation:** The `LowCmd_` message requires CRC32. Use the SDK's CRC utility:
   ```python
   from unitree_sdk2py.utils.crc import CRC
   crc = CRC()
   cmd.crc = crc.Crc32(cmd)
   ```
   The engineer must verify whether the Python SDK handles CRC automatically or requires manual computation. Check `unitree_sdk2py.utils.crc`.

4. **State subscription:** Store latest state in a thread-safe buffer (Lock + copy). State callback runs on DDS internal thread.

5. **Startup checks:**
   - Verify DDS connection (receive at least one state message within 5s timeout)
   - Check orientation via `safety.check_orientation()` before allowing policy start
   - Print IMU data and orientation check result to console

6. **Communication monitoring:** Track time since last state message. If timeout exceeded (100ms), trigger E-stop.

**Tests (`tests/test_real_robot.py`):**

*Basic lifecycle (no DDS required):*
- `test_real_robot_init_without_dds`: Init succeeds, `connect()` fails gracefully without robot
- `test_real_robot_step_is_noop`: `step()` returns immediately, no side effects
- `test_real_robot_reset_logs_warning`: `reset()` logs a warning (cannot reset physical robot)
- `test_real_robot_n_dof`: `n_dof` returns 29 for 29-DOF config, 23 for 23-DOF config

*Command construction (mock DDS):*
- `test_real_robot_send_command_motor_mode`: Mock DDS publisher, send command, verify `motor_cmd[i].mode == 0x01` for all joints
- `test_real_robot_send_command_field_mapping`: Mock DDS publisher, send known `RobotCommand`, verify `LowCmd_` fields are populated correctly (`q`, `dq`, `tau`, `kp`, `kd` per motor)
- `test_real_robot_send_command_crc`: Mock DDS publisher, verify CRC32 is computed and set on `LowCmd_` before publishing

*State subscription (mock DDS):*
- `test_real_robot_get_state_mapping`: Create a mock `LowState_` message with known motor positions/velocities and IMU data, invoke the state callback, verify `get_state()` returns correctly mapped `RobotState`
- `test_real_robot_get_state_thread_safe`: Invoke state callback from one thread while calling `get_state()` from another — no corruption or exception
- `test_real_robot_get_state_nan_base`: Verify `get_state().base_position` and `.base_velocity` are NaN (not available on real robot)

*Communication monitoring:*
- `test_real_robot_watchdog_timeout`: Simulate no state messages for >100ms, verify E-stop is triggered (pass a mock `SafetyController` and verify `estop()` called)
- `test_real_robot_connect_timeout`: Mock DDS subscriber that never delivers a message, verify `connect()` raises `TimeoutError` after 5 seconds

*Configuration:*
- `test_real_robot_domain_id`: Verify `ChannelFactoryInitialize` is called with `domain_id=0` (real robot default)
- `test_real_robot_dds_topic_names`: Verify subscriber listens on `rt/lowstate` and publisher targets `rt/lowcmd`

**Run tests:** `pytest tests/test_real_robot.py -v`

---

## Phase 12: CLI Entry Point **[Metal-specific]**

### Task 12.1: Implement Main Entry Point (`src/main.py`)

```python
#!/usr/bin/env python3
"""Unitree G1 Deployment Stack - Main Entry Point."""

# MUST be first: patch SDK threading for macOS compatibility
from src.compat import patch_unitree_threading
patch_unitree_threading()

import argparse
import time
from pathlib import Path
from datetime import datetime

from src.config import (Config, load_config, apply_cli_overrides,
                        G1_29DOF_JOINTS, G1_23DOF_JOINTS)
from src.robot.sim_robot import SimRobot
from src.robot.real_robot import RealRobot
from src.policy.base import detect_policy_format
from src.policy.isaaclab_policy import IsaacLabPolicy
from src.policy.beyondmimic_policy import BeyondMimicPolicy
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from src.control.safety import SafetyController
from src.control.controller import Controller
from src.logging.logger import DataLogger


def add_common_args(parser):
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policy", required=True, help="ONNX policy file")
    parser.add_argument("--policy-dir", default="policies/",
                       help="Directory of ONNX files (future: runtime switching)")
    parser.add_argument("--robot", default=None, help="Robot variant override")
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--log-dir", default="logs/")
    parser.add_argument("--no-est", action="store_true",
                       help="Override policy.use_estimator to false (omit base_lin_vel)")


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 Deployment Stack")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Sim mode
    sim_parser = subparsers.add_parser("sim", help="Simulation mode")
    add_common_args(sim_parser)
    sim_parser.add_argument("--headless", action="store_true",
                           help="Run without viewer (for server evals)")
    sim_parser.add_argument("--duration", type=float, default=None,
                           help="Auto-stop after N seconds (headless only)")
    sim_parser.add_argument("--steps", type=int, default=None,
                           help="Auto-stop after N policy steps (headless only)")

    # Real mode
    real_parser = subparsers.add_parser("real", help="Real robot mode")
    add_common_args(real_parser)
    real_parser.add_argument("--interface", required=True,
                           help="Network interface (e.g., eth0)")

    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    # Override domain_id defaults: sim=1, real=0
    if args.domain_id is not None:
        config.network.domain_id = args.domain_id
    elif args.mode == "real":
        config.network.domain_id = 0

    # Override interface for real mode
    if args.mode == "real":
        config.network.interface = args.interface

    # Resolve variant
    variant = config.robot.variant

    # Create robot
    if args.mode == "sim":
        robot = SimRobot(config)
    else:
        robot = RealRobot(config)

    # Resolve policy format
    policy_format = config.policy.format or detect_policy_format(args.policy)

    # Create joint mapper
    robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS
    joint_mapper = JointMapper(
        robot_joints=robot_joints,
        observed_joints=config.policy.observed_joints,
        controlled_joints=config.policy.controlled_joints,
    )

    # Create policy
    if policy_format == "isaaclab":
        policy = IsaacLabPolicy(joint_mapper, config.control)
    else:
        policy = BeyondMimicPolicy(joint_mapper, config.control)
    policy.load(args.policy)

    # Create observation builder (IsaacLab only)
    obs_builder = None
    if policy_format == "isaaclab":
        # Config is primary; --no-est CLI flag overrides to False
        use_estimator = config.policy.use_estimator
        if getattr(args, 'no_est', False):
            use_estimator = False
        obs_builder = ObservationBuilder(joint_mapper, config.control,
                                         use_estimator=use_estimator)
        assert policy.observation_dim == obs_builder.observation_dim, \
            f"Policy obs_dim={policy.observation_dim} != builder {obs_builder.observation_dim}"

    # Create safety controller
    safety = SafetyController(config.control, robot.n_dof)

    # Create logger
    policy_name = Path(args.policy).stem
    run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{args.mode}_{policy_name}"
    logger = DataLogger(config.logging, run_name, args.log_dir)

    # Create controller
    controller = Controller(robot, policy, safety, joint_mapper,
                           obs_builder, config, logger=logger)

    # Connect
    robot.connect()
    logger.start()

    # Run
    try:
        if args.mode == "sim" and not getattr(args, 'headless', False):
            from src.main import run_with_viewer
            run_with_viewer(robot, controller)
        else:
            from src.main import run_headless
            run_headless(robot, controller,
                        duration=getattr(args, 'duration', None),
                        max_steps=getattr(args, 'steps', None))
    finally:
        robot.disconnect()
        logger.stop()


if __name__ == "__main__":
    main()
```

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | path | `configs/default.yaml` | Configuration file |
| `--policy` | path | required | ONNX policy file |
| `--policy-dir` | path | `policies/` | Directory of ONNX files (future: runtime switching) |
| `--robot` | str | from config | Robot variant override |
| `--headless` | flag | false | No viewer (sim mode only) |
| `--duration` | float | None | Auto-stop after N seconds (headless only) |
| `--steps` | int | None | Auto-stop after N policy steps (headless only) |
| `--interface` | str | — | Network interface (real only, required) |
| `--domain-id` | int | 1 (sim) / 0 (real) | DDS domain ID |
| `--log-dir` | path | `logs/` | Log output directory |
| `--no-est` | flag | false | Override `policy.use_estimator` to false (omit `base_lin_vel`) |

**Tests (`tests/test_main.py`):**

*Argument parsing:*
- `test_parse_sim_args`: `parse_args(["sim", "--policy", "test.onnx"])` succeeds with `mode="sim"`, `policy="test.onnx"`, `headless=False`
- `test_parse_real_args`: `parse_args(["real", "--policy", "test.onnx", "--interface", "eth0"])` succeeds with `mode="real"`, `interface="eth0"`
- `test_missing_policy_errors`: `parse_args(["sim"])` without `--policy` raises `SystemExit` (argparse required arg)
- `test_real_requires_interface`: `parse_args(["real", "--policy", "test.onnx"])` without `--interface` raises `SystemExit`
- `test_headless_args`: `parse_args(["sim", "--policy", "p.onnx", "--headless", "--duration", "10", "--steps", "500"])` parses `duration=10.0, steps=500`
- `test_no_est_flag`: `parse_args(["sim", "--policy", "p.onnx", "--no-est"])` sets `no_est=True`
- `test_no_est_default_false`: Without `--no-est`, verify `no_est=False`
- `test_no_est_cli_overrides_config`: Config has `policy.use_estimator: true`, CLI passes `--no-est` -> `use_estimator` resolves to `False`
- `test_use_estimator_from_config`: Config has `policy.use_estimator: false`, no CLI flag -> `use_estimator` resolves to `False`
- `test_use_estimator_default_true`: Config omits `use_estimator` -> defaults to `True`

*Config integration:*
- `test_sim_default_domain_id`: In sim mode without `--domain-id`, verify `config.network.domain_id == 1`
- `test_real_default_domain_id`: In real mode without `--domain-id`, verify `config.network.domain_id == 0`
- `test_explicit_domain_id_override`: `--domain-id 5` overrides both sim and real defaults
- `test_apply_cli_overrides`: `--robot g1_23dof` overrides config variant

*Component wiring:*
- `test_variant_resolution_29dof`: `variant="g1_29dof"` selects `G1_29DOF_JOINTS` (29 joints)
- `test_variant_resolution_23dof`: `variant="g1_23dof"` selects `G1_23DOF_JOINTS` (23 joints)
- `test_policy_format_auto_detection`: Mock `detect_policy_format()`, verify it is called when `config.policy.format` is None
- `test_model_path_not_found`: `--policy nonexistent.onnx` raises `FileNotFoundError` during `policy.load()`

**Run tests:** `pytest tests/test_main.py -v`

### Task 12.2: Implement Shell Scripts

**`scripts/run_sim.sh`:**
```bash
#!/bin/bash
python -m src.main sim --config configs/default.yaml "$@"
```

**`scripts/run_real.sh`:**
```bash
#!/bin/bash
python -m src.main real --config configs/default.yaml "$@"
```

**`scripts/run_eval.sh`:** (headless batch eval)
```bash
#!/bin/bash
python -m src.main sim --headless --config configs/default.yaml "$@"
# Example: ./scripts/run_eval.sh --policy policies/walk.onnx --duration 30
```

---

## Phase 13: Integration Testing

### Task 13.1: End-to-End Sim Test (IsaacLab) — macOS with Viewer

1. `python -m src.main sim --policy policies/<policy>.onnx`
2. Viewer opens, robot visible
3. Press Space to start
4. Press W to command forward velocity
5. Robot walks in simulation
6. Press X to zero velocity, press Space to stop
7. Press E for E-stop, verify damping
8. Press C to clear, verify STOPPED
9. Close viewer, verify clean shutdown
10. Verify logs in `logs/`

### Task 13.2: End-to-End Headless Eval — Linux Server

1. `python -m src.main sim --headless --policy policies/<policy>.onnx --duration 10`
2. Verify stdout shows periodic status lines
3. Verify auto-termination after 10 seconds
4. Verify logs contain expected data
5. Verify timing shows ~50 Hz policy

### Task 13.3: End-to-End Headless Eval — Steps Termination

1. `python -m src.main sim --headless --policy policies/<policy>.onnx --steps 500`
2. Verify termination after 500 steps (~10 seconds at 50 Hz)
3. Verify log has exactly 500 entries

### Task 13.4: BeyondMimic Simulation Test

1. `python -m src.main sim --policy policies/<beyondmimic_policy>.onnx`
2. Press Space to start
3. Robot performs motion trajectory
4. At trajectory end: smooth interpolation to home position
5. System enters STOPPED state automatically

### Task 13.5: 23-DOF Smoke Test

1. `python -m src.main sim --robot g1_23dof --config configs/g1_23dof.yaml --policy policies/<23dof_policy>.onnx`
2. Verify model loads with 23 joints
3. If no 23-DOF policy, verify standing in damping mode

### Task 13.6: Automated Headless Integration Test

**File:** `tests/test_integration.py`

This is a fully automated end-to-end test that runs the system in headless mode with mock/test components. No manual intervention required.

```python
def test_headless_sim_isaaclab_100_steps():
    """Full pipeline: config -> SimRobot -> IsaacLabPolicy -> Controller -> Logger.
    Run 100 steps in headless mode, verify no crash and logs are produced."""

def test_headless_sim_policy_reload():
    """Start with one policy, reload to another mid-run, verify switch occurs."""

def test_headless_sim_estop_recovery():
    """Start -> E-stop -> clear -> resume -> stop. Verify state transitions and logs."""

def test_headless_sim_23dof_smoke():
    """Run 10 steps with 23-DOF model. Verify no crash."""

def test_headless_performance_50hz():
    """Run 200 steps, verify mean loop time < 20ms (50 Hz target).
    Skip if running in CI without adequate CPU (use pytest.mark.slow)."""
```

**Run tests:** `pytest tests/test_integration.py -v --timeout=60`

### Task 13.7: Comprehensive Unit Test Run

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Target:** 100% of unit tests pass. Coverage >80%.

**Test count expectations:**

| Module | Approximate Count | Key Coverage |
|--------|------------------|-------------|
| `test_config.py` | 21-24 | Constants, data structures, config validation, merge semantics |
| `test_compat.py` | 13-15 | RecurrentThread, network resolution, SDK patching |
| `test_joint_mapper.py` | 21-24 | Mapping, reordering, roundtrips, 23-DOF, error paths |
| `test_observations.py` | 19-22 | Gravity, velocity transforms, segment positions, 23-DOF |
| `test_safety.py` | 26-30 | State machine, damping, orientation, thread safety, limit clamping |
| `test_isaaclab_policy.py` | 10-12 | Load, inference, determinism, corruption |
| `test_beyondmimic_policy.py` | 20-23 | Obs construction values, 6D rotation, history, metadata errors |
| `test_logger.py` | 20-23 | Both formats, roundtrips, events, auto-detect, replay |
| `test_sim_robot.py` | 19-22 | Impedance values, sensor mapping, 23-DOF, substeps, DDS mock |
| `test_controller.py` | 26-30 | Command values, all keys, reload, trajectory end, lifecycle |
| `test_viewer.py` | 7-9 | GLFW map, key dispatch, headless start/terminate |
| `test_real_robot.py` | 14-16 | Command construction, state mapping, watchdog, CRC, thread safety |
| `test_main.py` | 12-14 | Arg parsing, domain ID, variant resolution, wiring |
| `test_integration.py` | 5-7 | Headless E2E, policy reload, estop recovery, performance |
| **Total** | **~233-271** |

---

## Phase Summary and Execution Order

See also the **Execution Guide** section at the top of this document for the dependency graph and parallelization strategy.

| Phase | Description | Tasks | Depends On | Parallel With |
|-------|-------------|-------|------------|---------------|
| 0 | Environment Validation [Metal] | 3 | — | — |
| 1 | Project Scaffolding | 4 | Phase 0 | — |
| 2 | Core Data Structures & Config [Shared] | 6 | Phase 1 | Phase 3 |
| 3 | Cross-Platform Compat [Shared] | 2 | Phase 1 | Phase 2 |
| 4 | Joint Mapping & Observations [Shared] | 2 | Phase 2 | Phases 5, 7, 10 |
| 5 | Safety System [Shared] | 1 | Phase 2 | Phases 4, 6, 7, 10 |
| 6 | Policy Interfaces [Shared] | 4 | Phases 2, 4 | Phases 5, 7, 10 |
| 7 | MuJoCo Sim & DDS Bridge | 3 | Phases 2, 3 | Phases 4, 5, 6, 10 |
| 8 | Control Loop | 1 | Phases 5, 6, 7 | Phase 10 |
| 9 | MuJoCo Viewer Integration [Metal] | 3 | Phases 7, 8 | Phases 10, 11 |
| 10 | Logging System [Shared] | 2 | Phase 2 | Phases 4-9, 11 |
| 11 | Real Robot Interface [Shared] | 1 | Phases 2, 3 | Phases 9, 10 |
| 12 | CLI Entry Point [Metal] | 2 | Phases 8, 9, 10, 11 | — |
| 13 | Integration Testing | 7 | Phase 12 | — |

**Total: 41 tasks**

**Recommended execution order (sequential, for a single engineer):**

| Order | Task | Parallel? | Key Test File |
|-------|------|-----------|---------------|
| 1 | 0.1-0.3 Environment validation | — | manual |
| 2 | 1.1-1.4 Scaffolding | — | — |
| 3 | 2.1-2.6 Data structures + config | ‖ with step 4 | `test_config.py` |
| 4 | 3.1-3.2 Compat layer | ‖ with step 3 | `test_compat.py` |
| 5 | 4.1 JointMapper | ‖ with steps 6, 8, 12 | `test_joint_mapper.py` |
| 6 | 5.1 Safety | ‖ with steps 5, 8, 12 | `test_safety.py` |
| 7 | 4.2 ObservationBuilder | after step 5 | `test_observations.py` |
| 8 | 7.1-7.3 SimRobot + DDS | ‖ with steps 5, 6 (needs 3, 2) | `test_sim_robot.py` |
| 9 | 6.1-6.4 Policies | after steps 5, 7 | `test_isaaclab_policy.py`, `test_beyondmimic_policy.py` |
| 10 | 8.1 Controller (+policy reload) | after steps 6, 8, 9 | `test_controller.py` |
| 11 | 9.1-9.3 Viewer | after step 10 | `test_viewer.py` + manual |
| 12 | 10.1-10.2 Logger (HDF5 + NPZ) | ‖ with steps 5-11 | `test_logger.py` |
| 13 | 11.1 RealRobot | ‖ with steps 9-11 | `test_real_robot.py` |
| 14 | 12.1-12.2 CLI | after all above | `test_main.py` |
| 15 | 13.1-13.7 Integration | after step 14 | `test_integration.py` + manual |

---

## Open Items Requiring Engineer Investigation

| ID | Item | When | Impact |
|----|------|------|--------|
| OI-1 | **MuJoCo viewer key_callback API**: Verify `launch_passive` accepts `key_callback` parameter in the installed MuJoCo version. Check GLFW keycode values. | Task 9.1 | If not supported, fall back to polling `viewer.key` or stdin-based input. |
| OI-2 | **unitree_sdk2_python macOS import**: Does the SDK import cleanly, or does it fail at import time due to RecurrentThread? | Task 0.1 / 3.2 | Determines whether monkey-patch is needed at import time or only at usage time. |
| OI-3 | **CycloneDDS on macOS lo0**: Does DDS multicast work on macOS loopback? | Task 0.1 | If not, may need CycloneDDS XML config for unicast. |
| OI-4 | **unitree_sdk2_python version**: What version is compatible with G1 robot firmware? | Task 7.2 | Wrong version could send malformed commands. |
| OI-5 | **BeyondMimic observation construction**: Now fully specified in SPEC section 4.3. Implementation should follow the 7-step approach in Task 6.4. Validate against a real BeyondMimic ONNX model when available. | Task 6.4 | Resolved — see SPEC section 4.3 for complete observation vector specification. |
| OI-6 | **DDS CRC calculation**: Does the Python SDK's `CRC` utility handle `LowCmd_` CRC automatically? | Task 11.1 | If not, manual CRC before publishing. |
| OI-7 | **MuJoCo viewer overlay API**: What's the Python API for text overlays in the viewer? | Task 9.3 | If too complex, stdout status lines are sufficient. |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Joint ordering bug sends wrong commands to real robot | Medium | **Critical** | Exhaustive JointMapper tests. Validate with known-good states before real robot. Start in harness. |
| CycloneDDS doesn't work on macOS ARM64 | Low | High | Phase 0 validates this before any code is written. Fall back to PLAN_DOCKER.md. |
| RecurrentThread patch breaks SDK internals | Low | Medium | The patch only replaces the thread wrapper, not DDS internals. Phase 0 / Task 3.2 validate end-to-end. |
| MuJoCo viewer key_callback unavailable | Low | Medium | Fall back to stdin-based input or polling. |
| Policy frequency not maintained (>20ms per loop) | Low | High (real robot) | Profile control loop. ONNX inference is typically <5ms for small policies. |
| BeyondMimic observation construction incorrect | Medium | Medium | Now spec'd in SPEC section 4.3. Study training code. Log observations and verify offline against known-good policy outputs. |

---

## Appendix A: Testing Philosophy

### Write Tests First (Where Practical)

For pure-logic modules (JointMapper, ObservationBuilder, SafetyController, Config), write tests first based on the spec, then implement until tests pass.

For I/O-heavy modules (SimRobot, RealRobot), write implementation first, then add tests.

### Test Granularity

- **Unit tests**: One function/method, mocked dependencies, <100ms each
- **Integration tests**: Multiple modules together, can be slower
- **Manual tests**: Visual verification or hardware, documented as checklists

### Value-Level Testing for Safety-Critical Math

Any math formula that directly computes robot commands must have tests with **known numerical inputs and hand-computed expected outputs**. Shape/dtype tests are necessary but not sufficient. This includes:
- **Impedance control law**: `ctrl = tau_ff + kp*(q_des - q) + kd*(dq_des - dq)` in `SimRobot.send_command()`
- **IsaacLab command building**: `target_pos = q_home + Ka * action`
- **BeyondMimic command building**: `target_pos = target_q + Ka * action`, `dq_target = target_dq`
- **Projected gravity**: Quaternion -> gravity vector rotation
- **Body-relative transforms**: Body position/orientation relative to anchor body
- **Safety limit clamping**: Joint position, velocity, and torque clamping

### Thread Safety Testing

The system runs 3-4 concurrent threads (control loop, DDS publisher, viewer/main thread, optionally logger). Key concurrent access patterns must be tested:
- Safety state machine: concurrent `estop()` + `start()` + `stop()` calls
- Velocity command: writer thread + reader thread
- DDS state buffer: callback thread + `get_state()` caller

### Phase Gate Testing

Each phase must have sufficient automated tests to prove it works before moving on. The principle is: **if a phase's tests pass, the phase is complete.** This means:
- Every public method must have at least one test
- Every error path documented in the code must have a test
- Every math formula must have a value-level test

### Test Naming Convention

```
test_{module}_{behavior}_{scenario}
```
Example: `test_joint_mapper_robot_to_observation_isaaclab_ordering`

---

## Appendix B: Coding Standards

- **Type hints** on all function signatures
- **Docstrings** on all public classes and methods (Google style)
- **No wildcard imports**
- **Constants** in `UPPER_SNAKE_CASE`, variables in `lower_snake_case`, classes in `PascalCase`
- **No magic numbers** — use named constants from config
- **Logging** via Python `logging` module (not `print`, except startup banners and headless status)
- **Line length**: 100 characters max
- **Python 3.10 max** — no 3.11+ features (walrus operator OK, match/case is 3.10+ so OK)

---

## Appendix C: Joint Name Reference Quick Table

**29-DOF Robot-Native Order (Config Names):**
```
 0: left_hip_pitch       15: left_shoulder_pitch
 1: left_hip_roll        16: left_shoulder_roll
 2: left_hip_yaw         17: left_shoulder_yaw
 3: left_knee            18: left_elbow
 4: left_ankle_pitch     19: left_wrist_roll
 5: left_ankle_roll      20: left_wrist_pitch
 6: right_hip_pitch      21: left_wrist_yaw
 7: right_hip_roll       22: right_shoulder_pitch
 8: right_hip_yaw        23: right_shoulder_roll
 9: right_knee           24: right_shoulder_yaw
10: right_ankle_pitch    25: right_elbow
11: right_ankle_roll     26: right_wrist_roll
12: waist_yaw            27: right_wrist_pitch
13: waist_roll           28: right_wrist_yaw
14: waist_pitch
```

**29-DOF IsaacLab Order (MuJoCo Joint Names):**
```
 0: left_hip_pitch_joint        15: left_shoulder_roll_joint
 1: right_hip_pitch_joint       16: right_shoulder_roll_joint
 2: waist_yaw_joint             17: left_ankle_roll_joint
 3: left_hip_roll_joint         18: right_ankle_roll_joint
 4: right_hip_roll_joint        19: left_shoulder_yaw_joint
 5: waist_roll_joint            20: right_shoulder_yaw_joint
 6: left_hip_yaw_joint          21: left_elbow_joint
 7: right_hip_yaw_joint         22: right_elbow_joint
 8: waist_pitch_joint           23: left_wrist_roll_joint
 9: left_knee_joint             24: right_wrist_roll_joint
10: right_knee_joint            25: left_wrist_pitch_joint
11: left_shoulder_pitch_joint   26: right_wrist_pitch_joint
12: right_shoulder_pitch_joint  27: left_wrist_yaw_joint
13: left_ankle_pitch_joint      28: right_wrist_yaw_joint
14: right_ankle_pitch_joint
```

---

## Appendix D: SPEC Sections Superseded by This Plan

The following SPEC.md sections are superseded for the Metal plan. The SPEC now includes `[Metal]` and `[Docker]` tags for sections that differ between plans.

| SPEC Section | Metal Behavior |
|---|---|
| 6.1-6.5 (Viser visualization) | Replaced by MuJoCo native viewer + keyboard controls |
| 9.1-9.6 (Docker configuration) | Not used. `uv` virtualenv instead. |
| 10.1 (Simulation mode CLI) | No `--viser-port` flag. Adds `--headless`, `--duration`, `--steps`. |
| 10.3 (Log replay `--visualize`) | Viser replay deferred. MuJoCo replay is future work. |
| 13.1 (Development workflow) | No Docker. Direct `python -m src.main sim ...` |

All other SPEC sections apply unchanged to this plan.
