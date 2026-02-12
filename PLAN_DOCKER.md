# Unitree G1 Deployment Stack - Comprehensive Implementation Plan (Docker)

## Document Purpose

This document is the authoritative plan of work for building the Unitree G1 deployment stack using Docker and Viser browser-based UI. It is written so that an engineer can pick it up and execute each task exactly, in order, with clear acceptance criteria. Every assumption, design decision, and trade-off is declared explicitly.

**Dual-Plan Context:** A companion plan, **PLAN_METAL.md**, implements the same stack for bare-metal macOS using MuJoCo's native viewer. Both plans share the same core `src/` modules (`config`, `robot`, `policy`, `control`, `logging`, `compat`). The only Docker-plan-specific modules are `src/viz/` (Viser UI) and Docker configuration files. See **SPEC.md** for the unified specification with `[Docker]`/`[Metal]` tags marking divergent sections.

**Project documentation files (`SPEC.md`, `PLAN_DOCKER.md`, `PLAN_METAL.md`, `WORK.md`) must never be deleted.** They are the authoritative record of requirements, design decisions, and implementation plans. They should be updated in place as the project evolves.

---

## Guiding Principles

1. **Test as you go.** Every module gets unit tests written alongside the implementation code. Tests are run after every change. Do not move to the next task until the current task's tests pass.
2. **Build bottom-up.** Start with foundational data structures and pure-logic modules that have zero external dependencies. These are the easiest to test and the hardest to change later.
3. **One concern per module.** Each file does one thing. If a module is doing two things, split it.
4. **Sim-first, real-second.** The entire sim path must work end-to-end before real robot code is tested. Real robot code is too dangerous to iterate on casually.
5. **Fail loudly at startup, not at runtime.** All configuration validation, dimension checks, and joint name resolution happen at initialization. Runtime errors trigger E-stop.

---

## Assumptions

| ID | Assumption | Rationale |
|----|-----------|-----------|
| A1 | **Pure Python SDK** (`unitree_sdk2_python`) is used. No ROS/ROS2. | User decision. Simpler Docker, fewer dependencies, matches reference implementation. |
| A2 | **29-DOF is the primary target.** 23-DOF support is architected but lightly tested. | User has only a 29-DOF robot. 23-DOF is future-proofing. |
| A3 | **Real ONNX policy files are available** from the user for integration testing. | User confirmed. Mock ONNX models will still be created for unit tests to avoid coupling. |
| A4 | **Ubuntu 20.04 is the Docker base image.** Python 3.8+ via deadsnakes PPA if the system Python is too old. | Spec requirement. Oldest supported Ubuntu. |
| A5 | **EGL works for headless MuJoCo rendering** on both Linux native and macOS Docker (Linux VM). | Spec states this. Must be validated during Docker setup. If EGL fails on macOS Docker, OSMesa is the fallback. |
| A6 | **DDS multicast works inside a single Docker container** on the loopback interface for simulation. | Reference implementation uses `lo` interface with domain ID 1. Must be validated. |
| A7 | **Viser runs inside the Docker container** and is accessible from the host browser via port mapping. | Viser binds to `0.0.0.0` and uses WebSocket, which works through Docker port forwarding. |
| A8 | **The control loop must never be blocked** by visualization, logging, or UI. On real hardware, a missed cycle could cause a fall. | Spec section 3.4. Thread-safe shared state with non-blocking reads in the control thread. |
| A9 | **BeyondMimic and IsaacLab policies are developed in parallel**, but IsaacLab is simpler and will likely be testable first. | User decision. Both share the same JointMapper infrastructure. |
| A10 | **No GPU inference.** ONNX Runtime CPU provider is sufficient for the small policy networks used. | Spec section 9.2. Avoids CUDA dependency in Docker. |
| A11 | **Both HDF5 and NPZ log formats are supported**, selectable via config. HDF5 (default) offers compression and partial reads. NPZ is a simpler alternative with no `h5py` dependency. | Spec section 8.1. Both formats use the same dataset schema. |
| A12 | **Joint name mapping accepts all three naming conventions** (config names like `left_hip_pitch`, IsaacLab/MuJoCo names like `left_hip_pitch_joint`, or DDS names like `L_LEG_HIP_PITCH`) and resolves to a canonical form at startup. | Spec sections 2.2, 4.6. The mapper is validated at startup. |
| A13 | **macOS Docker networking**: `network_mode: host` does not provide true host networking on macOS. Viser port must be explicitly mapped. DDS simulation works internally on `lo`. | Known Docker-on-Mac limitation. Real robot deployment on macOS is not a target (real robot requires Linux host with ethernet). |

---

## Design Decisions and Trade-offs

### D1: Threading Model - Threads vs AsyncIO

**Decision:** Use `threading` with locks, not `asyncio`.

**Rationale:** The reference implementation uses threads. MuJoCo is not async-compatible. The DDS SDK uses callback threads internally. Mixing asyncio with these would add complexity for no benefit. Thread-safe shared state with a `threading.Lock` is straightforward and matches the reference.

**Trade-off:** Threads have GIL contention, but the control loop does mostly NumPy/ONNX operations which release the GIL. If profiling shows GIL as a bottleneck, the control loop can be moved to a subprocess with shared memory.

### D2: DDS Architecture - Bridge vs Direct

**Decision:** In simulation mode, the DDS bridge runs in-process (same process as the controller). In real robot mode, the controller publishes/subscribes directly to the robot's DDS topics.

**Rationale:** The reference implementation has the bridge in the same process. Keeping it in-process avoids inter-process DDS discovery issues in Docker. The SimRobot class encapsulates MuJoCo + DDS bridge as a single unit.

**Trade-off:** Physics stepping (`mj_step`) runs synchronously in the control loop thread, while a DDS publishing thread runs in the background. GIL contention between these is minimal since `mj_step` and NumPy/ONNX operations release the GIL.

### D3: Configuration Hierarchy

**Decision:** CLI args override YAML config. YAML config has a default file that can be extended by robot-specific configs. BeyondMimic policies override gains from ONNX metadata (if `use_onnx_metadata: true`).

**Rationale:** This gives maximum flexibility. The user can have a base config and override just what they need per policy or per session.

**Resolution order:** CLI > YAML (robot-specific merges over default) > Code defaults > ONNX metadata (BeyondMimic only, when `use_onnx_metadata: true`).

### D4: Observation Normalization

**Decision:** No observation normalization is applied by default. If a policy requires normalization (e.g., running mean/std), it must be embedded in the ONNX model or configured via a separate normalization config.

**Rationale:** IsaacLab policies typically handle normalization internally. Adding normalization logic here would couple this system to specific training pipelines.

### D5: Torque Commands vs Position Commands

**Decision:** The system sends position targets with PD gains via the `LowCmd` message (fields: `q`, `dq`, `kp`, `kd`, `tau`). The motor controller on the robot computes the actual torque. This matches the impedance control law: `tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff`.

**Rationale:** This is how the reference implementation works and how the robot hardware expects commands. The policy outputs position offsets, which are converted to position targets via the control law.

### D6: Single Container

**Decision:** Use a single Docker container for both sim and real modes.

**Rationale:** The only difference between sim and real is which `RobotInterface` subclass is instantiated. A single container avoids maintaining two Dockerfiles. Mode is selected via CLI argument.

### D7: Volume Mount Strategy

**Decision:** Mount `src/`, `configs/`, `assets/`, `policies/`, and `logs/` as bind mounts. Code edits inside or outside the container are reflected in both places.

**Rationale:** User requirement. Enables development workflow where code is edited on the host (IDE) and run in the container.

### D8: Joint Name Mapping Architecture

**Decision:** Three name spaces exist: (1) Config names used in YAML (e.g., `left_hip_pitch`), (2) IsaacLab/MuJoCo joint names (e.g., `left_hip_pitch_joint`), (3) DDS/IDL names used by the Unitree SDK (e.g., `L_LEG_HIP_PITCH`). IsaacLab uses the same naming convention as MuJoCo MJCF files. A single `JointMapper` class handles all translations via lookup tables defined in robot-specific config files.

**Rationale:** Policies from different training frameworks use different naming conventions. A centralized mapper validated at startup prevents silent ordering bugs that could damage the robot.

---

## Execution Guide

This section provides a high-level roadmap for executing this plan. Use it to understand the critical path, identify parallelization opportunities, and track progress.

### Dependency Graph

```
Phase 0: Project Scaffolding
  │
  v
Phase 1: Core Data Structures & Config [Shared]
  │
  ├──────────┬──────────┐
  v          v          v
Phase 2    Phase 3    Phase 4    Phase 5: MuJoCo Sim
Joint Map  Safety     Policies    & DDS Bridge
[Shared]   [Shared]   [Shared]   [Shared]
  │          │          │            │
  │          └──────────┴────────────┤
  │                                  │
  │          ┌───────────────────────┤
  v          v                       │
Phase 8    Phase 6: Control Loop ◄──┘
Logging       │
[Shared]      ├─────────────────────┐
  │           v                     v
  │        Phase 7: Viser UI    Phase 9: Real Robot
  │        [Docker]             [Shared]
  │           │                     │
  └───────────┴─────────────────────┤
                                    v
                          Phase 10: CLI & Entry Point
                                    │
                                    v
                          Phase 11: Docker Configuration
                                    │
                                    v
                          Phase 12: Integration Tests
                                    │
                                    v
                          Phase 13: Documentation
```

### Critical Path

The longest dependency chain determines the minimum sequential work:

**Phase 0 → 1 → 2 → 4 → 5 → 6 → 7 → 10 → 11 → 12 → 13**

All other phases can be parallelized alongside this chain.

### Phase Execution Summary

| Phase | Description | Depends On | Can Parallelize With | Est. Tasks |
|-------|-------------|------------|---------------------|------------|
| **0** | Project Scaffolding | — | — | 4 |
| **1** | Core Data Structures & Config [Shared] | Phase 0 | — | 7 |
| **2** | Joint Mapping & Observations [Shared] | Phase 1 | Phases 3, 4, 5, 8 | 2 |
| **3** | Safety System [Shared] | Phase 1 | Phases 2, 4, 5, 8 | 1 |
| **4** | Policy Interfaces [Shared] | Phases 1, 2 | Phases 3, 5, 8 | 4 |
| **5** | MuJoCo Sim & DDS Bridge [Shared] | Phase 1 | Phases 2, 3, 4, 8 | 3 |
| **6** | Control Loop [Shared] | Phases 3, 4, 5 | Phase 8 | 1 |
| **7** | Visualization (Viser) [Docker] | Phase 6 | Phases 8, 9 | 4 |
| **8** | Logging System [Shared] | Phase 1 | Phases 2-7, 9 | 2 |
| **9** | Real Robot Interface [Shared] | Phase 1 | Phases 7, 8 | 1 |
| **10** | CLI & Entry Point | Phases 6, 7, 8, 9 | — | 2 |
| **11** | Docker Configuration [Docker] | Phase 10 | — | 3 |
| **12** | Integration Testing | Phase 11 | — | 6 |
| **13** | Documentation | Phase 12 | — | 2 |

### Parallelization Opportunities

**After Phase 1 completes**, four independent tracks can proceed simultaneously:
- **Track A**: Phase 2 (joint mapping) → Phase 4 (policies)
- **Track B**: Phase 3 (safety)
- **Track C**: Phase 5 (MuJoCo sim + DDS)
- **Track D**: Phase 8 (logging)

**After Phase 6 (control loop) completes**, two more independent tracks:
- **Track A**: Phase 7 (Viser visualization)
- **Track B**: Phase 9 (real robot)

### Cross-Plan Reference (PLAN_METAL.md)

Phases marked **[Shared]** produce identical code in both plans. If implementing both plans, shared phases only need to be done once. The Metal plan differs in:
- Has Phase 0 for macOS environment validation (CycloneDDS, SDK imports)
- Has a separate Phase 3 for the compat layer (`RecurrentThread` shim)
- Phase 9 is MuJoCo viewer integration instead of Viser
- No Docker configuration phase

---

## Phase 0: Project Scaffolding

### Task 0.1: Create Directory Structure

Create the full directory tree as specified in SPEC.md section 3.2.

**Files to create:**
```
unitree_launcher/
├── docker/
│   ├── Dockerfile              (empty placeholder)
│   └── docker-compose.yml      (empty placeholder)
├── configs/
│   ├── default.yaml            (empty placeholder)
│   ├── g1_29dof.yaml           (empty placeholder)
│   └── g1_23dof.yaml           (empty placeholder)
├── assets/
│   └── robots/
│       └── g1/                 (empty, will receive model files)
├── policies/                   (empty, user will add ONNX files)
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── main.py                 (empty placeholder)
│   ├── config.py               (empty placeholder)
│   ├── compat.py               (cross-platform shims, shared with Metal plan)
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
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── robot_viz.py
│   │   └── ui.py
│   └── logging/
│       ├── __init__.py
│       ├── logger.py
│       └── replay.py
├── scripts/
│   ├── run_sim.sh
│   ├── run_real.sh
│   └── replay_log.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py             (shared fixtures)
│   ├── test_config.py
│   ├── test_joint_mapper.py
│   ├── test_observations.py
│   ├── test_safety.py
│   ├── test_isaaclab_policy.py
│   ├── test_beyondmimic_policy.py
│   ├── test_logger.py
│   ├── test_sim_robot.py
│   ├── test_controller.py
│   └── test_viz.py
├── pyproject.toml
└── requirements.txt
```

**Acceptance criteria:**
- All directories exist
- All `__init__.py` files exist (can be empty)
- `git status` shows the new structure

### Task 0.2: Copy MuJoCo Model Files

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

**Important note:** The MJCF files reference mesh paths relative to their location. The engineer must verify that mesh paths resolve correctly from `assets/robots/g1/`. If the XML uses `<compiler meshdir="meshes"/>` (relative), this should work. If it uses absolute paths, they must be patched. The engineer should also create a minimal `scene.xml` for simulation that includes the robot model, a ground plane, and lighting.

### Task 0.3: Set Up `pyproject.toml` and `requirements.txt`

**`pyproject.toml`:**
```toml
[project]
name = "unitree-launcher"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "mujoco>=3.0",
    "viser>=0.2.0",
    "onnxruntime>=1.16",
    "unitree_sdk2_python",
    "cyclonedds",
    "numpy",
    "pyyaml",
    "h5py",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**`requirements.txt`:** Mirror of `pyproject.toml` dependencies for Docker.

**Note on `unitree_sdk2_python`:** This package may need to be installed from GitHub (`pip install unitree_sdk2_python`). It depends on `cyclonedds`. The engineer must verify that CycloneDDS builds cleanly in the Docker image (it requires CMake and a C compiler). If build issues arise, the engineer should check whether prebuilt wheels are available for the target architecture. On macOS (Metal plan), `cyclonedds==0.10.2` provides ARM64 wheels for Python 3.10.

**Note on version pinning:** Pin major versions only (e.g., `mujoco>=3.0,<4.0`) to balance reproducibility and compatibility. If exact pinning is needed later, generate a `requirements.lock` from the working environment.

**Acceptance criteria:**
- `pip install -e ".[dev]"` succeeds in a fresh virtualenv
- `pytest` runs (even with 0 tests) without import errors

### Task 0.4: Create `conftest.py` with Shared Fixtures

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

## Phase 1: Core Data Structures and Configuration

These modules have **zero external dependencies** beyond Python stdlib and NumPy. They are the foundation everything else builds on. Every function must be unit tested.

### Task 1.1: Implement Robot Constants (`src/config.py` - Part 1: Constants)

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

### Task 1.2: Implement State and Command Dataclasses (`src/robot/base.py` - Part 1)

Define `RobotState` and `RobotCommand` as Python dataclasses (from SPEC section 5.2).

**Contents:**
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

### Task 1.3: Implement Abstract `RobotInterface` (`src/robot/base.py` - Part 2)

Define the abstract base class from SPEC section 5.1. This is a pure interface with no implementation.

**Contents:**
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

**Tests:** No tests needed for a pure ABC. It will be validated by the type system when SimRobot/RealRobot implement it.

### Task 1.4: Implement Abstract `PolicyInterface` (`src/policy/base.py`)

Define the abstract base class from SPEC section 4.9.

**Contents:**
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

**Note:** The `get_action` method uses `**kwargs` to accommodate BeyondMimic's `time_step` input without polluting the interface for IsaacLab policies.

**Tests:** No tests needed for a pure ABC.

### Task 1.5: Implement Configuration Loading (`src/config.py` - Part 2: Config Dataclass and YAML Loading)

Define a `Config` dataclass that represents the entire YAML configuration (SPEC section 11.1). Implement YAML loading with validation.

**Contents:**
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
    interface: str = "lo"
    domain_id: int = 1

@dataclass
class ViserConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    default_camera: str = "follow"
    default_render: str = "mesh"

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
    viser: ViserConfig
    logging: LoggingConfig

def load_config(path: str) -> Config:
    """Load and validate config from YAML file."""
    ...

def merge_configs(base: Config, override: Config) -> Config:
    """Merge override config into base config (non-None values win)."""
    ...
```

**Joint name resolution in configs:**
The config parser must accept joint names in **any** of the three naming conventions (per Decision D8 and Assumption A12):
- Config names: `left_hip_pitch`
- IsaacLab/MuJoCo names: `left_hip_pitch_joint`
- DDS/IDL names: `L_LEG_HIP_PITCH`

Implement a `resolve_joint_name(name: str, variant: str) -> str` function that normalizes any accepted name to the canonical config-name form (e.g., `left_hip_pitch`). This function should be called during config loading for every joint name in `observed_joints` and `controlled_joints`. If a name doesn't match any known joint in any convention, raise `ValueError` with the unrecognized name and a list of valid names.

**Validation at load time:**
- `variant` must be `"g1_29dof"` or `"g1_23dof"`
- If `controlled_joints` is specified, every name must resolve to a valid joint for the variant
- If `observed_joints` is specified, every name must resolve to a valid joint for the variant
- If `kp`/`kd`/`ka` are lists, their length must match the number of controlled joints
- `policy_frequency` must evenly divide `sim_frequency` (so `sim_steps_per_policy_step` is an integer)
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
- `test_idl_mode_validation`: `idl_mode=2` raises `ValueError` (only 0 or 1 allowed)
- `test_logging_format_validation`: `format="invalid"` raises `ValueError` (only "hdf5" or "npz" allowed)
- `test_merge_configs_none_preserves_base`: `merge_configs(base, override)` where override field is `None` does not overwrite the base value
- `test_ka_list_wrong_length_rejected`: `ka` list length != number of controlled joints raises `ValueError`
- `test_viser_config_defaults`: `ViserConfig` has `host="0.0.0.0", port=8080`

### Task 1.6: Write Default YAML Configuration Files

**File:** `configs/default.yaml`
Write the full default configuration from SPEC section 11.1.

**File:** `configs/g1_29dof.yaml`
Write the 29-DOF specific configuration. Include the BeyondMimic reference gains from SPEC section 11.7 as comments (not active values, since they are policy-specific).

**File:** `configs/g1_23dof.yaml`
Write the 23-DOF specific configuration.

**Acceptance criteria:**
- `load_config("configs/default.yaml")` succeeds
- `load_config("configs/g1_29dof.yaml")` succeeds
- `load_config("configs/g1_23dof.yaml")` succeeds
- All three pass validation

### Task 1.7: Implement Cross-Platform Compatibility Module (`src/compat.py`)

This module provides cross-platform shims shared between both Docker and Metal plans.

**Contents:**
```python
import platform
import threading
import time

class RecurrentThread:
    """Pure-Python replacement for unitree_sdk2_python's RecurrentThread.
    The SDK's native RecurrentThread uses Linux-specific threading primitives.
    This implementation uses threading.Thread + time.sleep() and works on all platforms."""

    def __init__(self, interval: float, target, name: str = ""):
        self._interval = interval
        self._target = target
        self._name = name
        self._stop_event = threading.Event()
        self._thread = None

    def Start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def Shutdown(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop_event.is_set():
            self._target()
            time.sleep(self._interval)

def resolve_network_interface(interface: str) -> str:
    """Resolve 'auto' interface to platform-appropriate loopback.
    Returns 'lo0' on macOS, 'lo' on Linux. Pass-through for explicit names."""
    if interface == "auto":
        return "lo0" if platform.system() == "Darwin" else "lo"
    return interface
```

**Note:** In the Docker plan, this module is still used since it provides the same `RecurrentThread` API. The Docker container runs Linux, so `lo` is always correct for simulation. However, using `compat.py` consistently means the same `SimRobot` code works in both plans.

**Tests (`tests/test_compat.py`):**
- `test_recurrent_thread_start_stop`: Thread starts, fires target, stops cleanly
- `test_recurrent_thread_timing`: Fires at approximately the correct interval (within 20% tolerance)
- `test_recurrent_thread_shutdown_idempotent`: Calling `Shutdown()` twice does not raise
- `test_recurrent_thread_shutdown_before_start`: Calling Shutdown() before Start() doesn't crash or raise
- `test_recurrent_thread_target_exception`: If target raises, thread logs error and continues running (does not silently die). Verify thread stays alive after an exception in one iteration.
- `test_recurrent_thread_slow_target`: When target takes longer than interval, thread still runs (no negative sleep crash). Verify it calls target again immediately.
- `test_recurrent_thread_daemon`: Thread is a daemon thread (won't block exit)
- `test_resolve_interface_auto`: Returns `lo0` on macOS, `lo` on Linux
- `test_resolve_interface_explicit`: Returns explicit name unchanged

**Run tests:** `pytest tests/test_compat.py -v`

**Acceptance criteria:**
- `RecurrentThread` starts, runs, and stops cleanly
- `resolve_network_interface("auto")` returns correct loopback for current platform
- All tests pass

---

## Phase 2: Joint Mapping and Observation Building

These modules depend only on Phase 1 (data structures + config). They are pure NumPy operations with no I/O. Every function must be unit tested thoroughly because **incorrect joint mapping can damage the physical robot.**

### Task 2.1: Implement `JointMapper` (`src/policy/joint_mapper.py`)

Implement the full `JointMapper` class from SPEC section 4.10.

**Key behaviors to implement:**
1. Accept `robot_joints` (all joints in native order), optional `observed_joints`, optional `controlled_joints`
2. Resolution order when args are None (from SPEC section 4.10 docstring)
3. Build index arrays mapping policy order <-> robot-native order for both observed and controlled joints
4. `robot_to_observation()`: Extract observed joints from full state, in policy order
5. `robot_to_action()`: Extract controlled joints from full state, in policy order
6. `action_to_robot()`: Map policy action (n_controlled,) to full robot array (n_total,), with default for uncontrolled
7. Validate all joint names at init time, raise `ValueError` for unknowns

**Edge cases to handle:**
- `observed_joints` and `controlled_joints` are the same list -> valid, common case
- `observed_joints` is a superset of `controlled_joints` -> valid (full observe, partial control)
- `controlled_joints` is a superset of `observed_joints` -> unusual but valid
- Duplicate joint names in either list -> raise `ValueError`
- Empty lists -> raise `ValueError` (at least one joint must be controlled)

**Tests (`tests/test_joint_mapper.py`):**

This is the most critical test file. Every mapping permutation must be tested.

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

### Task 2.2: Implement `ObservationBuilder` (`src/policy/observations.py`)

Implement the **IsaacLab** observation builder from SPEC section 4.11. This class is used only for IsaacLab policies. BeyondMimic observation construction is handled internally by `BeyondMimicPolicy` (see Task 4.4) because it depends on policy-specific `obs_terms` metadata.

**Contents:**
```python
class ObservationBuilder:
    def __init__(self, joint_mapper: JointMapper, config: ControlConfig,
                 use_estimator: bool = True): ...

    @property
    def observation_dim(self) -> int:
        """2*n_observed + n_controlled + (12 if use_estimator else 9)"""
        ...

    def build(self, robot_state: RobotState, last_action: np.ndarray,
              velocity_command: np.ndarray) -> np.ndarray: ...

    def compute_projected_gravity(self, quaternion_wxyz: np.ndarray) -> np.ndarray: ...

    def compute_body_velocity_in_body_frame(self, world_velocity: np.ndarray,
                                             quaternion_wxyz: np.ndarray) -> np.ndarray: ...
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

**Quaternion math note:** MuJoCo uses wxyz quaternion convention. The rotation of a vector `v` by quaternion `q` is: `v' = q * [0, v] * q_conj`. Implement this using the rotation matrix form to avoid quaternion multiplication edge cases.

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

## Phase 3: Safety System

The safety system is pure state machine logic with no external dependencies. It can be tested completely in isolation.

### Task 3.1: Implement Safety State Machine (`src/control/safety.py`)

Implement the state machine from SPEC section 7.3 and the damping mode from section 7.2.

**Contents:**
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
        """Transition to RUNNING. Returns False if transition invalid."""
        ...

    def stop(self) -> bool:
        """Transition to STOPPED. Returns False if transition invalid."""
        ...

    def estop(self) -> None:
        """Trigger E-stop. Always succeeds. Latching."""
        ...

    def clear_estop(self) -> bool:
        """Clear E-stop -> STOPPED. Returns False if not in ESTOP state."""
        ...

    def get_damping_command(self, current_state: RobotState) -> RobotCommand:
        """Generate a damping-mode command (for E-stop or non-controlled joints)."""
        ...

    def check_orientation(self, imu_quaternion: np.ndarray) -> tuple[bool, str]:
        """Check if robot orientation is safe for policy start (real robot).
        Returns (is_safe, message)."""
        ...
```

**State transition rules (from SPEC section 7.3):**
- IDLE -> RUNNING (via `start()`)
- RUNNING -> STOPPED (via `stop()`)
- RUNNING -> ESTOP (via `estop()`)
- ESTOP -> STOPPED (via `clear_estop()`)
- STOPPED -> IDLE (implicit, or via reset)
- ESTOP from any state except IDLE

**Damping command (from SPEC section 7.2):**
```
target position = current position (zero position error)
kp = 0
kd = kd_damp (from config)
tau = 0 (no feedforward)
```

**Tests (`tests/test_safety.py`):**
- `test_initial_state_idle`: New SafetyController starts in IDLE
- `test_idle_to_running`: `start()` succeeds from IDLE
- `test_running_to_stopped`: `stop()` succeeds from RUNNING
- `test_running_to_estop`: `estop()` succeeds from RUNNING
- `test_estop_to_stopped`: `clear_estop()` succeeds from ESTOP
- `test_estop_latching`: After estop(), state remains ESTOP until cleared
- `test_cannot_start_from_estop`: `start()` returns False from ESTOP
- `test_cannot_start_from_running`: `start()` returns False from RUNNING
- `test_estop_from_stopped`: `estop()` transitions from STOPPED to ESTOP
- `test_estop_from_idle_rejected`: `estop()` has no effect in IDLE (robot not active)
- `test_damping_command_shape`: Damping command arrays have correct shape
- `test_damping_command_zero_kp`: kp is all zeros
- `test_damping_command_kd_set`: kd equals config.kd_damp
- `test_damping_command_position_is_current`: Target position equals current joint positions
- `test_orientation_check_upright`: Quaternion [1,0,0,0] passes
- `test_orientation_check_tilted_safe`: Small tilt passes
- `test_orientation_check_tilted_unsafe`: Large tilt (>35 deg) fails
- `test_orientation_check_inverted`: Upside down fails
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

## Phase 4: Policy Interfaces

These modules depend on ONNX Runtime for loading models. Unit tests should use small synthetic ONNX models created in the test fixtures.

### Task 4.1: Create Test ONNX Model Fixtures (`tests/conftest.py` - additions)

Create helper functions that generate minimal ONNX models for testing:

```python
def create_isaaclab_onnx(obs_dim: int, action_dim: int, path: str) -> None:
    """Create a minimal ONNX model that mimics IsaacLab policy.
    Input: 'obs' shape [1, obs_dim]
    Output: 'action' shape [1, action_dim]
    The model should be a simple linear layer (or identity-like) so outputs are predictable.
    """
    ...

def create_beyondmimic_onnx(obs_dim: int, action_dim: int, n_joints: int,
                             path: str, metadata: dict) -> None:
    """Create a minimal ONNX model that mimics BeyondMimic policy.
    Inputs: 'obs' shape [1, obs_dim], 'time_step' shape [1]
    Outputs: 'action' [1, action_dim], 'target_q' [1, n_joints],
             'target_dq' [1, n_joints], etc.
    Metadata is embedded in the ONNX model properties.
    """
    ...
```

**Implementation note:** Use `onnx.helper.make_graph()` and `onnx.helper.make_model()` to construct minimal ONNX graphs. The model body can be a simple constant output or identity transformation. The goal is to verify loading, dimension checking, and metadata extraction — not policy quality.

**Acceptance criteria:**
- `onnxruntime.InferenceSession(path)` succeeds on the generated models
- Input/output names and shapes match expectations

### Task 4.2: Implement Policy Format Detection (`src/policy/base.py` - addition)

Add the format detection function from SPEC section 4.4:

```python
def detect_policy_format(onnx_path: str) -> str:
    """Auto-detect policy format from ONNX structure.
    Returns 'isaaclab' or 'beyondmimic'.
    """
    session = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]
    if "time_step" in input_names:
        return "beyondmimic"
    return "isaaclab"
```

**Tests (`tests/test_isaaclab_policy.py` - Part 1):**
- `test_detect_isaaclab`: IsaacLab model detected correctly
- `test_detect_beyondmimic`: BeyondMimic model (with `time_step` input) detected correctly

### Task 4.3: Implement IsaacLab Policy (`src/policy/isaaclab_policy.py`)

**Contents:**
```python
class IsaacLabPolicy(PolicyInterface):
    def __init__(self, joint_mapper: JointMapper, config: ControlConfig): ...
    def load(self, path: str) -> None: ...
    def reset(self) -> None: ...
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray: ...

    @property
    def observation_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
```

**Key behaviors:**
- `load()`: Create ONNX InferenceSession, validate input/output dimensions against JointMapper
- `get_action()`: Run ONNX inference, return action array (n_controlled,)
- `reset()`: Clear any internal state (for future recurrent policies)
- Dimension validation: `obs_dim` from ONNX must match `ObservationBuilder.observation_dim`. `action_dim` from ONNX must match `joint_mapper.n_controlled`.

**Tests (`tests/test_isaaclab_policy.py` - Part 2):**
- `test_load_valid_policy`: Load succeeds, dimensions set correctly
- `test_load_invalid_path_raises`: Nonexistent file raises clear error
- `test_load_dimension_mismatch_raises`: Policy with wrong obs_dim raises with descriptive message
- `test_get_action_output_shape`: Output shape is (n_controlled,)
- `test_get_action_output_dtype`: Output is float32
- `test_reset_clears_state`: Reset doesn't crash
- `test_observation_dim_matches_builder`: Policy's obs_dim matches ObservationBuilder's obs_dim
- `test_get_action_deterministic`: Same observation input twice produces identical output (ONNX CPU inference must be deterministic)
- `test_load_corrupt_onnx_raises`: Truncated or corrupt file raises a clear error (not a segfault)
- `test_load_twice_replaces_session`: Calling `load()` a second time replaces the ONNX session cleanly (no resource leak)

### Task 4.4: Implement BeyondMimic Policy (`src/policy/beyondmimic_policy.py`)

**Contents:**
```python
class BeyondMimicPolicy(PolicyInterface):
    def __init__(self, joint_mapper: JointMapper, config: ControlConfig): ...
    def load(self, path: str) -> None: ...
    def reset(self) -> None: ...
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray: ...

    @property
    def observation_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def target_q(self) -> np.ndarray: ...
    @property
    def target_dq(self) -> np.ndarray: ...

    def load_metadata(self, onnx_path: str) -> dict: ...
```

**Key behaviors:**
- `load()`: Load ONNX, extract metadata (joint_names, stiffness, damping, action_scale, etc.)
- `get_action(**kwargs)`: Expects `time_step` kwarg. Returns action + stores `target_q`, `target_dq` as properties for the control law.
- `build_observation()`: Build the observation vector internally. Unlike IsaacLab (which uses a separate `ObservationBuilder`), BeyondMimic builds observations inside the policy class because the observation structure depends on policy-specific `obs_terms` metadata that is only known after loading the ONNX model.
- When `use_onnx_metadata: true`, override config gains with ONNX metadata gains.
- Metadata parsing: The metadata values are stored as string representations of Python lists. Use `ast.literal_eval()` (not `eval()`) for safe parsing. **Note:** The SPEC (section 4.4) shows `eval()` in its example code — this is a security risk. Always use `ast.literal_eval()` instead.

**Observation construction (see SPEC section 4.3 for full specification):**
Unlike IsaacLab (which uses the shared `ObservationBuilder`), BeyondMimic builds observations *inside* the policy class via a `build_observation()` method. This is because the observation structure depends on policy-specific metadata (`obs_terms`, `controlled_bodies`) only known after loading the ONNX model. Key steps:
1. Read `obs_terms` and `controlled_bodies` from ONNX metadata at load time
2. Use MuJoCo's `xpos`/`xquat` for body state (sim) or forward kinematics (real)
3. Convert all rotations to 6D representation (first 2 columns of rotation matrix)
4. Compute body-relative positions/orientations relative to anchor body
5. Cache previous ONNX outputs for motion target construction
6. Stack history frames if `obs_history_lengths` requires it
7. Concatenate into observation vector: `obs_dim = 21 + (9 × N_bodies)` (without history)

**Control law difference (from SPEC section 4.3):**
```
tau = Kp * (target_q + Ka * action - q) - Kd * (qdot - target_dq)
```
Note: BeyondMimic uses `target_q` from the ONNX output (not `q_home`), and subtracts `target_dq` from the velocity term.

**Tests (`tests/test_beyondmimic_policy.py`):**
- `test_load_valid_policy`: Load succeeds, metadata extracted
- `test_metadata_extraction`: Joint names, stiffness, damping extracted correctly
- `test_get_action_with_time_step`: Inference with time_step input works
- `test_get_action_stores_targets`: After inference, `target_q` and `target_dq` are set
- `test_get_action_output_shape`: Action shape matches n_controlled
- `test_metadata_overrides_config`: When `use_onnx_metadata=True`, gains come from metadata
- `test_metadata_not_used_when_disabled`: When `use_onnx_metadata=False`, config gains are used
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

## Phase 5: MuJoCo Simulation and DDS Bridge

This phase introduces external dependencies (MuJoCo, CycloneDDS, unitree_sdk2_python). Tests require these packages to be installed.

### Task 5.1: Create MuJoCo Scene Files (`assets/robots/g1/scene_29dof.xml`, `scene_23dof.xml`)

Create two minimal scene files for simulation (one per robot variant):
- Each includes the appropriate robot model via `<include file="g1_29dof.xml"/>` or `<include file="g1_23dof.xml"/>`
- Adds a flat ground plane with appropriate friction
- Adds lighting (directional + ambient)
- Sets simulation timestep to 0.005s (200 Hz)
- Sets gravity to [0, 0, -9.81]

The `SimRobot` class selects the scene file based on `config.robot.variant`.

**Reference:** Look at `reference/unitree_mujoco/unitree_robots/g1/scene.xml` for the pattern, but simplify (no terrain).

**Acceptance criteria:**
- `mujoco.MjModel.from_xml_path("assets/robots/g1/scene_29dof.xml")` succeeds
- Model has 29 actuators
- Stepping the simulation 1000 times without commands produces a falling robot (gravity works)

### Task 5.2: Implement `SimRobot` (`src/robot/sim_robot.py`)

Implement the simulation robot interface that wraps MuJoCo + DDS bridge.

**Architecture:**
```python
class SimRobot(RobotInterface):
    def __init__(self, config: Config):
        # Load MuJoCo model
        # Initialize DDS (ChannelFactoryInitialize)
        # Create DDS publishers/subscribers

    def connect(self) -> None:
        """Start DDS state publishing thread. No physics thread —
        physics stepping is synchronous via step()."""

    def disconnect(self) -> None:
        """Stop DDS threads, clean up."""

    def get_state(self) -> RobotState:
        """Read current state from MuJoCo sensor data."""

    def send_command(self, cmd: RobotCommand) -> None:
        """Apply command to MuJoCo ctrl array (impedance control law)."""

    def step(self) -> None:
        """Advance simulation by one policy timestep
        (sim_frequency / policy_frequency physics steps, e.g. 4 at defaults).
        Called synchronously from the control loop."""

    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Reset MuJoCo to initial keyframe or specified state."""

    @property
    def n_dof(self) -> int:
        """Return number of actuators."""
```

**Threading model (critical — must be consistent):**

The control loop calls `get_state()`, `send_command()`, and `step()` **synchronously in sequence** from a single thread. There is **no separate physics thread.** This gives deterministic simulation behavior: command is applied, then physics advances. The only background thread is for DDS state publishing (external monitoring).

```
Control loop thread:  get_state() → policy → send_command() → step() → sleep
DDS publishing thread: periodically reads mj_data and publishes LowState_ (read-only)
```

Thread safety is only needed between the control loop and the DDS publishing thread. Use `threading.Lock` to protect `mj_data` reads in the DDS thread. The control loop holds the lock during `step()` (which calls `mj_step()`), and the DDS thread briefly acquires the lock to snapshot sensor data.

**Note on DDS publishing thread:** Use the `RecurrentThread` from `src/compat.py` (not the SDK's native `RecurrentThread`) for the periodic DDS publishing loop. This ensures cross-platform compatibility (shared code with Metal plan).

**Key implementation details:**

1. **Sensor data mapping** (from reference `unitree_sdk2py_bridge.py`):
   Compute offsets dynamically from the model, do **not** hard-code:
   ```python
   num_motor = mj_model.nu                    # e.g. 29
   dim_motor_sensor = 3 * num_motor           # e.g. 87
   # Motor sensors (3 per joint: position, velocity, torque)
   # Positions:  sensordata[0 : num_motor]
   # Velocities: sensordata[num_motor : 2 * num_motor]
   # Torques:    sensordata[2 * num_motor : 3 * num_motor]
   # IMU (starts at dim_motor_sensor):
   # Quaternion (wxyz): sensordata[dim_motor_sensor + 0 : dim_motor_sensor + 4]
   # Gyroscope:         sensordata[dim_motor_sensor + 4 : dim_motor_sensor + 7]
   # Accelerometer:     sensordata[dim_motor_sensor + 7 : dim_motor_sensor + 10]
   ```
   The reference implementation dynamically detects IMU and frame sensors by name (`mj_id2name`). The engineer should verify that the sensor order in `g1_29dof.xml` matches these offsets. For base position/velocity, check whether the model has `frame_pos`/`frame_vel` sensors or use `mj_data.qpos`/`mj_data.qvel` directly.

2. **Command application** (impedance control, from reference):
   ```python
   mj_data.ctrl[i] = cmd.joint_torques[i] + \
       cmd.kp[i] * (cmd.joint_positions[i] - sensor_pos[i]) + \
       cmd.kd[i] * (cmd.joint_velocities[i] - sensor_vel[i])
   ```

3. **DDS publishing:** A background `RecurrentThread` publishes `LowState_` on `rt/lowstate` at sim frequency for external tools. The control loop does **not** use DDS internally — it calls `send_command()` / `get_state()` directly on the `SimRobot` object. DDS is for external monitoring only.

**Tests (`tests/test_sim_robot.py`):**
- `test_sim_robot_init`: SimRobot initializes without crashing
- `test_sim_robot_n_dof_29`: 29-DOF model has n_dof == 29
- `test_sim_robot_get_state_shape`: State arrays have correct shapes
- `test_sim_robot_reset`: After reset, joint positions match home position
- `test_sim_robot_step`: After stepping, simulation time advances
- `test_sim_robot_gravity`: Robot falls when no commands are sent (base_position z decreases)
- `test_sim_robot_damping_holds`: With damping command, robot velocity decreases over time
- `test_sim_robot_send_command_shape`: Send a command with correct shapes, no crash
- `test_sim_robot_imu_upright`: After reset, IMU quaternion is approximately [1,0,0,0]
- `test_sim_robot_connect_disconnect`: Connect and disconnect without crash or leak
- `test_sim_robot_impedance_control_values`: Set known sensor positions/velocities (via `mj_data.sensordata`), send a command with known `kp, kd, joint_positions, joint_velocities, joint_torques`, then verify `mj_data.ctrl[i]` matches hand-computed `tau_ff + kp*(q_des - q) + kd*(dq_des - dq)` for each joint
- `test_sim_robot_sensor_mapping_correctness`: After reset, verify `get_state().joint_positions` matches `mj_data.sensordata[0:29]`, `get_state().joint_velocities` matches `sensordata[29:58]`, and IMU quaternion matches `sensordata[87:91]`
- `test_sim_robot_23dof`: Load 23-DOF scene, verify `n_dof == 23`, `get_state()` shapes are `(23,)`, and `step()` works
- `test_sim_robot_substep_count`: With `sim_frequency=200, policy_frequency=50`, verify `step()` advances MuJoCo time by exactly `4 * 0.005 = 0.02` seconds
- `test_sim_robot_base_position`: After reset, verify `get_state().base_position` matches expected initial position. Step with gravity, verify base_position Z decreases.
- `test_sim_robot_reset_custom_state`: Call `reset(initial_state=custom)`, verify joint positions match the custom state
- `test_sim_robot_dds_publish_mock`: Mock the DDS `ChannelPublisher`, call `connect()`, wait briefly, verify `publish()` was called with a valid `LowState_` message containing the correct number of motor states

**Run tests:** `pytest tests/test_sim_robot.py -v`

**Note:** These tests require MuJoCo to be installed. If running in CI without MuJoCo, mark them with `@pytest.mark.skipif(not HAS_MUJOCO, ...)`.

### Task 5.3: Validate DDS Communication (Manual Test)

This is a manual validation step, not automated. The engineer should:

1. Start `SimRobot` in one terminal
2. In another terminal, write a small script that subscribes to `rt/lowstate` and prints IMU data
3. Verify that state messages are received at the expected frequency
4. Send a `LowCmd` message and verify the robot responds

**Document results** in a test log or comments in the code.

---

## Phase 6: Control Loop

The control loop ties together the policy, robot interface, and safety system. It is the most integration-heavy component.

### Task 6.1: Implement `Controller` (`src/control/controller.py`)

**Contents:**
```python
class Controller:
    def __init__(self, robot: RobotInterface, policy: PolicyInterface,
                 safety: SafetyController, joint_mapper: JointMapper,
                 obs_builder: Optional[ObservationBuilder], config: Config): ...
    # obs_builder is None for BeyondMimic (policy builds its own observations)

    def start(self) -> None:
        """Start the control loop in a new thread."""

    def stop(self) -> None:
        """Stop the control loop."""

    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        """Set velocity command (thread-safe)."""

    def get_telemetry(self) -> dict:
        """Get latest telemetry data (thread-safe, non-blocking)."""

    def reload_policy(self, policy_path: str) -> None:
        """Load a new ONNX policy while the control loop is stopped.
        Thread-safe. Stops control loop if running, loads new policy, resets state.
        Called by Viser dropdown (Docker) or keyboard N/P (Metal)."""

    def handle_key(self, key: str) -> None:
        """Handle keyboard input (shared interface with Metal plan).
        Called by Viser keyboard events (Docker) or GLFW key_callback (Metal).
        Keys: 'space', 'r', 'e', 'c', 'w', 's', 'a', 'd', 'q', 'z', 'x', 'm', '1', '2', '3'
        See SPEC section 6.4 for full key mapping."""

    @property
    def is_running(self) -> bool:
        """Whether the control loop is active."""

    def _control_loop(self) -> None:
        """Main control loop running at policy_frequency."""
```

**Note on `handle_key()`:** This method provides a unified keyboard input interface shared between both plans. In the Docker plan, Viser keyboard events are translated to key strings and passed to `handle_key()`. In the Metal plan, GLFW keycodes are translated to the same key strings via a `GLFW_KEY_MAP`. The Controller handles velocity command updates (arrow keys), E-stop (Backspace/Enter), start/stop (Space), reset (Delete), policy cycling (=/−), etc. Key names are chosen to avoid conflicts with MuJoCo's built-in viewer shortcuts (which use most letter keys). This ensures the same keyboard behavior regardless of UI frontend.

**Control loop pseudocode:**
```python
def _control_loop(self):
    last_action = np.zeros(self.joint_mapper.n_controlled)
    while self._running:
        loop_start = time.perf_counter()

        # 1. Check E-stop
        if self.safety.state == SystemState.ESTOP:
            state = self.robot.get_state()
            cmd = self.safety.get_damping_command(state)
            self.robot.send_command(cmd)
            self._sleep_until_next_tick(loop_start)
            continue

        # 2. Skip if not RUNNING
        if self.safety.state != SystemState.RUNNING:
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

        # 6. Build command (PD control law)
        cmd = self._build_command(state, action)

        # 7. Send command
        self.robot.send_command(cmd)

        # 8. Step simulation (no-op for real robot)
        self.robot.step()

        # 9. Store for next iteration
        last_action = action.copy()

        # 10. Log
        self._log_step(state, obs, action, cmd, inference_time, loop_start)

        # 11. Sleep to maintain frequency
        self._sleep_until_next_tick(loop_start)
```

**Command building (`_build_command`):**
For controlled joints (IsaacLab):
```
target_pos[i] = q_home[i] + Ka[i] * action[i]
kp[i] = config.kp[i]
kd[i] = config.kd[i]
tau[i] = 0 (no feedforward)
```

For non-controlled joints (damping mode):
```
target_pos[i] = current_pos[i]  (zero position error)
kp[i] = 0
kd[i] = config.kd_damp
tau[i] = 0
```

For BeyondMimic controlled joints:
```
target_pos[i] = target_q[i] + Ka[i] * action[i]
kp[i] = config.kp[i]  (or metadata kp if use_onnx_metadata)
kd[i] = config.kd[i]  (or metadata kd)
dq_target[i] = target_dq[i]
tau[i] = 0
```

**BeyondMimic end-of-trajectory handling:**
When BeyondMimic's time_step exceeds trajectory length:
1. Capture final joint positions
2. Linearly interpolate from final positions to `q_home` over 2 seconds (100 steps at 50 Hz)
3. Enter STOPPED state after interpolation completes

**Tests (`tests/test_controller.py`):**

*Command building (value-level):*
- `test_controller_init`: Controller initializes without starting
- `test_build_command_isaaclab_values`: With `q_home=0.5, Ka=0.3, action=1.0`, verify `target_pos = 0.5 + 0.3 * 1.0 = 0.8`, `kp = config.kp`, `kd = config.kd`, `dq_target = 0`, `tau = 0`
- `test_build_command_beyondmimic_values`: With `target_q=0.2, Ka=0.3, action=1.0, target_dq=0.5`, verify `target_pos = 0.2 + 0.3 * 1.0 = 0.5`, `dq_target = 0.5`, `kp = metadata_kp`, `kd = metadata_kd`
- `test_build_command_damping`: Non-controlled joints get `target_pos=current_pos, kp=0, kd=kd_damp, dq_target=0, tau=0`

*Safety integration:*
- `test_estop_sends_damping`: In ESTOP state, verify damping command is sent
- `test_control_loop_exception_triggers_estop`: Mock policy to raise `RuntimeError` during `get_action()`. Verify controller enters ESTOP state (not crash).

*Velocity command and telemetry:*
- `test_velocity_command_thread_safe`: Set velocity from one thread, read from another, no corruption
- `test_telemetry_updates`: After running, telemetry dict contains expected keys with reasonable values
- `test_control_loop_timing`: Control loop runs at approximately 50 Hz (allow 10% tolerance)

*Key handling (complete coverage):*
- `test_handle_key_space_toggles`: Space from IDLE -> RUNNING, Space from RUNNING -> STOPPED
- `test_handle_key_estop`: 'e' triggers E-stop, verify state == ESTOP
- `test_handle_key_clear_estop`: 'c' from ESTOP -> STOPPED
- `test_handle_key_reset`: 'r' calls `robot.reset()` (mock and verify)
- `test_handle_key_velocity_wasd`: W increases vx by 0.1, S decreases, A increases vy, D decreases
- `test_handle_key_velocity_clamps`: After 20 W presses, vx is clamped to 1.0 (not 2.0)
- `test_handle_key_qz_yaw`: Q increases yaw by 0.1, Z decreases, clamped to [-1.0, 1.0]
- `test_handle_key_x_zeros_velocity`: X zeros all velocity components
- `test_handle_key_unknown_noop`: Unknown key (e.g., 'j') does nothing, no error

*Policy reloading:*
- `test_reload_policy_while_stopped`: `reload_policy("new.onnx")` loads successfully, resets state
- `test_reload_policy_while_running`: If RUNNING, `reload_policy()` stops first, loads, resets (verify stop -> load -> reset sequence)
- `test_reload_policy_invalid_path`: Invalid path raises error, original policy is preserved

*BeyondMimic trajectory:*
- `test_beyondmimic_end_of_trajectory`: When `time_step` exceeds trajectory length, verify: (1) final positions captured, (2) linear interpolation to `q_home` over 2s/100 steps, (3) state transitions to STOPPED after interpolation

*Integration:*
- `test_control_loop_lifecycle`: Create controller with mock robot/policy, `start()` -> verify running -> `stop()` -> verify stopped. No exceptions.
- `test_control_loop_stopped_still_steps`: In STOPPED state, verify `robot.step()` is still called (sim keeps advancing) but no command is sent

**Run tests:** `pytest tests/test_controller.py -v`

---

## Phase 7: Visualization

Viser visualization can be developed in parallel with earlier phases since it has minimal dependencies on the control system. It reads shared state but does not write to it.

### Task 7.1: Implement Viser Server (`src/viz/server.py`)

**Contents:**
```python
class ViserServer:
    def __init__(self, config: ViserConfig): ...

    def start(self) -> None:
        """Start the Viser server. Non-blocking."""

    def stop(self) -> None:
        """Stop the server."""

    @property
    def server(self) -> viser.ViserServer:
        """Access the underlying viser server for adding scene elements."""
```

**Key details:**
- Bind to `config.host` (`0.0.0.0`) and `config.port` (8080)
- Server runs in its own thread (Viser manages this internally)
- Scene setup (ground plane, lighting, coordinate frame) done at startup

**Acceptance criteria:**
- Server starts and browser can connect to `http://localhost:8080`
- Ground plane is visible

### Task 7.2: Implement Robot Visualization (`src/viz/robot_viz.py`)

**Contents:**
```python
class RobotVisualizer:
    def __init__(self, server: viser.ViserServer, model_path: str): ...

    def load_meshes(self) -> None:
        """Load robot meshes from MuJoCo model into Viser scene."""

    def load_collision_geometry(self) -> None:
        """Load collision geometry as wireframe/transparent."""

    def update(self, robot_state: RobotState) -> None:
        """Update robot visualization with new state. Called from viz thread."""

    def set_render_mode(self, mode: str) -> None:
        """Switch between 'mesh' and 'collision' rendering."""
```

**Key implementation details:**
- Parse MuJoCo model XML or use MuJoCo Python bindings to get mesh file paths, geometry shapes, and body transforms
- Use `viser.SceneApi.add_mesh()` for visual meshes
- For collision geometry, use `viser.SceneApi.add_box()`, `add_sphere()`, `add_capsule()` etc.
- `update()` must be efficient — update transforms only, don't re-add meshes
- Toggle visibility when switching render mode

**Reference:** Study how `mjlab` (https://github.com/mujocolab/mjlab) renders MuJoCo models in Viser. Borrow patterns for mesh loading and transform updates.

**Acceptance criteria:**
- Robot appears in Viser scene
- Joint angles update when `update()` is called with different states
- Mesh/collision toggle works

### Task 7.3: Implement Control Panel UI (`src/viz/ui.py`)

**Contents:**
```python
class ControlPanelUI:
    def __init__(self, server: viser.ViserServer, controller: Controller,
                 policy_dir: str): ...

    def setup(self) -> None:
        """Create all UI elements."""

    def _update_telemetry(self) -> None:
        """Periodically update telemetry display. Called by timer."""
```

**UI elements to create (from SPEC section 6.3):**
- Status indicator (text: IDLE / RUNNING / STOPPED / E-STOP with color)
- START button -> calls `controller.safety.start()` + `controller.start()`
- STOP button -> calls `controller.safety.stop()` + `controller.stop()`
- RESET button -> calls `controller.robot.reset()`
- E-STOP button (large, red) -> calls `controller.safety.estop()`
- CLEAR E-STOP button -> calls `controller.safety.clear_estop()`
- Policy dropdown: List ONNX files from `policy_dir`, on change reload policy
- Velocity command sliders: Vx [-1.0, 1.0], Vy [-0.5, 0.5], Yaw [-1.0, 1.0]
- Camera view dropdown: Follow / Fixed / Orbit
- Render mode toggle: Mesh / Collision
- Telemetry panel: Base height, base velocity, policy Hz, sim Hz

**Keyboard shortcuts (from SPEC section 6.4):**
- Investigate Viser's keyboard event support. If Viser doesn't support global keyboard shortcuts natively, document this limitation and implement via the GUI only. The engineer should check `viser.SceneApi` for keyboard event callbacks.
- If keyboard shortcuts are supported, translate Viser keyboard events to key name strings and call `controller.handle_key(key_name)`. This is the same interface used by the Metal plan's GLFW `key_callback`. Key names: `"space"`, `"up"`, `"down"`, `"left"`, `"right"`, `"comma"`, `"period"`, `"slash"`, `"backspace"`, `"enter"`, `"delete"`, `"equal"`, `"minus"`. These avoid conflicts with MuJoCo's built-in letter-key shortcuts.
- The `handle_key()` method on Controller handles all keyboard actions (velocity commands, E-stop, start/stop, reset, etc.) so the UI layer only needs to translate platform-specific key events to string names.

**Acceptance criteria:**
- All UI elements render in the browser
- Buttons trigger correct state transitions
- Sliders update velocity command
- Telemetry updates periodically
- Policy dropdown lists available ONNX files

### Task 7.4: Implement Camera Controls

Implement the three camera modes from SPEC section 6.2:
1. **Follow**: Camera tracks robot base position, maintaining relative offset
2. **Fixed**: Camera at a fixed world position, looking at origin
3. **Orbit**: User-controlled orbit (Viser's default behavior)

**Implementation note:** Viser has built-in camera control. The Follow mode requires updating the camera target position each frame to track the robot. The engineer should use `server.scene.set_camera_look_at()` or similar API.

**Tests (`tests/test_viz.py`):**

*Server and visualization (mock-based):*
- `test_viser_server_lifecycle`: Mock `viser.ViserServer`, verify `start()` and `stop()` don't crash
- `test_robot_visualizer_update`: Mock server, create `RobotVisualizer`, call `update(state)`. Verify transform update methods are called (no crash).
- `test_robot_visualizer_render_mode_toggle`: Call `set_render_mode("collision")` then `set_render_mode("mesh")`. Verify visibility toggles.
- `test_camera_mode_enum`: Camera modes are valid strings ("follow", "fixed", "orbit")

*Control panel UI (mock-based):*
- `test_ui_element_creation`: Mock viser server, verify `ControlPanelUI.setup()` creates expected UI elements
- `test_ui_start_button_calls_controller`: Mock controller, simulate START button callback, verify `controller.safety.start()` and `controller.start()` are called
- `test_ui_estop_button_calls_controller`: Simulate E-STOP button, verify `controller.safety.estop()` called
- `test_ui_policy_dropdown_lists_files`: Create temp dir with 3 `.onnx` files, verify dropdown options list all three
- `test_ui_policy_dropdown_triggers_reload`: Simulate dropdown change, verify `controller.reload_policy()` called with selected path
- `test_ui_velocity_sliders`: Simulate slider change, verify `controller.set_velocity_command()` called with correct values

*Integration testing is primarily manual (visual verification in browser)*

---

## Phase 8: Logging System

### Task 8.1: Implement Data Logger (`src/logging/logger.py`)

**Contents:**
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

    def _write_metadata(self, config: Config) -> None:
        """Write run metadata to metadata.yaml."""
```

**File structure (from SPEC section 8.2):**
```
logs/{timestamp}_{mode}_{policy_name}/
├── metadata.yaml      # Run configuration snapshot
├── data.hdf5          # Time-series data (or data.npz if format=npz)
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

**Performance consideration:** HDF5 writes should be buffered. Use a write buffer (e.g., accumulate 100 steps, then flush). This prevents I/O from blocking the control loop. Alternatively, use a separate logging thread with a queue.

**Tests (`tests/test_logger.py`):**
- `test_logger_creates_directory`: Start creates the log directory
- `test_logger_writes_metadata`: metadata.yaml is created with config
- `test_logger_writes_data_hdf5`: Log 100 steps, verify HDF5 file has correct shapes
- `test_logger_writes_data_npz`: Log 100 steps, verify NPZ file has correct shapes
- `test_logger_writes_events`: Log start/stop events, verify events.json
- `test_logger_compression`: HDF5 file is gzip-compressed
- `test_logger_roundtrip_hdf5`: Write data via HDF5, read back, verify values match
- `test_logger_roundtrip_npz`: Write data via NPZ, read back, verify values match
- `test_logger_stop_prints_summary`: Stop logs summary stats (capture stdout)
- `test_logger_handles_empty_run`: Start then immediately stop, no crash
- `test_logger_log_event`: Call `log_event("estop", {"reason": "orientation"})`, verify event appears in `events.json` with timestamp
- `test_logger_metadata_contains_config`: Verify `metadata.yaml` contains the full config snapshot (robot variant, policy format, control gains, etc.)
- `test_logger_nonblocking`: Log 100 steps, measure wall time. Verify logging overhead is <1ms per step (no I/O blocking on the control loop thread).

### Task 8.2: Implement Log Replay (`src/logging/replay.py` and `scripts/replay_log.py`)

**Contents of `replay.py`:**
```python
class LogReplay:
    def __init__(self, log_dir: str): ...

    def load(self) -> None:
        """Load log files."""

    @property
    def metadata(self) -> dict: ...
    @property
    def duration(self) -> float: ...
    @property
    def n_steps(self) -> int: ...

    def get_state_at(self, step: int) -> RobotState: ...
    def get_observation_at(self, step: int) -> np.ndarray: ...
    def get_action_at(self, step: int) -> np.ndarray: ...

    def to_csv(self, output_path: str) -> None:
        """Export log data to CSV for external analysis."""

    def summary(self) -> str:
        """Print summary statistics."""
```

**Contents of `scripts/replay_log.py`:**
```python
# Standalone script (no Docker required)
# Usage: python scripts/replay_log.py logs/<run>/ [--format csv] [--output file] [--visualize]
```

**Acceptance criteria:**
- `replay_log.py` works on macOS without Docker
- CSV export produces a valid CSV file
- `--visualize` flag launches a Viser replay (optional, can be deferred)

**Tests (`tests/test_logger.py` - additions):**
- `test_replay_load`: Load a previously written log
- `test_replay_metadata`: Metadata matches what was written
- `test_replay_state_at`: State at step 50 matches what was logged at step 50
- `test_replay_observation_at`: `get_observation_at(step)` returns correct observation array matching what was logged
- `test_replay_action_at`: `get_action_at(step)` returns correct action array
- `test_replay_to_csv`: CSV output has correct columns and values
- `test_replay_summary`: Summary includes duration, step count, timing stats
- `test_replay_auto_detect_format_hdf5`: `LogReplay` with an HDF5 log directory auto-detects and loads HDF5 format
- `test_replay_auto_detect_format_npz`: `LogReplay` with an NPZ log directory auto-detects and loads NPZ format

**Run tests:** `pytest tests/test_logger.py -v`

---

## Phase 9: Real Robot Interface

**SAFETY WARNING:** Real robot code must be tested with extreme caution. All development should happen in simulation first. Real robot testing should start with the robot hanging from a support harness.

### Task 9.1: Implement `RealRobot` (`src/robot/real_robot.py`)

**Contents:**
```python
class RealRobot(RobotInterface):
    def __init__(self, config: Config): ...

    def connect(self) -> None:
        """Initialize DDS on the specified network interface.
        Subscribe to rt/lowstate. Prepare to publish rt/lowcmd."""

    def disconnect(self) -> None:
        """Stop publishing, clean up DDS."""

    def get_state(self) -> RobotState:
        """Return latest state from DDS subscription. Thread-safe."""

    def send_command(self, cmd: RobotCommand) -> None:
        """Publish LowCmd to rt/lowcmd via DDS."""

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
   from src.compat import resolve_network_interface
   interface = resolve_network_interface(config.network.interface)
   ChannelFactoryInitialize(domain_id=0, interface=interface)
   ```

2. **State subscription:**
   - Subscribe to `rt/lowstate` using `unitree_hg` IDL types
   - Store latest state in a thread-safe buffer (Lock + copy)
   - State callback runs on DDS internal thread

3. **Command publishing:**
   - Publish to `rt/lowcmd` using `unitree_hg` IDL types
   - Set `motor_cmd[i].mode = 0x01` (PMSM servo mode) for all motors
   - CRC calculation is required — use the SDK's CRC utility

4. **Startup checks (from SPEC section 5.4):**
   - Verify DDS connection (receive at least one state message within timeout)
   - Check orientation before allowing policy start (via `safety.check_orientation()`)
   - Print IMU data and orientation check result to console

5. **Communication monitoring:**
   - Track time since last received state message
   - If timeout exceeded (e.g., 100ms), trigger E-stop
   - Log DDS latency statistics

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

**Manual test checklist:**
- [ ] DDS subscription receives state messages from robot
- [ ] State message rate is approximately 500 Hz (Unitree default)
- [ ] Command publishing moves robot joints (test with small gains, robot in harness)
- [ ] Orientation check passes with upright robot
- [ ] Orientation check fails with tilted robot
- [ ] Communication timeout triggers E-stop

---

## Phase 10: CLI and Entry Point

### Task 10.1: Implement Main Entry Point (`src/main.py`)

**Contents:**
```python
def main():
    parser = argparse.ArgumentParser(description="Unitree G1 Deployment Stack")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Sim mode
    sim_parser = subparsers.add_parser("sim")
    add_common_args(sim_parser)

    # Real mode
    real_parser = subparsers.add_parser("real")
    add_common_args(real_parser)
    real_parser.add_argument("--interface", required=True, help="Network interface")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    # Override from CLI
    apply_cli_overrides(config, args)

    # Resolve model path from variant
    variant = config.robot.variant  # e.g. "g1_29dof"
    model_path = f"assets/robots/g1/{variant}.xml"

    # Create components
    if args.mode == "sim":
        robot = SimRobot(config)
    else:
        robot = RealRobot(config)

    # Resolve policy format
    policy_format = config.policy.format or detect_policy_format(args.policy)

    # Create joint mapper (resolves joint names from config)
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

    # Create observation builder (IsaacLab only; BeyondMimic builds its own)
    obs_builder = None
    if policy_format == "isaaclab":
        # Config is primary; --no-est CLI flag overrides to False
        use_estimator = config.policy.use_estimator
        if getattr(args, 'no_est', False):
            use_estimator = False
        obs_builder = ObservationBuilder(joint_mapper, config.control,
                                         use_estimator=use_estimator)
        assert policy.observation_dim == obs_builder.observation_dim, \
            f"Policy expects obs_dim={policy.observation_dim}, builder produces {obs_builder.observation_dim}"

    # Create safety controller
    safety = SafetyController(config.control, robot.n_dof)

    # Create controller
    controller = Controller(robot, policy, safety, joint_mapper, obs_builder, config)

    # Create visualization (optional)
    if not args.no_viser:
        viz_server = ViserServer(config.viser)
        robot_viz = RobotVisualizer(viz_server.server, model_path)
        ui = ControlPanelUI(viz_server.server, controller, args.policy_dir)
        viz_server.start()
        robot_viz.load_meshes()
        ui.setup()

    # Generate run name for logging
    policy_name = Path(args.policy).stem
    run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{args.mode}_{policy_name}"

    # Create logger
    logger = DataLogger(config.logging, run_name, args.log_dir)

    # Connect and run
    robot.connect()
    controller.start()
    logger.start()

    # Block until Ctrl+C
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        robot.disconnect()
        logger.stop()
```

**CLI arguments (from SPEC section 10.4):**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | path | `configs/default.yaml` | Configuration file |
| `--policy` | path | required | ONNX policy file |
| `--policy-dir` | path | `policies/` | Directory of ONNX files for UI dropdown |
| `--robot` | str | `g1_29dof` | Robot variant |
| `--interface` | str | — | Network interface (real only) |
| `--domain-id` | int | 1 (sim), 0 (real) | DDS domain ID |
| `--viser-port` | int | 8080 | Viser port |
| `--no-viser` | flag | false | Headless mode |
| `--log-dir` | path | `logs/` | Log output directory |
| `--no-est` | flag | false | Override `policy.use_estimator` to false (omit `base_lin_vel`) |

**Tests (`tests/test_main.py`):**

*Argument parsing:*
- `test_parse_sim_args`: `parse_args(["sim", "--policy", "test.onnx"])` succeeds with `mode="sim"`, `policy="test.onnx"`
- `test_parse_real_args`: `parse_args(["real", "--policy", "test.onnx", "--interface", "eth0"])` succeeds with `mode="real"`, `interface="eth0"`
- `test_missing_policy_errors`: `parse_args(["sim"])` without `--policy` raises `SystemExit` (argparse required arg)
- `test_real_requires_interface`: `parse_args(["real", "--policy", "test.onnx"])` without `--interface` raises `SystemExit`
- `test_no_viser_flag`: `parse_args(["sim", "--policy", "p.onnx", "--no-viser"])` sets `no_viser=True`
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

### Task 10.2: Implement Shell Scripts

**`scripts/run_sim.sh`:**
```bash
#!/bin/bash
# Convenience wrapper for simulation mode
python -m src.main sim --config configs/default.yaml "$@"
```

**`scripts/run_real.sh`:**
```bash
#!/bin/bash
# Convenience wrapper for real robot mode
python -m src.main real --config configs/default.yaml "$@"
```

**`scripts/replay_log.py`:**
Already implemented in Task 8.2.

---

## Phase 11: Docker Configuration

### Task 11.1: Write Dockerfile (`docker/Dockerfile`)

**Base:** `ubuntu:20.04`

**Stages:**
1. System dependencies (apt)
2. Python (3.10+ via deadsnakes PPA if needed, or system Python 3.8)
3. CycloneDDS build (requires CMake, gcc)
4. Python packages (pip install from requirements.txt)
5. unitree_sdk2_python (pip install from GitHub)
6. Copy application code

**Key environment variables:**
```dockerfile
ENV MUJOCO_GL=egl
ENV DEBIAN_FRONTEND=noninteractive
```

**Key apt packages:**
```dockerfile
RUN apt-get install -y \
    python3 python3-pip python3-dev \
    cmake gcc g++ \
    libegl1-mesa-dev libgl1-mesa-dev \
    libglib2.0-0 \
    git
```

**Acceptance criteria:**
- `docker build -t unitree_launcher .` succeeds
- `docker run unitree_launcher python -c "import mujoco; import viser; import onnxruntime; print('OK')"` prints OK
- `docker run unitree_launcher python -c "import unitree_sdk2py; print('OK')"` prints OK

### Task 11.2: Write docker-compose.yml (`docker/docker-compose.yml`)

```yaml
version: "3.8"
services:
  unitree:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../src:/app/src
      - ../configs:/app/configs
      - ../assets:/app/assets
      - ../policies:/app/policies
      - ../logs:/app/logs
    environment:
      - MUJOCO_GL=egl
    ports:
      - "8080:8080"  # Viser web UI
    command: python -m src.main sim --config configs/default.yaml --policy policies/your_policy.onnx
```

**Platform notes:**
- **Simulation mode** works on both Linux and macOS Docker. DDS uses the `lo` interface inside the container, which works regardless of host networking.
- **Viser** is accessible from the host browser at `http://localhost:8080` via the port mapping above.
- **Real robot mode** requires Linux host with `network_mode: host` for DDS multicast to reach the robot over Ethernet. Add a separate compose profile for this:
  ```yaml
  # docker-compose.real.yml
  services:
    unitree:
      network_mode: host  # Required for DDS to reach physical robot
      # ports: is ignored when network_mode: host (host ports are directly accessible)
  ```
  Use: `docker compose -f docker-compose.yml -f docker-compose.real.yml up`
- **macOS does not support real robot deployment** (Docker runs in a VM, `network_mode: host` doesn't provide true host networking).

Document these platform differences in the README.

### Task 11.3: Test Docker on All Target Platforms

**Manual test checklist:**
- [ ] `docker build` succeeds on Ubuntu 20.04
- [ ] `docker build` succeeds on Ubuntu 22.04
- [ ] `docker build` succeeds on Ubuntu 24.04
- [ ] `docker build` succeeds on macOS (Apple Silicon via Docker Desktop)
- [ ] `docker build` succeeds on macOS (Intel via Docker Desktop)
- [ ] Simulation runs inside container
- [ ] Viser accessible from host browser on all platforms
- [ ] Logs directory on host contains output after a run
- [ ] Code changes on host are reflected inside container (volume mounts)

---

## Phase 12: Integration Testing and Polish

### Task 12.1: End-to-End Simulation Test (IsaacLab)

Run a complete simulation session with a real IsaacLab ONNX policy:

1. Start the system: `python -m src.main sim --policy policies/<isaaclab_policy>.onnx`
2. Open Viser in browser
3. Click START
4. Set velocity command Vx = 0.3 m/s
5. Observe robot walking in simulation
6. Click STOP
7. Verify logs are generated
8. Replay logs with `scripts/replay_log.py`

**Expected outcome:** Robot walks forward in simulation. Telemetry shows ~50 Hz policy, ~200 Hz sim. Logs contain correct data.

**If the robot does not walk:** This could be a joint ordering issue, observation construction error, gain mismatch, or action scale problem. Debug systematically:
1. Print observation vector and verify each component matches expectations
2. Print action vector and verify magnitude is reasonable (<1.0 typically)
3. Compare computed torques with expected values from training environment
4. Check joint order matches what the policy was trained with

### Task 12.2: End-to-End Simulation Test (BeyondMimic)

Run a complete simulation session with a BeyondMimic ONNX policy:

1. Start: `python -m src.main sim --policy policies/<beyondmimic_policy>.onnx`
2. Open Viser
3. Click START
4. Observe robot performing the motion trajectory
5. When trajectory ends, verify smooth interpolation to home position
6. Verify system enters STOPPED state

**Expected outcome:** Robot performs tracked motion, then smoothly returns to standing.

### Task 12.3: E-Stop Integration Test

1. Start simulation with a running policy
2. Press E-STOP button in UI (or E key)
3. Verify robot enters damping mode (slows to stop)
4. Verify status shows E-STOP
5. Verify policy inference stops
6. Press CLEAR E-STOP
7. Verify system enters STOPPED state
8. Start policy again
9. Verify robot resumes walking

### Task 12.4: 23-DOF Smoke Test

1. Start with 23-DOF config: `python -m src.main sim --robot g1_23dof --config configs/g1_23dof.yaml --policy policies/<23dof_policy>.onnx`
2. Verify MuJoCo model loads
3. Verify joint count is 23
4. Verify home position is correct (visual check)
5. If no 23-DOF policy available, just verify the robot stands in damping mode

### Task 12.5: Docker End-to-End Test

Repeat Task 12.1 inside Docker container:
```bash
docker-compose up -d
docker exec -it unitree_launcher python -m src.main sim --policy policies/<policy>.onnx
```

Verify Viser accessible from host browser at `http://localhost:8080`.

### Task 12.6: Automated Headless Integration Test

**File:** `tests/test_integration.py`

Fully automated end-to-end tests that run the system with `--no-viser` using mock/test components. No manual intervention required.

```python
def test_headless_sim_isaaclab_100_steps():
    """Full pipeline: config -> SimRobot -> IsaacLabPolicy -> Controller -> Logger.
    Run 100 steps in headless mode with --no-viser, verify no crash and logs are produced."""

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

### Task 12.7: Comprehensive Unit Test Run

Run the full test suite and ensure all tests pass:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Target:** 100% of unit tests pass. Coverage target: >80% line coverage for all modules.

**Test count expectations by module:**
| Module | Approximate Count | Key Coverage |
|--------|------------------|-------------|
| `test_config.py` | 21-24 | Constants, data structures, config validation, merge semantics |
| `test_compat.py` | 9-11 | RecurrentThread lifecycle, exceptions, slow targets, network resolution |
| `test_joint_mapper.py` | 21-24 | Mapping, reordering, roundtrips, 23-DOF, error paths |
| `test_observations.py` | 19-22 | Gravity, velocity transforms, segment positions, 23-DOF |
| `test_safety.py` | 26-30 | State machine, damping, orientation, thread safety, limit clamping |
| `test_isaaclab_policy.py` | 12-14 | Load, inference, determinism, corruption, format detection |
| `test_beyondmimic_policy.py` | 20-23 | Obs construction values, 6D rotation, history, metadata errors |
| `test_sim_robot.py` | 17-20 | Impedance values, sensor mapping, 23-DOF, substeps, DDS mock |
| `test_controller.py` | 26-30 | Command values, all keys, reload, trajectory end, lifecycle |
| `test_viz.py` | 10-13 | Server lifecycle, UI callbacks, policy dropdown, slider values |
| `test_logger.py` | 20-23 | Both formats, roundtrips, events, auto-detect, replay |
| `test_real_robot.py` | 14-16 | Command construction, state mapping, watchdog, CRC, thread safety |
| `test_main.py` | 13-15 | Arg parsing, domain ID, variant resolution, wiring |
| `test_integration.py` | 5-7 | Headless E2E, policy reload, estop recovery, performance |
| **Total** | **~233-272** |

---

## Phase 13: Documentation

### Task 13.1: Write README.md

The README should cover:
- Project overview and purpose
- Quick start (Docker) — for this plan
- Quick start (bare metal / macOS) — reference to PLAN_METAL.md workflow
- Usage: simulation mode
- Usage: real robot mode
- Configuration reference
- Log replay
- Platform-specific notes (macOS vs Linux Docker networking)
- Troubleshooting (common DDS issues, EGL rendering issues)
- Architecture overview (shared modules, plan-specific components)

### Task 13.2: Inline Code Documentation

- Every public class and method should have a docstring
- Complex algorithms (quaternion math, joint mapping) should have inline comments explaining the math
- The control law should be documented with reference to the spec section

---

## Task Execution Order Summary

See also the **Execution Guide** section at the top of this document for the dependency graph and parallelization strategy. The key constraint is: **do not start a phase until all dependencies from prior phases are passing tests.**

| Order | Task | Dependencies | Parallel? | Key Test File |
|-------|------|-------------|-----------|---------------|
| 1 | 0.1 Directory structure | None | — | — |
| 2 | 0.2 Copy MuJoCo models | 0.1 | ‖ with step 3 | — |
| 3 | 0.3 pyproject.toml + requirements | 0.1 | ‖ with step 2 | — |
| 4 | 0.4 conftest.py fixtures | 0.3 | — | — |
| 5 | 1.1 Robot constants | 0.4 | — | `test_config.py` |
| 6 | 1.2 State/Command dataclasses | 1.1 | ‖ with step 7 | `test_config.py` |
| 7 | 1.3 Abstract RobotInterface | 1.2 | ‖ with step 6 | — |
| 8 | 1.4 Abstract PolicyInterface | — | ‖ with steps 5-7 | — |
| 9 | 1.5 Config loading + validation | 1.1 | — | `test_config.py` |
| 10 | 1.6 YAML config files | 1.5 | ‖ with step 10.5 | `test_config.py` |
| 10.5 | 1.7 compat.py | — | ‖ with step 10 | `test_compat.py` |
| 11 | 2.1 JointMapper | 1.1 | ‖ with steps 13, 18, 26 | `test_joint_mapper.py` |
| 12 | 2.2 ObservationBuilder | 2.1, 1.2 | ‖ with steps 13, 18, 26 | `test_observations.py` |
| 13 | 3.1 Safety state machine | 1.2, 1.5 | ‖ with steps 11, 18, 26 | `test_safety.py` |
| 14 | 4.1 Test ONNX fixtures | 0.3, 0.4 | ‖ with steps 11-13 | — |
| 15 | 4.2 Policy format detection | 4.1 | — | `test_isaaclab_policy.py` |
| 16 | 4.3 IsaacLab policy | 2.1, 4.1, 1.4 | ‖ with step 17 | `test_isaaclab_policy.py` |
| 17 | 4.4 BeyondMimic policy | 2.1, 4.1, 1.4 | ‖ with step 16 | `test_beyondmimic_policy.py` |
| 18 | 5.1 MuJoCo scene files | 0.2 | ‖ with steps 11-17 | — |
| 19 | 5.2 SimRobot | 5.1, 1.3, 1.5 | — | `test_sim_robot.py` |
| 20 | 5.3 DDS validation (manual) | 5.2 | — | — |
| 21 | 6.1 Controller (+policy reload) | 5.2, 4.3/4.4, 3.1, 2.2 | — | `test_controller.py` |
| 22 | 7.1 Viser server | — | ‖ with steps 26-28 | — |
| 23 | 7.2 Robot visualization | 7.1, 5.1 | ‖ with steps 24-25 | `test_viz.py` |
| 24 | 7.3 Control panel UI (+policy dropdown) | 7.1, 6.1 | ‖ with steps 23, 25 | `test_viz.py` |
| 25 | 7.4 Camera controls | 7.1 | ‖ with steps 23-24 | — |
| 26 | 8.1 Data logger (HDF5 + NPZ) | 1.2 | ‖ with steps 11-25 | `test_logger.py` |
| 27 | 8.2 Log replay | 8.1 | — | `test_logger.py` |
| 28 | 9.1 RealRobot | 1.3, 1.5 | ‖ with steps 22-27 | `test_real_robot.py` |
| 29 | 10.1 Main entry point | all above | — | `test_main.py` |
| 30 | 10.2 Shell scripts | 10.1 | — | — |
| 31 | 11.1 Dockerfile | 0.3 | — | — |
| 32 | 11.2 docker-compose.yml | 11.1 | — | — |
| 33 | 11.3 Docker platform testing | 11.2 | — | manual |
| 34 | 12.1-12.7 Integration tests | all above | — | `test_integration.py` + manual |
| 35 | 13.1-13.2 Documentation | all above | — | — |

---

## Open Items Requiring Engineer Investigation

| ID | Item | When to Investigate | Impact |
|----|------|-------------------|--------|
| OI-1 | **Viser keyboard shortcuts**: Does Viser support global keyboard events? | Task 7.3 | If not supported, keyboard shortcuts are dropped and all interaction goes through GUI. |
| OI-2 | **EGL on macOS Docker**: Does MuJoCo with EGL work inside Docker Desktop on macOS? | Task 11.1 | If not, try `MUJOCO_GL=osmesa`. If that fails, disable MuJoCo rendering entirely (headless sim with Viser-only visualization). |
| OI-3 | **CycloneDDS build in Docker**: Does CycloneDDS compile cleanly in Ubuntu 20.04 Docker? | Task 11.1 | If build issues, check for prebuilt wheels or use a newer Ubuntu base. |
| OI-4 | **unitree_sdk2_python version pinning**: What version of the SDK is compatible with the G1 robot firmware? | Task 5.2 | Using wrong SDK version could send malformed commands. Check Unitree documentation. |
| OI-5 | **Viser + MuJoCo mesh loading**: How does mjlab load MuJoCo meshes into Viser? Are there utilities to borrow? | Task 7.2 | Significant implementation effort if no reusable code exists. |
| OI-6 | **BeyondMimic observation construction**: Now fully specified in SPEC section 4.3. Implementation should follow the 7-step approach in Task 4.4. Validate against a real BeyondMimic ONNX model when available. | Task 4.4 | Resolved — see SPEC section 4.3 for complete observation vector specification. |
| OI-7 | **DDS CRC calculation**: The `LowCmd` message requires CRC32 for the real robot. Does the Python SDK handle this automatically? | Task 9.1 | If not, the engineer must compute CRC manually before publishing. |
| OI-8 | **Docker host networking on macOS**: Verify that DDS simulation works inside Docker on macOS with `lo` interface. | Task 11.3 | If DDS multicast fails, may need to configure CycloneDDS XML to use unicast. |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Joint ordering bug sends wrong commands to real robot | Medium | **Critical** (hardware damage) | Exhaustive unit tests on JointMapper. Validate with known-good states before first real robot test. Start with robot in harness. |
| EGL doesn't work on macOS Docker | Medium | Medium | Fallback to OSMesa. If both fail, run MuJoCo headless without rendering (sim only, Viser for viz). |
| CycloneDDS build issues in Docker | Low | Medium | Try prebuilt wheels. If not available, try Ubuntu 22.04 base. |
| Policy frequency not maintained (>20ms per loop) | Low | High (real robot instability) | Profile control loop. Move ONNX inference to separate thread if needed. Use ONNX Runtime's thread pool settings. |
| Viser rendering performance too slow for real-time | Medium | Low | Visualization is decoupled from control. Reduce mesh detail or update rate if needed. Does not affect safety. |
| BeyondMimic observation construction incorrect | High | Medium | Study training code carefully. Compare observations between this system and the training environment. Log observations and verify offline. |

---

## Appendix A: Testing Philosophy

### Write Tests First (Where Practical)

For pure-logic modules (JointMapper, ObservationBuilder, SafetyController, Config), write the test file first based on the spec, then implement the module until tests pass. This ensures the spec is the source of truth, not the implementation.

For I/O-heavy modules (SimRobot, RealRobot, Viser), write the implementation first, then add tests to lock down the behavior.

### Test Granularity

- **Unit tests**: Test one function or method in isolation. Mock all dependencies. These must be fast (<100ms each).
- **Integration tests**: Test multiple modules working together (e.g., Controller with SimRobot and Policy). These can be slower.
- **Manual tests**: Require visual verification or physical hardware. Documented as checklists.

### Value-Level Testing for Safety-Critical Math

Any math formula that directly computes robot commands must have tests with **known numerical inputs and hand-computed expected outputs**. Shape/dtype tests are necessary but not sufficient. This includes:
- **Impedance control law**: `ctrl = tau_ff + kp*(q_des - q) + kd*(dq_des - dq)` in `SimRobot.send_command()`
- **IsaacLab command building**: `target_pos = q_home + Ka * action`
- **BeyondMimic command building**: `target_pos = target_q + Ka * action`, `dq_target = target_dq`
- **Projected gravity**: Quaternion -> gravity vector rotation
- **Body-relative transforms**: Body position/orientation relative to anchor body
- **Safety limit clamping**: Joint position, velocity, and torque clamping

### Thread Safety Testing

The system runs 3-4 concurrent threads (control loop, DDS publisher, Viser/main thread, optionally logger). Key concurrent access patterns must be tested:
- Safety state machine: concurrent `estop()` + `start()` + `stop()` calls
- Velocity command: writer thread + reader thread
- DDS state buffer: callback thread + `get_state()` caller

### Phase Gate Testing

Each phase must have sufficient automated tests to prove it works before moving on. The principle is: **if a phase's tests pass, the phase is complete.** This means:
- Every public method must have at least one test
- Every error path documented in the code must have a test
- Every math formula must have a value-level test

### When to Run Tests

- After implementing any function: run that module's test file
- After completing a phase: run the full test suite (`pytest tests/ -v`)
- Before committing: run the full test suite
- Before any real robot test: run the full test suite AND verify joint mapping with a known-good state

### Test Naming Convention

```
test_{module}_{behavior}_{scenario}
```
Example: `test_joint_mapper_robot_to_observation_isaaclab_ordering`

---

## Appendix B: Coding Standards

- **Type hints** on all function signatures
- **Docstrings** on all public classes and methods (Google style)
- **No wildcard imports** (`from x import *`)
- **Constants in UPPER_SNAKE_CASE**, variables in `lower_snake_case`, classes in `PascalCase`
- **No magic numbers** in logic code — use named constants from config
- **Logging** via Python's `logging` module (not `print` statements, except for startup banners)
- **Line length**: 100 characters max
- **Formatter**: Use `black` if available (not required)

---

## Appendix C: Joint Name Reference Quick Table

For quick reference during development. The full tables are in SPEC.md sections 2.2-2.3.

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
