# Unitree G1 Deployment Stack - Technical Specification

## Note on Dual-Plan Support

This specification supports two implementation plans:

- **PLAN_METAL.md (Metal plan):** Native macOS bare metal with MuJoCo viewer. Primary development path.
- **PLAN_DOCKER.md (Docker plan):** Docker container with Viser browser-based UI. For CI, deployment, and collaborators without macOS.

Sections that differ between plans are tagged with **[Metal]** and **[Docker]**. Untagged sections are shared and must be implemented identically by both plans to ensure code compatibility.

Both plans share the same core `src/` modules (config, robot, policy, control, logging). The only plan-specific modules are:
- **[Docker only]:** `src/viz/` (Viser server, robot visualization, control panel UI)
- **[Metal only]:** Viewer integration lives in `src/main.py` (MuJoCo viewer launch + keyboard callbacks)
- **[Shared]:** `src/compat.py` (cross-platform RecurrentThread, loopback interface detection)

**Project documentation files (`SPEC.md`, `PLAN_DOCKER.md`, `PLAN_METAL.md`, `WORK.md`) must never be deleted.** They are the authoritative record of requirements, design decisions, and implementation plans. They should be updated in place as the project evolves.

---

## 1. Overview

### 1.1 Purpose
Build a unified deployment stack for the Unitree G1 humanoid robot that enables:
- Evaluation of RL policies in MuJoCo simulation
- Deployment of the same policies to the physical G1 robot
- Interactive visualization and control
  - **[Docker]** via browser-based Viser UI
  - **[Metal]** via MuJoCo native viewer with keyboard controls
- Cross-platform operation (macOS and Linux)
  - macOS: simulation and development only (no real robot connection)
  - Linux: simulation and real robot deployment (via Ethernet)
- Headless batch evaluations on Linux servers **[Metal]**

### 1.2 Success Criteria

**Shared:**
- [ ] ONNX policies can be loaded and evaluated at 50 Hz
- [ ] Seamless switching between simulation and real robot via CLI
- [ ] E-stop functionality works in both sim and real
- [ ] Logs are generated and can be replayed without Docker
- [ ] Runtime policy switching (load a new ONNX file while control loop runs)

**[Docker]:**
- [ ] Docker container builds and runs on both macOS and Ubuntu (20.04, 22.04, 24.04)
- [ ] Viser UI accessible from any device on local network
- [ ] Policy switching via Viser dropdown

**[Metal]:**
- [ ] Native macOS simulation with MuJoCo viewer runs on Apple Silicon
- [ ] CycloneDDS works on macOS loopback (lo0)
- [ ] Headless mode runs on Linux servers without display
- [ ] `--duration` and `--steps` flags auto-terminate headless evals
- [ ] WASD keyboard controls adjust velocity commands in viewer
- [ ] Policy switching via keyboard shortcut (N/P for next/prev in `--policy-dir`)

### 1.3 Non-Goals (Out of Scope for Initial Version)
- Dexterous hand control (Dex3-1)
- WiFi connectivity to robot (ethernet only)
- GPU-accelerated policy inference
- Tensorboard/W&B integration
- Non-flat terrain visualization
- Runtime policy switching in Metal plan (future work)

---

## 2. Robot Configuration

### 2.1 Supported Variants
| Variant | DOF | Default | Notes |
|---------|-----|---------|-------|
| G1-29   | 29  | Yes     | Full body with 3-DOF waist, 7-DOF arms |
| G1-23   | 23  | No      | Single-DOF torso, 5-DOF arms |

The system must support runtime selection of robot variant via configuration.

**Note**: The MuJoCo model files (`g1_29dof.xml`, `g1_23dof.xml`) should be copied from `reference/unitree_mujoco/unitree_robots/g1/` into `assets/robots/g1/` along with the `meshes/` directory. Both files contain 29-joint definitions for simulation; the 23-DOF constraint is enforced at the DDS/control layer by only commanding a subset of joints.

### 2.2 Joint Configuration (29-DOF)

Complete joint table with exact names for config files. **This is the robot-native order** used in MuJoCo and DDS.

**Note on Joint Name Mapping**: Three naming conventions exist across the system. The engineer must implement a joint name mapping system that translates between:
- **IsaacLab / MuJoCo joint names**: e.g., `left_hip_pitch_joint` (as defined in MJCF files and used by IsaacLab)
- **Config names**: e.g., `left_hip_pitch` (used in this spec, shorthand without `_joint` suffix)
- **DDS/IDL names**: e.g., `L_LEG_HIP_PITCH` (uppercase format used in Unitree SDK DDS messages)

IsaacLab uses the same naming convention as MuJoCo MJCF files (`left_hip_pitch_joint`). This mapping should be configurable via YAML and validated at startup.

| Index | Config Name | MuJoCo Joint | Torque Limit (Nm) | Position Range (rad) |
|-------|-------------|--------------|-------------------|----------------------|
| 0 | `left_hip_pitch` | `left_hip_pitch_joint` | 88 | [-2.53, 2.88] |
| 1 | `left_hip_roll` | `left_hip_roll_joint` | 88 | [-0.52, 2.97] |
| 2 | `left_hip_yaw` | `left_hip_yaw_joint` | 88 | [-2.76, 2.76] |
| 3 | `left_knee` | `left_knee_joint` | 139 | [-0.09, 2.88] |
| 4 | `left_ankle_pitch` | `left_ankle_pitch_joint` | 50 | [-0.87, 0.52] |
| 5 | `left_ankle_roll` | `left_ankle_roll_joint` | 50 | [-0.26, 0.26] |
| 6 | `right_hip_pitch` | `right_hip_pitch_joint` | 88 | [-2.53, 2.88] |
| 7 | `right_hip_roll` | `right_hip_roll_joint` | 88 | [-2.97, 0.52] |
| 8 | `right_hip_yaw` | `right_hip_yaw_joint` | 88 | [-2.76, 2.76] |
| 9 | `right_knee` | `right_knee_joint` | 139 | [-0.09, 2.88] |
| 10 | `right_ankle_pitch` | `right_ankle_pitch_joint` | 50 | [-0.87, 0.52] |
| 11 | `right_ankle_roll` | `right_ankle_roll_joint` | 50 | [-0.26, 0.26] |
| 12 | `waist_yaw` | `waist_yaw_joint` | 88 | [-2.62, 2.62] |
| 13 | `waist_roll` | `waist_roll_joint` | 50 | [-0.52, 0.52] |
| 14 | `waist_pitch` | `waist_pitch_joint` | 50 | [-0.52, 0.52] |
| 15 | `left_shoulder_pitch` | `left_shoulder_pitch_joint` | 25 | [-3.09, 2.67] |
| 16 | `left_shoulder_roll` | `left_shoulder_roll_joint` | 25 | [-1.59, 2.25] |
| 17 | `left_shoulder_yaw` | `left_shoulder_yaw_joint` | 25 | [-2.62, 2.62] |
| 18 | `left_elbow` | `left_elbow_joint` | 25 | [-1.05, 2.09] |
| 19 | `left_wrist_roll` | `left_wrist_roll_joint` | 25 | [-1.97, 1.97] |
| 20 | `left_wrist_pitch` | `left_wrist_pitch_joint` | 5 | [-1.61, 1.61] |
| 21 | `left_wrist_yaw` | `left_wrist_yaw_joint` | 5 | [-1.61, 1.61] |
| 22 | `right_shoulder_pitch` | `right_shoulder_pitch_joint` | 25 | [-3.09, 2.67] |
| 23 | `right_shoulder_roll` | `right_shoulder_roll_joint` | 25 | [-2.25, 1.59] |
| 24 | `right_shoulder_yaw` | `right_shoulder_yaw_joint` | 25 | [-2.62, 2.62] |
| 25 | `right_elbow` | `right_elbow_joint` | 25 | [-1.05, 2.09] |
| 26 | `right_wrist_roll` | `right_wrist_roll_joint` | 25 | [-1.97, 1.97] |
| 27 | `right_wrist_pitch` | `right_wrist_pitch_joint` | 5 | [-1.61, 1.61] |
| 28 | `right_wrist_yaw` | `right_wrist_yaw_joint` | 5 | [-1.61, 1.61] |

### 2.2.1 IsaacLab Joint Order (29-DOF)

Policies trained in IsaacLab for the G1-29 use the following joint ordering, which differs from the robot-native order above. The `JointMapper` must handle this reordering.

```python
ISAACLAB_G1_29DOF_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
```

**Key Observations:**
- IsaacLab interleaves left/right joints rather than grouping by limb
- Waist joints are interspersed with hip joints
- This ordering likely optimizes for the Isaac Gym parallel simulation structure

**Mapping Table (IsaacLab Index → Robot-Native Index):**
| IsaacLab Idx | Joint Name | Robot-Native Idx |
|--------------|------------|------------------|
| 0 | left_hip_pitch_joint | 0 |
| 1 | right_hip_pitch_joint | 6 |
| 2 | waist_yaw_joint | 12 |
| 3 | left_hip_roll_joint | 1 |
| 4 | right_hip_roll_joint | 7 |
| 5 | waist_roll_joint | 13 |
| 6 | left_hip_yaw_joint | 2 |
| 7 | right_hip_yaw_joint | 8 |
| 8 | waist_pitch_joint | 14 |
| 9 | left_knee_joint | 3 |
| 10 | right_knee_joint | 9 |
| 11 | left_shoulder_pitch_joint | 15 |
| 12 | right_shoulder_pitch_joint | 22 |
| 13 | left_ankle_pitch_joint | 4 |
| 14 | right_ankle_pitch_joint | 10 |
| 15 | left_shoulder_roll_joint | 16 |
| 16 | right_shoulder_roll_joint | 23 |
| 17 | left_ankle_roll_joint | 5 |
| 18 | right_ankle_roll_joint | 11 |
| 19 | left_shoulder_yaw_joint | 17 |
| 20 | right_shoulder_yaw_joint | 24 |
| 21 | left_elbow_joint | 18 |
| 22 | right_elbow_joint | 25 |
| 23 | left_wrist_roll_joint | 19 |
| 24 | right_wrist_roll_joint | 26 |
| 25 | left_wrist_pitch_joint | 20 |
| 26 | right_wrist_pitch_joint | 27 |
| 27 | left_wrist_yaw_joint | 21 |
| 28 | right_wrist_yaw_joint | 28 |

### 2.3 Joint Configuration (23-DOF)

The 23-DOF variant has a different physical structure (single torso joint, 5-DOF arms without wrist joints). Per the DDS IDL specification:

| Index | Config Name | Notes |
|-------|-------------|-------|
| 0-11 | Same as 29-DOF | Legs identical |
| 12 | `torso` | Single torso joint (replaces waist_yaw/roll/pitch) |
| 13-17 | `left_shoulder_pitch`, `left_shoulder_roll`, `left_shoulder_yaw`, `left_elbow_pitch`, `left_elbow_roll` | 5-DOF arm |
| 18-22 | `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`, `right_elbow_pitch`, `right_elbow_roll` | 5-DOF arm |

### 2.4 Default Home Position (Standing)

The default home position (`q_home`) for a stable standing pose. All values in radians. This configuration is derived from the BeyondMimic motion_tracking_controller standby position and provides a stable stance with slight knee bend.

**29-DOF Safe Standing Position:**
```python
Q_HOME_29DOF = {
    # Left leg - slight knee bend for stability
    "left_hip_pitch": -0.312,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.669,
    "left_ankle_pitch": -0.33,
    "left_ankle_roll": 0.0,
    # Right leg
    "right_hip_pitch": -0.312,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.669,
    "right_ankle_pitch": -0.33,
    "right_ankle_roll": 0.0,
    # Torso
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "waist_pitch": 0.0,
    # Left arm - slightly forward and outward
    "left_shoulder_pitch": 0.2,
    "left_shoulder_roll": 0.2,
    "left_shoulder_yaw": 0.0,
    "left_elbow": 0.6,
    "left_wrist_roll": 0.0,
    "left_wrist_pitch": 0.0,
    "left_wrist_yaw": 0.0,
    # Right arm - slightly forward and outward
    "right_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_yaw": 0.0,
    "right_elbow": 0.6,
    "right_wrist_roll": 0.0,
    "right_wrist_pitch": 0.0,
    "right_wrist_yaw": 0.0,
}
```

**Note**: The MuJoCo model places the pelvis at height 0.793m when all joints are at zero. The safe standing position above includes hip pitch (-0.312 rad), knee bend (0.669 rad), and compensating ankle pitch (-0.33 rad) for a stable, slightly crouched stance. Arms are positioned slightly forward with bent elbows to avoid collision with the body.

**23-DOF Safe Standing Position:**
```python
Q_HOME_23DOF = {
    # Left leg - identical to 29-DOF
    "left_hip_pitch": -0.312,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.669,
    "left_ankle_pitch": -0.33,
    "left_ankle_roll": 0.0,
    # Right leg
    "right_hip_pitch": -0.312,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.669,
    "right_ankle_pitch": -0.33,
    "right_ankle_roll": 0.0,
    # Torso - single joint (replaces 3 waist joints)
    "torso": 0.0,
    # Left arm - 5-DOF (no wrist joints)
    "left_shoulder_pitch": 0.2,
    "left_shoulder_roll": 0.2,
    "left_shoulder_yaw": 0.0,
    "left_elbow_pitch": 0.6,
    "left_elbow_roll": 0.0,
    # Right arm - 5-DOF (no wrist joints)
    "right_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": 0.6,
    "right_elbow_roll": 0.0,
}
```

**Note on 23-DOF vs 29-DOF Joint Differences:**
- **Torso**: 23-DOF has a single `torso` joint; 29-DOF has `waist_yaw`, `waist_roll`, `waist_pitch`
- **Elbow**: 23-DOF uses `elbow_pitch` and `elbow_roll`; 29-DOF uses a single `elbow` joint
- **Wrist**: 23-DOF has no wrist joints; 29-DOF has `wrist_roll`, `wrist_pitch`, `wrist_yaw`

### 2.5 IDL Mode
Use **Mode 0** (semantic joint names: ANKLE_PITCH/ROLL, WAIST_ROLL/PITCH) for policy interface. This provides a more intuitive coordinate system for RL policy development.

Mode 1 uses actuator-based names (ANKLE_A/B, WAIST_A/B) which map to the physical parallel mechanism actuators.

### 2.6 Communication Protocol
- **IDL Type**: `unitree_hg` (required for G1, distinct from `unitree_go` used by Go2/B2)
- **Topics**:
  - `rt/lowcmd` - Motor commands (subscribe by robot/sim)
  - `rt/lowstate` - Motor states (publish by robot/sim)
  - `rt/sportmodestate` - Robot pose/velocity (publish, **sim only** - not available on real robot with motion service disabled)
- **DDS Domain ID**:
  - Simulation: `1` (configurable)
  - Real robot: `0` (Unitree default)

### 2.7 Frame Conventions

**World Frame:**
- Z-axis: Up (vertical)
- X-axis: Forward (robot default facing direction)
- Y-axis: Left (right-hand rule)
- Origin: Ground level at simulation start

**Body Frame (pelvis/base):**
- Same orientation convention as world when robot is upright
- Origin: Center of pelvis
- IMU is located at the pelvis body

**IMU Data:**
- Quaternion format: **wxyz** (w, x, y, z) - MuJoCo convention
- Angular velocity: Body frame (rad/s)
- Linear acceleration: Body frame (m/s^2), includes gravity

**Coordinate Transform Notes:**
- To get base velocity in body frame (for observation), transform world-frame velocity using IMU quaternion
- Projected gravity = rotate [0, 0, -1] by inverse of IMU quaternion

---

## 3. Architecture

### 3.1 High-Level Component Diagram

**[Docker] Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Container                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Viser UI   │  │ Policy Runner│  │   Robot Interface    │  │
│  │  (Browser)   │◄─┤  (50 Hz)     │◄─┤                      │  │
│  │              │  │              │  │  ┌────────────────┐  │  │
│  │  - Controls  │  │  - ONNX Load │  │  │  SimRobot      │  │  │
│  │  - Viz       │  │  - Obs Build │  │  │  (MuJoCo+DDS)  │  │  │
│  │  - Telemetry │  │  - Action    │  │  ├────────────────┤  │  │
│  └──────────────┘  └──────────────┘  │  │  RealRobot     │  │  │
│                                       │  │  (DDS only)    │  │  │
│                                       │  └────────────────┘  │  │
│                                       └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │   Logger     │  │   Config     │                             │
│  │  (Compressed)│  │   Manager    │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
    Local Network                          Physical Robot
    (Browser access)                       (Ethernet DDS)
```

**[Metal] Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                   Native macOS / Linux Process                   │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  MuJoCo Viewer   │  │ Policy Runner│  │  Robot Interface  │  │
│  │  (Native Window) │  │  (50 Hz)     │◄─┤                   │  │
│  │                  │  │              │  │ ┌───────────────┐ │  │
│  │  - 3D Render     │  │  - ONNX Load │  │ │  SimRobot     │ │  │
│  │  - Keyboard WASD │  │  - Obs Build │  │ │  (MuJoCo+DDS) │ │  │
│  │  - Mouse Orbit   │  │  - Action    │  │ ├───────────────┤ │  │
│  │                  │  │              │  │ │  RealRobot    │ │  │
│  │  (or --headless  │  │              │  │ │  (DDS only)   │ │  │
│  │   for evals)     │  │              │  │ └───────────────┘ │  │
│  └──────────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │   Logger     │  │   Config     │                             │
│  │  (Compressed)│  │   Manager    │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
    stdout / logs                          Physical Robot
    (headless) or                          (Ethernet DDS)
    MuJoCo window (interactive)
```

### 3.2 Module Structure

**Shared modules** (both plans):

```
unitree_launcher/
├── configs/
│   ├── default.yaml           # Default configuration
│   ├── g1_29dof.yaml          # 29-DOF specific config
│   └── g1_23dof.yaml          # 23-DOF specific config
├── assets/
│   └── robots/
│       └── g1/
│           ├── g1_29dof.xml   # MuJoCo model (copied from reference)
│           ├── g1_23dof.xml   # MuJoCo model (copied from reference)
│           ├── scene_29dof.xml # Simulation scene (29-DOF)
│           ├── scene_23dof.xml # Simulation scene (23-DOF)
│           └── meshes/        # Robot mesh files (STL)
├── policies/                   # ONNX policy files
│   └── .gitkeep
├── logs/                       # Run logs
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── main.py                # Entry point (plan-specific, see below)
│   ├── config.py              # Configuration dataclasses
│   ├── compat.py              # Cross-platform shims (RecurrentThread, loopback detection)
│   ├── robot/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract RobotInterface, RobotState, RobotCommand
│   │   ├── sim_robot.py       # MuJoCo + DDS simulation
│   │   └── real_robot.py      # DDS to physical robot
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract PolicyInterface, detect_policy_format()
│   │   ├── isaaclab_policy.py # IsaacLab ONNX policy
│   │   ├── beyondmimic_policy.py # BeyondMimic ONNX policy
│   │   ├── joint_mapper.py    # Joint remapping utility
│   │   └── observations.py    # Observation builder (IsaacLab)
│   ├── control/
│   │   ├── __init__.py
│   │   ├── controller.py      # Main control loop + handle_key()
│   │   └── safety.py          # E-stop and safety logic
│   └── logging/
│       ├── __init__.py
│       ├── logger.py          # Data logger
│       └── replay.py          # Log replay utility
├── scripts/
│   ├── run_sim.sh             # Launch simulation
│   ├── run_real.sh            # Launch real robot
│   └── replay_log.py          # Standalone log replay
├── tests/
│   └── ...
├── pyproject.toml
└── README.md
```

**[Docker] additional modules:**

```
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   └── viz/                   # Viser visualization (Docker plan only)
│       ├── __init__.py
│       ├── server.py          # Viser server setup
│       ├── robot_viz.py       # Robot mesh/collision rendering
│       └── ui.py              # Control panel UI elements
├── requirements.txt           # Mirror of pyproject.toml for Docker
```

**[Metal] additional files:**

```
├── scripts/
│   ├── validate_macos.py      # Phase 0 validation
│   └── run_eval.sh            # Headless batch eval
├── .python-version            # Contains "3.10" for uv
├── tests/
│   └── test_compat.py         # Tests for RecurrentThread shim
```

**Note on `src/main.py`:** The entry point differs between plans. The shared components (config loading, policy creation, controller setup) are factored into helper functions. The plan-specific parts are the viewer launch (Metal: `mujoco.viewer.launch_passive`, Docker: Viser server start) and CLI arguments (`--headless`/`--duration`/`--steps` for Metal, `--viser-port`/`--no-viser` for Docker).

**Note on `src/compat.py`:** This module is shared between both plans. On Linux (Docker), the native `RecurrentThread` may be available, but the compat module's pure-Python implementation is used by default for consistency. The `resolve_network_interface("auto")` returns `lo0` on macOS and `lo` on Linux.

### 3.3 Design Decision: ROS vs Pure Python SDK

The engineer should evaluate both approaches and document the trade-offs:

**Pure Python SDK (unitree_sdk2_python) - Recommended starting point:**
- Pros: Simpler setup, fewer dependencies, easier Docker, no ROS complexity
- Cons: Less ecosystem tooling, manual message handling

**ROS2 (unitree_ros2):**
- Pros: Standard robotics tooling, rosbag logging, ecosystem integration
- Cons: Heavier container, more complex setup, rosbag requires ROS to read

**Recommendation**: Start with pure Python SDK for simplicity. ROS2 can be added later if ecosystem benefits outweigh complexity.

### 3.4 Threading Model

The system uses multiple threads to ensure the control loop cannot be blocked by other operations (critical for real robot safety).

**[Docker] Threading Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Main Process                                    │
│                                                                          │
│  ┌────────────────────┐    ┌────────────────────┐    ┌───────────────┐ │
│  │   Control Thread   │    │    Sim Thread      │    │  Viser Thread │ │
│  │   (50 Hz policy)   │    │   (200 Hz physics) │    │   (UI/render) │ │
│  │                    │    │                    │    │               │ │
│  │  - Read state      │◄──►│  - mj_step()       │    │  - Handle UI  │ │
│  │  - Build obs       │    │  - Publish state   │    │  - Update viz │ │
│  │  - Run policy      │    │  - Apply commands  │    │  - Callbacks  │ │
│  │  - Send command    │    │                    │    │               │ │
│  │  - Check E-stop    │    │  (Sim only, no-op  │    │               │ │
│  └─────────┬──────────┘    │   for real robot)  │    └───────┬───────┘ │
│            │               └────────────────────┘            │         │
│            │                                                  │         │
│            └──────────────────────┬───────────────────────────┘         │
│                                   │                                      │
│                    ┌──────────────▼──────────────┐                      │
│                    │     Shared State (locked)    │                      │
│                    │  - Robot state               │                      │
│                    │  - E-stop flag               │                      │
│                    │  - Velocity command          │                      │
│                    │  - System state enum         │                      │
│                    └─────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**[Metal] Threading Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Main Process                                    │
│                                                                          │
│  ┌────────────────────┐    ┌────────────────────┐    ┌───────────────┐ │
│  │   Control Thread   │    │    Sim Thread      │    │ MuJoCo Viewer │ │
│  │   (50 Hz policy)   │    │   (200 Hz physics) │    │ (Main Thread) │ │
│  │                    │    │                    │    │               │ │
│  │  - Read state      │◄──►│  - mj_step()       │    │  - GLFW loop  │ │
│  │  - Build obs       │    │  - Publish state   │    │  - key_callback│ │
│  │  - Run policy      │    │  - Apply commands  │    │  - 3D render  │ │
│  │  - Send command    │    │                    │    │  - Mouse orbit│ │
│  │  - Check E-stop    │    │  (Sim only, no-op  │    │               │ │
│  └─────────┬──────────┘    │   for real robot)  │    └───────┬───────┘ │
│            │               └────────────────────┘            │         │
│            │                                                  │         │
│            └──────────────────────┬───────────────────────────┘         │
│                                   │                                      │
│                    ┌──────────────▼──────────────┐                      │
│                    │     Shared State (locked)    │                      │
│                    │  - Robot state               │                      │
│                    │  - E-stop flag               │                      │
│                    │  - Velocity command          │                      │
│                    │  - System state enum         │                      │
│                    └─────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**[Metal] Headless Threading:** When `--headless` is used, the MuJoCo Viewer thread is replaced by a simple main-thread sleep loop that monitors for `--duration`/`--steps` termination conditions. No GLFW context is created.

**Thread Responsibilities:**

**[Docker]:**

| Thread | Frequency | Purpose | Blocking Allowed? |
|--------|-----------|---------|-------------------|
| Control | 50 Hz | Policy inference, command generation | **No** - must meet timing |
| Sim | 200 Hz | MuJoCo physics stepping | No (sim only) |
| Viser | Event-driven | UI updates, rendering | Yes |
| DDS (internal) | Varies | Pub/sub callbacks | Handled by SDK |

**[Metal]:**

| Thread | Frequency | Purpose | Blocking Allowed? |
|--------|-----------|---------|-------------------|
| Control | 50 Hz | Policy inference, command generation | **No** - must meet timing |
| Sim | 200 Hz | MuJoCo physics stepping | No (sim only) |
| MuJoCo Viewer | ~60 Hz (vsync) | GLFW render loop, keyboard input | Yes (main thread) |
| DDS (internal) | Varies | Pub/sub callbacks | Handled by SDK |

**[Metal] Viewer Thread Safety:** The MuJoCo viewer runs on the main thread (required by GLFW/macOS). `mujoco.viewer.launch_passive()` returns a handle. The `key_callback` fires on the main thread and must acquire the shared state lock before modifying velocity commands, E-stop flag, or system state. The control thread must hold the lock only briefly when reading shared state to avoid blocking the viewer.

**Critical Design Constraint**: The control thread must **never** be blocked by visualization, logging, or UI operations. On the real robot, a missed control cycle could cause instability or falls.

**Synchronization:**
- Use `threading.Lock` for shared state (both plans)
- E-stop flag should be atomic or lock-free for immediate response
- State reads in control thread should be non-blocking copies
- **[Metal]** `key_callback` acquires the lock briefly to update commands/flags

**E-Stop Handling:**
The E-stop flag is checked at the **start** of every control cycle. When set:
1. Control thread immediately switches to damping mode
2. No policy inference occurs
3. Damping commands are sent until E-stop is cleared

---

## 4. Policy Interface

The system supports two policy formats:
1. **IsaacLab** - Standard locomotion policies with velocity commands
2. **BeyondMimic** - Motion tracking policies with embedded reference trajectories

### 4.1 Supported Policy Formats

| Format | Source | Use Case | Config Key |
|--------|--------|----------|------------|
| IsaacLab | https://isaac-sim.github.io/IsaacLab/ | Velocity-commanded locomotion | `policy_format: isaaclab` |
| BeyondMimic | https://github.com/HybridRobotics/whole_body_tracking | Motion/pose tracking | `policy_format: beyondmimic` |

The policy format should be auto-detected from ONNX metadata when possible, with config override available.

### 4.2 IsaacLab Policy Format

Standard locomotion policies exported from IsaacLab.

**ONNX Interface:**
```
Inputs:
  - "obs": shape [1, obs_dim], float32

Outputs:
  - "action": shape [1, action_dim], float32
```

**Observation Space (IsaacLab):**

The observation vector follows IsaacLab's `PolicyCfg` ordering. Components are concatenated in the order shown below:

| # | Component | Dimension | Description |
|---|-----------|-----------|-------------|
| 1 | `base_lin_vel` | 3 | Base linear velocity in body frame (m/s). **Omitted when `use_estimator: false`.** |
| 2 | `base_ang_vel` | 3 | Base angular velocity in body frame (rad/s) |
| 3 | `projected_gravity` | 3 | Gravity vector projected into body frame (normalized) |
| 4 | `velocity_commands` | 3 | Desired [vx, vy, yaw_rate] |
| 5 | `joint_pos` | N_OBSERVED | Joint positions relative to home pose: `q - q_home` (rad) |
| 6 | `joint_vel` | N_OBSERVED | Joint velocities (rad/s) |
| 7 | `actions` | N_CONTROLLED | Previous policy output (last action) |

**No-estimator mode (`use_estimator: false`):** When no state estimator is available, the `base_lin_vel` term is **omitted entirely** (not zeroed). This reduces `obs_dim` by 3. Policies must be trained with or without this term accordingly. Set `use_estimator: false` in the policy config section, or override via `--no-est` CLI flag.

**Observation Dimension:**
```
# With state estimator (default):
obs_dim = 3 + 3 + 3 + 3 + N_OBSERVED + N_OBSERVED + N_CONTROLLED
        = 12 + 2*N_OBSERVED + N_CONTROLLED

# Without state estimator (use_estimator: false):
obs_dim = 3 + 3 + 3 + N_OBSERVED + N_OBSERVED + N_CONTROLLED
        = 9 + 2*N_OBSERVED + N_CONTROLLED
```

**Action Space (IsaacLab):**
| Component | Dimension | Description |
|-----------|-----------|-------------|
| Joint position offsets | N_CONTROLLED | Offset from home position (rad) |

**Control Law:**
```
τ = Kp * (q_home + Ka * action - q) - Kd * q̇
```
Where `Kp`, `Kd`, `Ka` come from the config file.

### 4.3 BeyondMimic Policy Format

Motion tracking policies from the HybridRobotics whole_body_tracking project.

**ONNX Interface:**
```
Inputs:
  - "obs": shape [1, obs_dim], float32
  - "time_step": shape [1], float32 (trajectory time index)

Outputs (tuple):
  - "action": shape [1, action_dim], float32 - RL policy output
  - "target_q": shape [1, n_joints], float32 - Reference joint positions
  - "target_dq": shape [1, n_joints], float32 - Reference joint velocities
  - "body_pos_w": shape [1, 3], float32 - Reference body position (world)
  - "body_quat_w": shape [1, 4], float32 - Reference body orientation (world)
  - "body_lin_vel_w": shape [1, 3], float32 - Reference linear velocity
  - "body_ang_vel_w": shape [1, 3], float32 - Reference angular velocity
```

**Embedded ONNX Metadata:**
The BeyondMimic exporter embeds configuration in ONNX metadata strings:
| Metadata Key | Description |
|--------------|-------------|
| `joint_names` | Ordered list of controlled joint names |
| `default_joint_pos` | Nominal joint positions (home pose) |
| `stiffness` | Per-joint position gains (Kp) |
| `damping` | Per-joint velocity gains (Kd) |
| `action_scale` | Per-joint action scaling (Ka) |
| `anchor_body` | Name of the anchor body for tracking |
| `controlled_bodies` | List of tracked body names |
| `obs_terms` | Observation term names |
| `obs_history_lengths` | History buffer size per observation term |

**Observation Space (BeyondMimic):**

Unlike IsaacLab (which uses raw joint states + velocity commands), BeyondMimic observations use body-relative states in 6D rotation representation. The observation structure is **metadata-driven**: the `obs_terms` field in the ONNX metadata specifies which components are included and in what order.

**Standard BeyondMimic observation components:**

| # | Component | Dim | Description | Source |
|---|-----------|-----|-------------|--------|
| 1 | `robot_anchor_ori_w` | 6 | Anchor body orientation in world frame (6D: first 2 columns of rotation matrix) | IMU quaternion → rotation matrix → columns 0,1 flattened |
| 2 | `robot_anchor_lin_vel_w` | 3 | Anchor body linear velocity in world frame (m/s) | IMU + velocity estimation (sim: `base_velocity`; real: integrated from IMU) |
| 3 | `robot_anchor_ang_vel_w` | 3 | Anchor body angular velocity in world frame (rad/s) | IMU angular velocity rotated to world frame |
| 4 | `robot_body_pos_b` | 3 × N_bodies | Body positions relative to anchor body (body-relative frame) | Forward kinematics from joint positions (MuJoCo `xpos` in sim) |
| 5 | `robot_body_ori_b` | 6 × N_bodies | Body orientations relative to anchor body (6D representation) | Forward kinematics from joint positions (MuJoCo `xquat` in sim) |
| 6 | `motion_anchor_pos_b` | 3 | Target/reference anchor position relative to robot anchor (body-relative) | Previous ONNX output `body_pos_w` transformed to body frame |
| 7 | `motion_anchor_ori_b` | 6 | Target/reference anchor orientation relative to robot anchor (6D) | Previous ONNX output `body_quat_w` transformed to body frame |

**Total observation dimension:**
```
obs_dim = 6 + 3 + 3 + (3 × N_bodies) + (6 × N_bodies) + 3 + 6
        = 21 + (9 × N_bodies)
```

Where `N_bodies` is the number of tracked bodies from the `controlled_bodies` metadata. For example, with 15 tracked bodies: `obs_dim = 21 + 135 = 156`.

**6D Rotation Representation:**
All rotations use the continuous 6D representation (first 2 columns of the 3×3 rotation matrix, flattened):
```python
def quat_to_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 6D rotation (first 2 columns of rotation matrix)."""
    R = quat_to_rotation_matrix(quat_wxyz)  # 3x3
    return R[:, :2].flatten()  # shape (6,)
```

**Body-Relative Coordinate Transform:**
All body positions and orientations are expressed relative to the anchor body:
```python
def world_to_body_relative(anchor_pos, anchor_quat, body_pos, body_quat):
    """Transform world-frame body state to anchor-relative body frame."""
    R_anchor_inv = quat_to_rotation_matrix(quat_inverse(anchor_quat))
    rel_pos = R_anchor_inv @ (body_pos - anchor_pos)
    rel_quat = quat_multiply(quat_inverse(anchor_quat), body_quat)
    return rel_pos, rel_quat
```

**Motion Target Construction:**
The reference/target information (components 6 and 7) comes from the **previous timestep's ONNX outputs** (`body_pos_w`, `body_quat_w`), transformed into the robot's current body frame. On the first timestep (no previous output), use the robot's own state as the target (zero relative offset).

**Observation History:**
The `obs_history_lengths` metadata specifies per-term history buffer sizes. If a term has `history_length > 1`, the observation includes stacked copies of that term from the last N timesteps:
- `obs_history_lengths = [1, 1, 1, ...]` → no history, single frame (most common)
- `obs_history_lengths = [3, 1, 1, ...]` → first term includes 3 frames stacked (triples its dimension)

When `history_length > 1`, the total `obs_dim` increases accordingly and must be computed dynamically after loading metadata.

**Key differences from IsaacLab:**
| Aspect | IsaacLab | BeyondMimic |
|--------|----------|-------------|
| Rotation format | Projected gravity (3D) | 6D rotation representation |
| Coordinate frame | Body frame for gravity/velocity | Body-relative for all bodies |
| User input | Velocity command [vx, vy, yaw_rate] | None (trajectory-driven) |
| State representation | Raw joint positions/velocities | Body positions/orientations via FK |
| Motion reference | None | Target from ONNX outputs |
| Observation structure | Fixed (determined by joint count) | Dynamic (determined by `obs_terms` metadata) |

**Implementation approach:**
The `BeyondMimicPolicy` class should build observations internally (not via the shared `ObservationBuilder`), because:
1. The observation structure depends on policy-specific metadata (`obs_terms`, `controlled_bodies`)
2. It requires forward kinematics to compute body positions/orientations
3. It requires tracking previous ONNX outputs for motion target construction
4. History buffering is policy-specific

The `BeyondMimicPolicy.build_observation()` method should:
1. Read `obs_terms` and `controlled_bodies` from ONNX metadata at load time
2. Use MuJoCo's `xpos`/`xquat` arrays for body state (sim) or forward kinematics (real)
3. Cache previous ONNX outputs for motion target construction
4. Stack history frames if `obs_history_lengths` requires it
5. Return the concatenated observation vector

**Reference implementation:** `whole_body_tracking/tasks/tracking/mdp/observations.py` contains the canonical observation construction. The `utils/exporter.py` shows how metadata is embedded during ONNX export.

**Control Law (BeyondMimic):**
```
τ = Kp * (target_q + Ka * action - q) - Kd * (q̇ - target_dq)
```
Where `Kp`, `Kd`, `Ka` come from ONNX metadata (`stiffness`, `damping`, `action_scale`), and `target_q`/`target_dq` come from ONNX outputs. Note the critical difference from IsaacLab: the damping term uses velocity **error** `(q̇ - target_dq)` rather than absolute velocity `q̇`.

**Time Step Management:**
- The policy requires a `time_step` input indexing into the embedded motion trajectory
- Time step is automatically incremented based on real time (or sim time) at 50 Hz
- User interaction is limited to start/stop (no velocity commands like IsaacLab)
- At trajectory end: smoothly interpolate back to `q_home` (safe standing position) over ~2 seconds

**End-of-Trajectory Behavior:**
When the trajectory completes:
1. Detect end of trajectory (time_step exceeds embedded trajectory length, or ONNX outputs become constant/NaN)
2. Capture final joint positions from the last valid `target_q` output
3. Linearly interpolate from final positions to `q_home` over 2 seconds (100 steps at 50 Hz)
4. During interpolation, apply: `τ = Kp * (q_interp - q) - Kd * q̇` (standard PD, no action offset)
5. Once at `q_home`, enter STOPPED state and await user input
6. Future enhancement: transition to a standing balance policy instead of static `q_home`

### 4.4 Loading Example

```python
import onnxruntime as ort
import onnx

def detect_policy_format(onnx_path: str) -> str:
    """Auto-detect policy format from ONNX structure."""
    session = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]

    if "time_step" in input_names:
        return "beyondmimic"
    return "isaaclab"

def load_beyondmimic_metadata(onnx_path: str) -> dict:
    """Extract embedded metadata from BeyondMimic policy."""
    model = onnx.load(onnx_path)
    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value
    return metadata

# Usage
format = detect_policy_format("policy.onnx")
if format == "beyondmimic":
    metadata = load_beyondmimic_metadata("policy.onnx")
    joint_names = eval(metadata["joint_names"])  # Parse string list
    stiffness = eval(metadata["stiffness"])
    # ... etc
```

### 4.5 Policy Configuration

```yaml
policy:
  # Policy format: "isaaclab" or "beyondmimic" (auto-detected if omitted)
  format: null  # Auto-detect from ONNX structure

  # For IsaacLab: joint configuration
  observed_joints: null   # Joints in observation (defaults to controlled)
  controlled_joints: null # Joints receiving commands (defaults to all)

  # For BeyondMimic: these are typically read from ONNX metadata
  # but can be overridden if needed
  use_onnx_metadata: true  # Use embedded joint names, gains, etc.
```

### 4.6 Joint Remapping, Observation, and Subset Control

Policies trained in IsaacLab may:
- Use a different joint ordering than the robot's native order
- Observe a different set of joints than they control (e.g., observe full body, control legs only)
- Control only a subset of joints

**Key Distinction: Observed vs Controlled Joints**

| Concept | Description | Affects |
|---------|-------------|---------|
| **Observed joints** | Joints included in policy observation | Observation vector size |
| **Controlled joints** | Joints receiving policy commands | Action vector size |

These can be configured independently. Common patterns:
- **Full observe, full control**: Standard full-body policy
- **Full observe, partial control**: Policy sees full state but only commands legs (for balance awareness)
- **Partial observe, partial control**: Isolated subsystem (e.g., arm-only policy that doesn't need leg info)

**Joint Remapping:**
Both observed and controlled joints support arbitrary reordering between the policy's expected order and the robot's native order.

**Subset Control Behavior:**
When controlling only a subset of joints:
- **Controlled joints**: Receive commands from the policy via the PD control law
- **Non-controlled joints**: Automatically placed in **damping mode** (same as E-stop behavior: `τ = -Kd_damp * q̇`)

**Configuration Example:**
```yaml
policy:
  # Joints included in observation, in policy-expected ORDER
  # If omitted or null, defaults to controlled_joints
  # If both omitted, all joints in robot-native order
  observed_joints:
    - L_LEG_HIP_PITCH
    - L_LEG_HIP_ROLL
    # ... (all 29 joints in policy observation order)

  # Joints controlled by policy, in policy-expected ORDER
  # If omitted or null, all joints are controlled in robot-native order
  controlled_joints:
    - L_LEG_HIP_PITCH
    - L_LEG_HIP_ROLL
    # ... (only leg joints - 12 total)
```

In this example:
- Observation includes all 29 joints (full body awareness)
- Policy outputs 12 actions (legs only)
- Arms and torso are in damping mode during policy execution

**Observation Space Adjustment:**
The observation builder must:
1. Compute relative joint positions (`q - q_home`) for **observed** joints in policy order
2. Extract joint velocities for **observed** joints in policy order
3. `actions` (last_action) dimension matches **controlled** joints
4. Prepend IMU-derived terms (`base_lin_vel`, `base_ang_vel`, `projected_gravity`) and `velocity_commands` before joint data
5. When `use_estimator: false` (or `--no-est` CLI override), omit `base_lin_vel` entirely

### 4.7 Control Law

**For Controlled Joints:**
The PD torque control law:

```
τ = Kp * (q_home + Ka * action - q) - Kd * q̇
```

Where:
- `τ`: Applied joint torque
- `Kp`: Position gain (per-joint, from config)
- `Kd`: Velocity gain (per-joint, from config)
- `Ka`: Action scale (per-joint, from config)
- `q_home`: Home/nominal joint positions
- `action`: Policy output (joint position offsets)
- `q`: Current joint position
- `q̇`: Current joint velocity

**For Non-Controlled Joints (Damping Mode):**
```
τ = -Kd_damp * q̇
```

This gently damps any motion without applying position-based torques.

### 4.8 Timing

| Parameter | Value | Configurable |
|-----------|-------|--------------|
| Policy frequency | 50 Hz (20 ms) | Yes |
| Simulation frequency | 200 Hz (5 ms) | Yes |
| Sim steps per policy step | 4 | Derived |

### 4.9 Policy Class Interface

```python
class PolicyInterface(ABC):
    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy from file."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset policy state (for recurrent policies)."""
        pass

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Compute action from observation."""
        pass

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Expected observation dimension."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Output action dimension."""
        pass
```

### 4.10 Joint Mapping Utility

A utility class handles the mapping between robot-native joint order and policy-expected order for both observations and actions:

```python
class JointMapper:
    def __init__(self,
                 robot_joints: List[str],
                 observed_joints: Optional[List[str]] = None,
                 controlled_joints: Optional[List[str]] = None):
        """
        Args:
            robot_joints: All joint names in robot-native order (from robot config)
            observed_joints: Joints included in observation, in policy-expected order.
                            If None, defaults to controlled_joints.
            controlled_joints: Joints to control, in policy-expected order.
                              If None, use all joints in robot-native order.

        Resolution order:
            - If both None: observe and control all joints in robot-native order
            - If only controlled_joints specified: observed_joints = controlled_joints
            - If only observed_joints specified: controlled_joints = all joints
            - If both specified: use as given (observed and controlled can differ)

        Raises:
            ValueError: If any joint name is not in robot_joints
        """
        pass

    # --- Observation mapping ---

    @property
    def observed_indices(self) -> np.ndarray:
        """Indices into robot state array for observed joints, in policy order."""
        pass

    @property
    def n_observed(self) -> int:
        """Number of observed joints."""
        pass

    def robot_to_observation(self, robot_values: np.ndarray) -> np.ndarray:
        """
        Extract and reorder observed joint values from full robot array.

        Args:
            robot_values: Array of shape (n_total,) in robot-native order

        Returns:
            Array of shape (n_observed,) in policy observation order
        """
        pass

    # --- Action/control mapping ---

    @property
    def controlled_indices(self) -> np.ndarray:
        """Indices into robot state array for controlled joints, in policy order."""
        pass

    @property
    def non_controlled_indices(self) -> np.ndarray:
        """Indices into robot state array for non-controlled joints."""
        pass

    @property
    def n_controlled(self) -> int:
        """Number of controlled joints."""
        pass

    def robot_to_action(self, robot_values: np.ndarray) -> np.ndarray:
        """
        Extract and reorder controlled joint values from full robot array.
        Used for extracting last_action or current positions for controlled joints.

        Args:
            robot_values: Array of shape (n_total,) in robot-native order

        Returns:
            Array of shape (n_controlled,) in policy action order
        """
        pass

    def action_to_robot(self, policy_action: np.ndarray,
                        default_value: float = 0.0) -> np.ndarray:
        """
        Map policy action to full robot command array.

        Args:
            policy_action: Array of shape (n_controlled,) in policy order
            default_value: Value for non-controlled joints

        Returns:
            Array of shape (n_total,) in robot-native order
        """
        pass

    # --- General ---

    @property
    def n_total(self) -> int:
        """Total number of robot joints."""
        pass
```

### 4.11 Observation Builder

The observation builder constructs the policy input from robot state, following the IsaacLab `PolicyCfg` observation ordering. It handles the distinction between observed and controlled joints, relative joint positions, and optional `base_lin_vel` omission.

```python
class ObservationBuilder:
    def __init__(self, joint_mapper: JointMapper, config: Config,
                 use_estimator: bool = True):
        """
        Args:
            joint_mapper: Handles joint ordering/subsetting for both obs and action
            config: Contains q_home, normalization params, etc.
            use_estimator: If False, omit base_lin_vel from observation.
                          Reads from config.policy.use_estimator; can be overridden by --no-est CLI flag.
        """
        self._use_estimator = use_estimator
        pass

    @property
    def observation_dim(self) -> int:
        """
        Total observation dimension.
        With estimator:    3 + 3 + 3 + 3 + n_observed + n_observed + n_controlled
        Without estimator:     3 + 3 + 3 + n_observed + n_observed + n_controlled
        """
        base = 2 * self._joint_mapper.n_observed + self._joint_mapper.n_controlled
        return base + (12 if self._use_estimator else 9)

    def build(self,
              robot_state: RobotState,
              last_action: np.ndarray,
              velocity_command: np.ndarray) -> np.ndarray:
        """
        Build observation vector in IsaacLab PolicyCfg order.

        Args:
            robot_state: Current state from robot interface
            last_action: Previous policy output, shape (n_controlled,)
            velocity_command: User velocity command [vx, vy, yaw_rate]

        Returns:
            Observation vector with components in order:
            [base_lin_vel?, base_ang_vel, projected_gravity, velocity_commands,
             joint_pos, joint_vel, actions]

            - base_lin_vel: shape (3,) - ONLY if use_estimator=True
            - base_ang_vel: shape (3,) - base angular velocity in body frame
            - projected_gravity: shape (3,) - gravity vector in body frame
            - velocity_commands: shape (3,) - velocity command [vx, vy, yaw_rate]
            - joint_pos: shape (n_observed,) - positions relative to home: q - q_home
            - joint_vel: shape (n_observed,) - velocities for OBSERVED joints
            - actions: shape (n_controlled,) - previous action for CONTROLLED joints
        """
        pass

    def compute_projected_gravity(self, quaternion: np.ndarray) -> np.ndarray:
        """Compute gravity vector in body frame from IMU quaternion."""
        pass
```

**Example: Full Observe, Legs Control**
```python
# Policy observes all 29 joints but only controls 12 leg joints
joint_mapper = JointMapper(
    robot_joints=G1_29DOF_JOINTS,        # 29 joints
    observed_joints=ALL_JOINTS_ISAACLAB_ORDER,  # 29 joints in IsaacLab order
    controlled_joints=LEG_JOINTS_ISAACLAB_ORDER  # 12 leg joints
)

# With state estimator (default: use_estimator: true in config):
obs_builder = ObservationBuilder(joint_mapper, config, use_estimator=True)
# obs_builder.observation_dim = 12 + 29 + 29 + 12 = 82

# Without state estimator (use_estimator: false in config, or --no-est CLI):
obs_builder = ObservationBuilder(joint_mapper, config, use_estimator=False)
# obs_builder.observation_dim = 9 + 29 + 29 + 12 = 79
```

---

## 5. Robot Interface

### 5.1 Abstract Interface

```python
class RobotInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to robot (sim or real)."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Clean shutdown."""
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """Get current robot state."""
        pass

    @abstractmethod
    def send_command(self, cmd: RobotCommand) -> None:
        """Send motor commands."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Advance simulation (no-op for real robot)."""
        pass

    @abstractmethod
    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Reset to initial state (sim only, no-op for real)."""
        pass
```

### 5.2 State and Command Dataclasses

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
    base_position: np.ndarray        # (3,) world frame (sim only)
    base_velocity: np.ndarray        # (3,) world frame (sim only)

@dataclass
class RobotCommand:
    joint_positions: np.ndarray   # (N_DOF,) target positions
    joint_velocities: np.ndarray  # (N_DOF,) target velocities (typically 0)
    joint_torques: np.ndarray     # (N_DOF,) feedforward torques
    kp: np.ndarray                # (N_DOF,) position gains
    kd: np.ndarray                # (N_DOF,) velocity gains
```

### 5.3 SimRobot Implementation

- Uses MuJoCo for physics simulation (`mj_step` for stepping; viewer is separate)
  - **[Docker]** No MuJoCo viewer — visualization via Viser
  - **[Metal]** `mujoco.viewer.launch_passive()` for interactive mode; no viewer for `--headless`
- Uses `unitree_sdk2_python` DDS bridge to publish state and receive commands
- Runs at configurable simulation frequency (default 200 Hz)
- Supports reset to initial configuration

### 5.4 RealRobot Implementation

- Uses `unitree_sdk2_python` DDS to communicate with physical robot
- Network interface specified via CLI argument
- `step()` is a no-op (real-time execution)
- `reset()` logs a warning (cannot reset physical robot)

**Startup Requirements (Real Robot):**
The robot must be in a safe initial state before the policy can be started:

1. **Expected Initial State**: Robot should be either:
   - Standing on the ground in approximately upright position
   - Hanging from a gantry/support harness (for initial testing)

2. **Orientation Check**: Before allowing policy start, verify:
   - IMU quaternion indicates approximately upright orientation
   - Projected gravity vector is close to [0, 0, -1] in body frame
   - Tolerance: gravity Z component > 0.8 (roughly < 35 degrees from vertical)

3. **Startup Rejection**: If orientation check fails:
   - Print clear error message: `"ERROR: Robot orientation unsafe for policy start. Projected gravity: [x, y, z]. Robot must be approximately upright."`
   - Refuse to transition from IDLE to RUNNING state
   - User must physically reposition robot and retry

4. **Console Output**: On startup attempt, print:
   - Current IMU orientation (quaternion and euler angles)
   - Projected gravity vector
   - Pass/fail status of orientation check
   - Clear instructions if check fails

**Threading Requirements (Real Robot):**
In real robot mode, the control thread and any simulation/visualization threads must be strictly separated:
- The control thread handles policy inference and DDS command publishing
- Visualization updates must not block or delay command publishing
- Use thread-safe queues or atomic operations for state sharing between threads
- Command latency from policy output to DDS publish should be monitored and logged

---

## 6. Visualization

This section is plan-specific. Both plans share the same keyboard shortcuts and velocity command ranges, but the UI implementation differs.

### 6.1 [Docker] Viser Server Configuration

- **Host**: `0.0.0.0` (accessible on local network)
- **Port**: Default Viser port (8080, configurable)
- **Access**: Any browser on local network via `http://<host-ip>:<port>`

### 6.1M [Metal] MuJoCo Viewer Configuration

- **Viewer**: `mujoco.viewer.launch_passive(model, data, key_callback=handle_key)`
- **Window**: Native GLFW window on macOS/Linux
- **Rendering**: OpenGL (macOS native), no EGL needed
- **Mouse**: Built-in MuJoCo orbit/pan/zoom controls
- **Headless mode** (`--headless`): No viewer is launched. Status is printed to stdout at 1 Hz:
  ```
  [00:05.2] step=260 state=RUNNING vel=[0.3, 0.0, 0.0] hz=50.1
  ```

### 6.2 Scene Elements (Shared)

**Robot Visualization:**
- Visual meshes (default): Full robot appearance
- Collision geometry: Simplified shapes for debugging
- Toggle between mesh/collision via keyboard (`M` key) or UI button **[Docker]** / keyboard only **[Metal]**

**Camera Views:**
- Follow robot (default): Camera tracks robot base
- Fixed world: Stationary camera
- Orbit: User-controlled orbit around robot
- Switch via keyboard (`1`, `2`, `3`) or dropdown menu **[Docker]** / keyboard only **[Metal]**

**Coordinate Frames:**
- World frame indicator
- Robot base frame
- Optional: Joint frames (toggle-able)

### 6.3 [Docker] Control Panel UI

```
┌─────────────────────────────────────────┐
│           UNITREE G1 CONTROL            │
├─────────────────────────────────────────┤
│  Status: [RUNNING / STOPPED / E-STOP]   │
│                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │  START  │  │  STOP   │  │  RESET  │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│                                         │
│  ┌─────────────────────────────────────┐│
│  │         E-STOP                      ││
│  └─────────────────────────────────────┘│
│                                         │
│  Policy: [Dropdown: policy1.onnx ▼]     │
│                                         │
│  ┌─────────────────────────────────────┐│
│  │ Velocity Command                    ││
│  │  Vx: [====|====] 0.0 m/s           ││
│  │  Vy: [====|====] 0.0 m/s           ││
│  │ Yaw: [====|====] 0.0 rad/s         ││
│  └─────────────────────────────────────┘│
│                                         │
│  View: [Dropdown: Follow Robot ▼]       │
│  Render: [○ Mesh  ● Collision]          │
├─────────────────────────────────────────┤
│  Telemetry                              │
│  Base Height: 0.85 m                    │
│  Base Vel: [0.1, 0.0, 0.0] m/s         │
│  Policy Hz: 50.2                        │
│  Sim Hz: 201.3                          │
└─────────────────────────────────────────┘
```

### 6.3M [Metal] Viewer Controls

The Metal plan uses MuJoCo's native viewer window. All interaction is via keyboard (no GUI panels). The MuJoCo viewer window title bar shows the current system state.

**On-screen feedback**: The MuJoCo viewer supports text overlays via `mjv_defaultOption`. If feasible, display a minimal HUD with:
- System state (IDLE/RUNNING/E-STOP)
- Current velocity command [vx, vy, yaw]
- Policy Hz

If text overlay is not feasible with `launch_passive`, fall back to stdout-only feedback at 1 Hz (same as headless format).

### 6.4 Keyboard Shortcuts (Shared)

Both plans use the same keyboard shortcuts. **[Docker]** uses Viser keyboard events; **[Metal]** uses GLFW `key_callback` with press-only detection (no repeat).

| Key | GLFW Code | Action |
|-----|-----------|--------|
| `Space` | 32 | Toggle start/stop |
| `R` | 82 | Reset simulation |
| `E` | 69 | E-stop (latching) |
| `C` | 67 | Clear E-stop |
| `M` | 77 | Toggle mesh/collision |
| `1` | 49 | Follow camera |
| `2` | 50 | Fixed camera |
| `3` | 51 | Orbit camera |
| `W` | 87 | Increase Vx by +0.1 (clamp to 1.0) |
| `S` | 83 | Decrease Vx by -0.1 (clamp to -1.0) |
| `A` | 65 | Increase Vy by +0.1 (clamp to 0.5) |
| `D` | 68 | Decrease Vy by -0.1 (clamp to -0.5) |
| `Q` | 81 | Increase yaw by +0.1 (clamp to 1.0) |
| `Z` | 90 | Decrease yaw by -0.1 (clamp to -1.0) |
| `X` | 88 | Zero all velocity commands |

**[Metal] GLFW Key Callback Signature:**
```python
def handle_key(keycode: int) -> None:
    """Called by mujoco.viewer on key press (GLFW keycode, press only)."""
```

**[Docker] Implementation Note:**
The engineer should investigate whether Viser supports global keyboard shortcuts or only when the 3D canvas is focused. If global shortcuts are not natively supported, custom keyboard event handling may need to be implemented. Document findings and any workarounds in the code.

### 6.5 Velocity Command Ranges (Shared)

| Command | Range | Default | Step Size |
|---------|-------|---------|-----------|
| Vx (forward/back) | [-1.0, 1.0] m/s | 0.0 | 0.1 |
| Vy (left/right) | [-0.5, 0.5] m/s | 0.0 | 0.1 |
| Yaw rate | [-1.0, 1.0] rad/s | 0.0 | 0.1 |

**Note**: Start with small commands (< 0.3 m/s) when first testing on real robot.

---

## 7. Safety System

### 7.1 E-Stop Behavior

**Trigger Conditions:**
- User presses E-stop button in UI
- User presses `E` key
- (Future: watchdog timeout, joint limit violation, etc.)

**E-Stop State (Latching):**
- System enters DAMPING mode
- Remains in E-stop until explicitly cleared
- Clear via UI button or `C` key

### 7.2 Damping Mode

When E-stop is active:

```
τ = -Kd_damp * q̇
```

Where:
- `Kd_damp`: Damping gain for E-stop (configurable, should be gentle)
- Target position = current position (no position error contribution)
- Only velocity damping is applied to gently slow the robot

### 7.3 State Machine

```
        ┌─────────┐
        │  IDLE   │◄──────────────────┐
        └────┬────┘                   │
             │ START                  │
             ▼                        │
        ┌─────────┐    E-STOP    ┌────┴────┐
        │ RUNNING │─────────────►│ E-STOP  │
        └────┬────┘              └────┬────┘
             │ STOP                   │ CLEAR
             ▼                        │
        ┌─────────┐                   │
        │ STOPPED │───────────────────┘
        └─────────┘
```

### 7.4 Applicability

E-stop functionality applies to **both simulation and real robot**:
- Simulation: Applies damping torques in MuJoCo
- Real robot: Sends damping commands via DDS

### 7.5 Error Handling

The system should handle errors gracefully without crashing or leaving the robot in an unsafe state.

| Error Type | Handling | User Feedback |
|------------|----------|---------------|
| Invalid ONNX file | Refuse to start, show error | "Failed to load policy: [reason]" |
| ONNX dimension mismatch | Refuse to start | "Policy expects obs_dim=X, config specifies Y" |
| Invalid joint name in config | Refuse to start | "Unknown joint: [name]" |
| DDS connection failure (sim) | Retry with backoff, then fail | "Cannot connect to simulator DDS" |
| DDS connection failure (real) | Refuse to start | "Cannot connect to robot on [interface]" |
| Robot state timeout (real) | Trigger E-stop | "Lost communication with robot" |
| Policy inference error | Trigger E-stop | "Policy inference failed: [reason]" |
| Viser connection lost **[Docker]** | Continue running (headless) | Log warning |
| Viewer window closed **[Metal]** | Graceful shutdown (stop policy, save logs) | N/A |

**Critical Principle**: When in doubt, trigger E-stop. It's always safer to stop the robot than to continue with potentially bad commands.

---

## 8. Logging System

### 8.1 Log Format

Logs are stored in a compressed binary format for efficiency.

**Supported Formats**: Both HDF5 (`.hdf5`) and compressed NumPy archives (`.npz`) must be supported, selectable via config (`logging.format: hdf5` or `logging.format: npz`). HDF5 is the default — it supports compression, partial reads, and embedded metadata. NPZ is a simpler alternative with no `h5py` dependency.

**Logged Data (per timestep):**
| Field | Type | Description |
|-------|------|-------------|
| timestamp | float64 | Monotonic time (seconds) |
| robot_state | RobotState | Full robot state |
| observation | float32[] | Policy input |
| action | float32[] | Policy output |
| command | RobotCommand | Sent motor command |
| system_state | enum | IDLE/RUNNING/STOPPED/E-STOP |
| velocity_command | float32[3] | User velocity command |
| policy_inference_time_ms | float32 | Time to run ONNX inference |
| control_loop_time_ms | float32 | Total control loop iteration time |
| sim_step_time_ms | float32 | MuJoCo step time (sim only) |

**Timing Statistics:**
The system should track and log timing statistics for performance monitoring:
- Policy inference time (ONNX runtime)
- Control loop total time (state read -> command send)
- Simulation step time (MuJoCo physics, sim only)
- DDS publish/subscribe latency

These statistics should be:
- Logged per-timestep in the data file
- **[Docker]** Displayed in the Viser telemetry panel (rolling average)
- **[Metal]** Displayed in stdout status lines at 1 Hz (headless) or viewer HUD (interactive)
- Printed as summary statistics at session end

### 8.2 Log File Structure

```
logs/
├── 2024-01-15_14-30-22_sim_policy1/
│   ├── metadata.yaml      # Run configuration
│   ├── data.hdf5          # Compressed time-series data
│   └── events.json        # Discrete events (start, stop, e-stop, etc.)
```

### 8.3 Replay Utility

A standalone Python script that can replay logs **outside the Docker container**:

```bash
python scripts/replay_log.py logs/2024-01-15_14-30-22_sim_policy1/
```

Features:
- Load and parse log files
- Print summary statistics
- Export to CSV for external analysis
- (Optional) Launch Viser visualization of recorded trajectory

**Important**: Replay must work on macOS without Docker (no ROS dependency).

---

## 9. Docker Configuration [Docker]

*This entire section applies only to the Docker plan. The Metal plan uses `uv` for environment management on bare metal (see Section 9M).*

### 9.1 Base Image

```dockerfile
FROM ubuntu:20.04
```

### 9.2 Key Dependencies

| Package | Purpose |
|---------|---------|
| Python 3.8+ | Runtime (use system Python or deadsnakes PPA) |
| mujoco | Physics simulation |
| viser | Visualization server |
| onnxruntime | Policy inference |
| unitree_sdk2_python | Robot communication |
| cyclonedds | DDS middleware (dependency of unitree_sdk2_python) |
| numpy | Numerical operations |
| pyyaml | Configuration |
| h5py | Log storage |

### 9.3 Rendering Backend

Use **EGL** for headless MuJoCo rendering:

```dockerfile
ENV MUJOCO_GL=egl
RUN apt-get install -y libegl1-mesa-dev libgl1-mesa-dev
```

This should work on both Linux (native) and macOS (via Docker's Linux VM).

### 9.4 Volume Mounts

```yaml
volumes:
  - ./src:/app/src           # Code (live reload)
  - ./configs:/app/configs   # Configuration files
  - ./assets:/app/assets     # Robot models and meshes
  - ./policies:/app/policies # ONNX policy files
  - ./logs:/app/logs         # Output logs
```

### 9.5 Network Configuration

```yaml
network_mode: host  # Required for DDS multicast (sim)
# OR for real robot:
# network_mode: bridge with specific port mappings
```

**Note**: DDS on the real robot requires the container to access the host network interface connected to the robot.

**macOS Docker Networking (To Be Documented in README):**
Docker networking behaves differently on macOS than Linux because Docker runs in a Linux VM:
- `network_mode: host` does not provide direct host network access on macOS
- DDS multicast may require additional configuration
- The engineer should investigate and document the specific configuration needed for:
  - Simulation mode (DDS between container processes)
  - Viser web UI access from host browser
- This should be documented in the project README with platform-specific instructions

### 9.6 Ports

| Port | Service |
|------|---------|
| 8080 | Viser web UI (configurable) |

### 9M. Environment Setup [Metal]

*This section applies only to the Metal plan. The Docker plan uses Docker containers (see Section 9).*

#### 9M.1 Python Version

Python 3.10 is required for CycloneDDS macOS ARM64 wheel compatibility (`cyclonedds==0.10.2` provides wheels for Python 3.8-3.10).

```
# .python-version (read by uv)
3.10
```

#### 9M.2 Environment Manager

Use **uv** (`pip install uv` or `brew install uv`) for deterministic environment setup:

```bash
uv venv              # Creates .venv with Python 3.10
uv pip install -e ".[dev]"   # Install in editable mode
```

#### 9M.3 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco | >=3.0 | Physics simulation + native viewer |
| onnxruntime | >=1.16 | Policy inference (CPU) |
| cyclonedds | ==0.10.2 | DDS middleware (macOS ARM64 wheels) |
| unitree_sdk2_python | >=0.1 | Robot communication (installed from git) |
| numpy | >=1.24 | Numerical operations |
| pyyaml | >=6.0 | Configuration |
| h5py | >=3.9 | Log storage |

**Dev dependencies**: `pytest`, `pytest-timeout`

#### 9M.4 No Docker, No EGL

The Metal plan does **not** use Docker or EGL. MuJoCo rendering uses the native OpenGL backend on macOS. On Linux headless servers, `MUJOCO_GL=egl` can be set for `--headless` mode if needed, but is not required (headless mode does not render).

#### 9M.5 Network Configuration

DDS loopback configuration differs by platform. The `resolve_network_interface("auto")` function in `src/compat.py` handles this:

| Platform | Loopback Interface | Environment Variable |
|----------|--------------------|---------------------|
| macOS | `lo0` | `CYCLONEDDS_URI` with `<NetworkInterface name="lo0"/>` |
| Linux | `lo` | `CYCLONEDDS_URI` with `<NetworkInterface name="lo"/>` |

---

## 10. Command Line Interface

### 10.1 Simulation Mode

**[Docker]:**
```bash
# Inside container
python -m src.main sim \
    --config configs/default.yaml \
    --policy policies/walking_policy.onnx \
    --robot g1_29dof \
    --viser-port 8080

# Via convenience script
./scripts/run_sim.sh --policy policies/walking_policy.onnx
```

**[Metal]:**
```bash
# Interactive (MuJoCo viewer window)
python -m src.main sim \
    --config configs/default.yaml \
    --policy policies/walking_policy.onnx \
    --robot g1_29dof

# Headless (no viewer, stdout status)
python -m src.main sim \
    --policy policies/walking_policy.onnx \
    --headless --duration 30

# Via convenience script
./scripts/run_sim.sh --policy policies/walking_policy.onnx
```

### 10.2 Real Robot Mode

**[Docker]:**
```bash
# Inside container
python -m src.main real \
    --config configs/default.yaml \
    --policy policies/walking_policy.onnx \
    --robot g1_29dof \
    --interface eth0 \
    --viser-port 8080

# Via convenience script
./scripts/run_real.sh --interface eth0 --policy policies/walking_policy.onnx
```

**[Metal]:**
```bash
# Interactive (MuJoCo viewer window)
python -m src.main real \
    --config configs/default.yaml \
    --policy policies/walking_policy.onnx \
    --robot g1_29dof \
    --interface en0

# Via convenience script
./scripts/run_real.sh --interface en0 --policy policies/walking_policy.onnx
```

### 10.3 Log Replay (No Docker Required)

```bash
# On host machine (macOS or Linux)
python scripts/replay_log.py logs/<run_dir>/ --format csv --output analysis.csv
python scripts/replay_log.py logs/<run_dir>/ --visualize  # Opens Viser replay [Docker] or MuJoCo replay [Metal]
```

### 10.4 Headless Batch Evaluation [Metal]

```bash
# Run for 30 seconds
python -m src.main sim --policy policies/walk.onnx --headless --duration 30

# Run for 1000 policy steps
python -m src.main sim --policy policies/walk.onnx --headless --steps 1000

# Via convenience script
./scripts/run_eval.sh --policy policies/walk.onnx --duration 60
```

**Auto-termination**: Headless mode exits when any of these conditions is met:
- `--duration` seconds elapsed (wall clock)
- `--steps` policy steps completed
- BeyondMimic trajectory ends (auto-detected)
- Ctrl+C received

### 10.5 CLI Arguments

**Shared arguments (both plans):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | path | `configs/default.yaml` | Configuration file |
| `--policy` | path | required | ONNX policy file |
| `--robot` | str | `g1_29dof` | Robot variant (`g1_29dof`, `g1_23dof`) |
| `--interface` | str | `auto` (sim) | Network interface (real robot: `eth0`/`en0`) |
| `--domain-id` | int | 1 (sim), 0 (real) | DDS domain ID |
| `--log-dir` | path | `logs/` | Log output directory |
| `--no-est` | flag | false | Override `policy.use_estimator` to false (omit `base_lin_vel` from observations) |

**[Docker] additional arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--viser-port` | int | 8080 | Viser server port |
| `--no-viser` | flag | false | Run without Viser visualization |

**[Metal] additional arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--headless` | flag | false | Run without MuJoCo viewer window |
| `--duration` | float | None | Auto-stop after N seconds (headless only) |
| `--steps` | int | None | Auto-stop after N policy steps (headless only) |
| `--policy-dir` | path | `policies/` | Directory for policy discovery (future use) |

---

## 11. Configuration Schema

### 11.1 Main Configuration (`default.yaml`)

```yaml
robot:
  variant: g1_29dof          # g1_29dof or g1_23dof
  idl_mode: 0                # 0 = semantic names, 1 = actuator names

policy:
  # Whether the policy expects base_lin_vel in observations.
  # Set to false when no state estimator is available (omits base_lin_vel, reduces obs_dim by 3).
  # Can be overridden by --no-est CLI flag.
  use_estimator: true

  # Joints included in observation, in policy-expected ORDER.
  # If omitted or null, defaults to controlled_joints.
  # If both omitted, all joints in robot-native order.
  observed_joints: null

  # Joints controlled by the policy (receiving actions), in policy-expected ORDER.
  # If omitted or null, all joints are controlled in robot-native order.
  controlled_joints: null

  # Example: Full observe, legs control (common for locomotion)
  # observed_joints: [all 29 joints in IsaacLab order]
  # controlled_joints: [12 leg joints only]

  # Example: Isolated arm control (arm doesn't need leg info)
  # observed_joints: null  # defaults to controlled_joints
  # controlled_joints:
  #   - R_SHOULDER_PITCH
  #   - R_SHOULDER_ROLL
  #   - ... (7 arm joints)

control:
  policy_frequency: 50       # Hz
  sim_frequency: 200         # Hz (sim only)

  # PD gains for controlled joints (per-joint array or scalar for all)
  kp: 100.0                  # Position gain
  kd: 10.0                   # Velocity gain
  ka: 0.5                    # Action scale

  # Damping mode gains (E-stop AND non-controlled joints)
  kd_damp: 5.0

  # Home position (optional, defaults to standing)
  # If specified, must include ALL joints (not just controlled)
  q_home: null               # If null, use robot default

safety:
  joint_position_limits: true
  joint_velocity_limits: true
  torque_limits: true

network:
  interface: lo              # Overridden by CLI for real robot
  domain_id: 1               # Overridden by CLI

# [Docker] Viser configuration
viser:
  host: "0.0.0.0"
  port: 8080
  default_camera: follow     # follow, fixed, orbit
  default_render: mesh       # mesh, collision

# [Metal] Viewer configuration (replaces viser section)
viewer:
  default_camera: follow     # follow, fixed, orbit
  default_render: mesh       # mesh, collision

logging:
  enabled: true
  format: hdf5               # hdf5 or npz
  compression: gzip
  log_frequency: 50          # Hz (match policy frequency)
```

### 11.2 Example Configuration: Full Observe, Legs Control (Locomotion Policy)

This is the most common pattern for locomotion policies - the policy observes the full robot state (for balance/coordination awareness) but only controls the legs.

```yaml
# configs/locomotion_legs.yaml
# Observation: 29 joints, Action: 12 joints
# Obs dim = 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 3 (vel_cmd) + 29 (joint_pos) + 29 (joint_vel) + 12 (actions) = 82
# With use_estimator: false: 3 + 3 + 3 + 29 + 29 + 12 = 79

policy:
  # Full body observation (all 29 joints in IsaacLab order)
  observed_joints:
    - left_hip_pitch_joint        # IsaacLab idx 0
    - right_hip_pitch_joint       # IsaacLab idx 1
    - waist_yaw_joint             # IsaacLab idx 2
    - left_hip_roll_joint         # IsaacLab idx 3
    - right_hip_roll_joint        # IsaacLab idx 4
    - waist_roll_joint            # IsaacLab idx 5
    - left_hip_yaw_joint          # IsaacLab idx 6
    - right_hip_yaw_joint         # IsaacLab idx 7
    - waist_pitch_joint           # IsaacLab idx 8
    - left_knee_joint             # IsaacLab idx 9
    - right_knee_joint            # IsaacLab idx 10
    - left_shoulder_pitch_joint   # IsaacLab idx 11
    - right_shoulder_pitch_joint  # IsaacLab idx 12
    - left_ankle_pitch_joint      # IsaacLab idx 13
    - right_ankle_pitch_joint     # IsaacLab idx 14
    - left_shoulder_roll_joint    # IsaacLab idx 15
    - right_shoulder_roll_joint   # IsaacLab idx 16
    - left_ankle_roll_joint       # IsaacLab idx 17
    - right_ankle_roll_joint      # IsaacLab idx 18
    - left_shoulder_yaw_joint     # IsaacLab idx 19
    - right_shoulder_yaw_joint    # IsaacLab idx 20
    - left_elbow_joint            # IsaacLab idx 21
    - right_elbow_joint           # IsaacLab idx 22
    - left_wrist_roll_joint       # IsaacLab idx 23
    - right_wrist_roll_joint      # IsaacLab idx 24
    - left_wrist_pitch_joint      # IsaacLab idx 25
    - right_wrist_pitch_joint     # IsaacLab idx 26
    - left_wrist_yaw_joint        # IsaacLab idx 27
    - right_wrist_yaw_joint       # IsaacLab idx 28

  # Only control legs (12 joints, in IsaacLab order)
  controlled_joints:
    - left_hip_pitch_joint
    - right_hip_pitch_joint
    - left_hip_roll_joint
    - right_hip_roll_joint
    - left_hip_yaw_joint
    - right_hip_yaw_joint
    - left_knee_joint
    - right_knee_joint
    - left_ankle_pitch_joint
    - right_ankle_pitch_joint
    - left_ankle_roll_joint
    - right_ankle_roll_joint

# Torso and arm joints will be in damping mode
```

### 11.3 Example Configuration: Legs Only, Isolated (12 DOF)

For a simpler policy that doesn't need upper body awareness:

```yaml
# configs/legs_only_isolated.yaml
# Observation: 12 joints, Action: 12 joints
# Obs dim = 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 3 (vel_cmd) + 12 (joint_pos) + 12 (joint_vel) + 12 (actions) = 48
# With use_estimator: false: 3 + 3 + 3 + 12 + 12 + 12 = 45

policy:
  # observed_joints defaults to controlled_joints when omitted
  controlled_joints:
    - L_LEG_HIP_YAW
    - L_LEG_HIP_ROLL
    - L_LEG_HIP_PITCH
    - L_LEG_KNEE
    - L_LEG_ANKLE_PITCH
    - L_LEG_ANKLE_ROLL
    - R_LEG_HIP_YAW
    - R_LEG_HIP_ROLL
    - R_LEG_HIP_PITCH
    - R_LEG_KNEE
    - R_LEG_ANKLE_PITCH
    - R_LEG_ANKLE_ROLL
```

### 11.4 Example Configuration: Right Arm Only (7 DOF)

```yaml
# configs/right_arm_only.yaml
# Observation: 7 joints, Action: 7 joints
# Obs dim = 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 3 (vel_cmd) + 7 (joint_pos) + 7 (joint_vel) + 7 (actions) = 33
# With use_estimator: false: 3 + 3 + 3 + 7 + 7 + 7 = 30

policy:
  # observed_joints defaults to controlled_joints
  controlled_joints:
    - R_SHOULDER_PITCH
    - R_SHOULDER_ROLL
    - R_SHOULDER_YAW
    - R_ELBOW
    - R_WRIST_ROLL
    - R_WRIST_PITCH
    - R_WRIST_YAW

control:
  # Arm-specific gains (gentler than legs)
  kp: 50.0
  kd: 5.0
  ka: 0.3
  kd_damp: 2.0  # Gentle damping for non-controlled joints
```

### 11.5 Example Configuration: Full Body with IsaacLab Joint Order (29 DOF)

```yaml
# configs/isaaclab_fullbody.yaml
# Full body control with IsaacLab joint ordering
# Observation: 29 joints, Action: 29 joints
# Obs dim = 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 3 (vel_cmd) + 29 (joint_pos) + 29 (joint_vel) + 29 (actions) = 99
# With use_estimator: false: 3 + 3 + 3 + 29 + 29 + 29 = 96

policy:
  # When observed_joints is omitted, it defaults to controlled_joints
  controlled_joints:
    # All 29 joints in IsaacLab interleaved left/right order
    - left_hip_pitch_joint        # IsaacLab index 0
    - right_hip_pitch_joint       # IsaacLab index 1
    - waist_yaw_joint             # IsaacLab index 2
    - left_hip_roll_joint         # IsaacLab index 3
    - right_hip_roll_joint        # IsaacLab index 4
    - waist_roll_joint            # IsaacLab index 5
    - left_hip_yaw_joint          # IsaacLab index 6
    - right_hip_yaw_joint         # IsaacLab index 7
    - waist_pitch_joint           # IsaacLab index 8
    - left_knee_joint             # IsaacLab index 9
    - right_knee_joint            # IsaacLab index 10
    - left_shoulder_pitch_joint   # IsaacLab index 11
    - right_shoulder_pitch_joint  # IsaacLab index 12
    - left_ankle_pitch_joint      # IsaacLab index 13
    - right_ankle_pitch_joint     # IsaacLab index 14
    - left_shoulder_roll_joint    # IsaacLab index 15
    - right_shoulder_roll_joint   # IsaacLab index 16
    - left_ankle_roll_joint       # IsaacLab index 17
    - right_ankle_roll_joint      # IsaacLab index 18
    - left_shoulder_yaw_joint     # IsaacLab index 19
    - right_shoulder_yaw_joint    # IsaacLab index 20
    - left_elbow_joint            # IsaacLab index 21
    - right_elbow_joint           # IsaacLab index 22
    - left_wrist_roll_joint       # IsaacLab index 23
    - right_wrist_roll_joint      # IsaacLab index 24
    - left_wrist_pitch_joint      # IsaacLab index 25
    - right_wrist_pitch_joint     # IsaacLab index 26
    - left_wrist_yaw_joint        # IsaacLab index 27
    - right_wrist_yaw_joint       # IsaacLab index 28

control:
  # Per-joint gains in IsaacLab order (matching controlled_joints above)
  kp: [100, 100, 200,             # hip_pitch L/R, waist_yaw
       100, 100, 200,             # hip_roll L/R, waist_roll
       100, 100, 200,             # hip_yaw L/R, waist_pitch
       150, 150,                  # knee L/R
       80, 80,                    # shoulder_pitch L/R
       50, 50,                    # ankle_pitch L/R
       80, 80,                    # shoulder_roll L/R
       50, 50,                    # ankle_roll L/R
       80, 80,                    # shoulder_yaw L/R
       60, 60,                    # elbow L/R
       40, 40,                    # wrist_roll L/R
       40, 40,                    # wrist_pitch L/R
       40, 40]                    # wrist_yaw L/R
```

### 11.6 Example Configuration: G1-23 Full Body (23 DOF)

Configuration for the 23-DOF G1 variant with single torso joint and 5-DOF arms.

```yaml
# configs/g1_23dof_fullbody.yaml
# G1-23 variant: 12 leg + 1 torso + 10 arm = 23 joints
# Obs dim = 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 3 (vel_cmd) + 23 (joint_pos) + 23 (joint_vel) + 23 (actions) = 81
# With use_estimator: false: 3 + 3 + 3 + 23 + 23 + 23 = 78

robot:
  variant: g1_23dof

policy:
  # All 23 joints in robot-native order (same as controlled for full-body)
  controlled_joints:
    # Left leg (same as 29-DOF)
    - left_hip_pitch
    - left_hip_roll
    - left_hip_yaw
    - left_knee
    - left_ankle_pitch
    - left_ankle_roll
    # Right leg (same as 29-DOF)
    - right_hip_pitch
    - right_hip_roll
    - right_hip_yaw
    - right_knee
    - right_ankle_pitch
    - right_ankle_roll
    # Torso (single joint, not 3 like 29-DOF)
    - torso
    # Left arm (5-DOF: no wrist, elbow is pitch+roll)
    - left_shoulder_pitch
    - left_shoulder_roll
    - left_shoulder_yaw
    - left_elbow_pitch
    - left_elbow_roll
    # Right arm (5-DOF)
    - right_shoulder_pitch
    - right_shoulder_roll
    - right_shoulder_yaw
    - right_elbow_pitch
    - right_elbow_roll

control:
  # Per-joint gains for 23-DOF
  kp: [100, 100, 100, 150, 50, 50,   # Left leg
       100, 100, 100, 150, 50, 50,   # Right leg
       200,                          # Torso (single joint)
       80, 80, 80, 60, 60,           # Left arm (5 joints)
       80, 80, 80, 60, 60]           # Right arm (5 joints)
  kd: [5, 5, 5, 10, 5, 5,            # Left leg
       5, 5, 5, 10, 5, 5,            # Right leg
       5,                            # Torso
       3, 3, 3, 3, 3,                # Left arm
       3, 3, 3, 3, 3]                # Right arm
```

### 11.7 Reference PD Gains from BeyondMimic

These per-joint gains are derived from the BeyondMimic motion_tracking_controller and provide a good starting point for tuning. Note that actual gains may need adjustment based on the specific policy.

```yaml
# Reference gains for 29-DOF (from BeyondMimic standby controller)
control:
  kp:
    # Legs (higher gains for load-bearing)
    - 350.0  # left_hip_pitch
    - 200.0  # left_hip_roll
    - 200.0  # left_hip_yaw
    - 300.0  # left_knee
    - 300.0  # left_ankle_pitch
    - 150.0  # left_ankle_roll
    - 350.0  # right_hip_pitch
    - 200.0  # right_hip_roll
    - 200.0  # right_hip_yaw
    - 300.0  # right_knee
    - 300.0  # right_ankle_pitch
    - 150.0  # right_ankle_roll
    # Torso
    - 200.0  # waist_yaw
    - 200.0  # waist_roll
    - 200.0  # waist_pitch
    # Arms (lower gains for compliance)
    - 40.0   # left_shoulder_pitch
    - 40.0   # left_shoulder_roll
    - 40.0   # left_shoulder_yaw
    - 40.0   # left_elbow
    - 40.0   # left_wrist_roll
    - 40.0   # left_wrist_pitch
    - 40.0   # left_wrist_yaw
    - 40.0   # right_shoulder_pitch
    - 40.0   # right_shoulder_roll
    - 40.0   # right_shoulder_yaw
    - 40.0   # right_elbow
    - 40.0   # right_wrist_roll
    - 40.0   # right_wrist_pitch
    - 40.0   # right_wrist_yaw
  kd:
    # Legs
    - 5.0    # left_hip_pitch
    - 5.0    # left_hip_roll
    - 5.0    # left_hip_yaw
    - 10.0   # left_knee
    - 5.0    # left_ankle_pitch
    - 5.0    # left_ankle_roll
    - 5.0    # right_hip_pitch
    - 5.0    # right_hip_roll
    - 5.0    # right_hip_yaw
    - 10.0   # right_knee
    - 5.0    # right_ankle_pitch
    - 5.0    # right_ankle_roll
    # Torso
    - 5.0    # waist_yaw
    - 5.0    # waist_roll
    - 5.0    # waist_pitch
    # Arms
    - 3.0    # left_shoulder_pitch
    - 3.0    # left_shoulder_roll
    - 3.0    # left_shoulder_yaw
    - 3.0    # left_elbow
    - 3.0    # left_wrist_roll
    - 3.0    # left_wrist_pitch
    - 3.0    # left_wrist_yaw
    - 3.0    # right_shoulder_pitch
    - 3.0    # right_shoulder_roll
    - 3.0    # right_shoulder_yaw
    - 3.0    # right_elbow
    - 3.0    # right_wrist_roll
    - 3.0    # right_wrist_pitch
    - 3.0    # right_wrist_yaw
```

---

## 12. Testing Requirements

### 12.1 Unit Tests (Shared)

| Component | Tests |
|-----------|-------|
| `joint_mapper.py` | Joint reordering, observed vs controlled separation, robot↔policy mapping, invalid joint names |
| `observations.py` | Observation vector construction (IsaacLab PolicyCfg order), relative joint positions, `use_estimator` mode, projected gravity |
| `isaaclab_policy.py` | IsaacLab policy loading, inference, input/output shapes |
| `beyondmimic_policy.py` | BeyondMimic policy loading, metadata extraction, multi-output inference, time stepping |
| `safety.py` | E-stop state machine, damping torque calculation |
| `config.py` | Configuration loading, validation, controlled_joints parsing |
| `logger.py` | Log writing, compression, replay parsing |

### 12.1M Unit Tests [Metal]

| Component | Tests |
|-----------|-------|
| `compat.py` | `RecurrentThread` start/stop/shutdown, timing accuracy, `resolve_network_interface()` returns correct loopback per platform |
| `config.py` | `ViewerConfig` loaded instead of `ViserConfig` when in Metal mode |

### 12.2 Integration Tests (Shared)

| Test | Description |
|------|-------------|
| Sim standalone (29-DOF) | MuJoCo simulation runs without policy |
| Sim standalone (23-DOF) | MuJoCo simulation with 23-DOF model runs correctly |
| Sim + IsaacLab policy | IsaacLab policy controls simulated robot with velocity commands |
| Sim + BeyondMimic policy | BeyondMimic policy runs motion tracking with time-indexed trajectories |
| Sim + subset policy | Subset control with non-controlled joints in damping mode |
| Sim + full obs partial ctrl | Full observation (29 joints) with partial control (12 legs) |
| Sim + reordered joints | Policy with different joint ordering than robot-native |
| 23-DOF joint mapping | 23-DOF joint names correctly mapped (torso, elbow_pitch/roll) |
| Policy format detection | Auto-detect IsaacLab vs BeyondMimic from ONNX structure |
| BeyondMimic metadata | Correctly extract gains/joints from ONNX metadata |
| E-stop sim | E-stop triggers and clears correctly |
| Log roundtrip | Write log, read log, verify data integrity |

### 12.2D Integration Tests [Docker]

| Test | Description |
|------|-------------|
| Viser connection | Browser can connect and see robot |
| Viser keyboard events | Keyboard shortcuts work in Viser UI |
| Policy dropdown | Runtime policy switching via Viser dropdown |

### 12.2M Integration Tests [Metal]

| Test | Description |
|------|-------------|
| Headless sim | `--headless --duration 5` runs and exits cleanly |
| Headless steps | `--headless --steps 100` runs correct number of steps |
| WASD velocity | Velocity commands update correctly from keyboard input (manual test with viewer) |
| DDS loopback macOS | CycloneDDS communicates over `lo0` on macOS |

### 12.3 Manual Tests (Checklist)

**Shared:**
- [ ] Subset control: non-controlled joints stay in damping mode
- [ ] Full observe, partial control: observation includes all joints, actions only for subset
- [ ] Joint reordering: policy with different order works correctly
- [ ] Mesh/collision toggle works
- [ ] E-stop latches and clears
- [ ] Logs generated and readable on host
- [ ] 29-DOF simulation loads and runs correctly
- [ ] 23-DOF simulation loads and runs correctly
- [ ] 23-DOF home position is correct (check arm/torso joints)
- [ ] Switching between 29-DOF and 23-DOF via config works

**[Docker]:**
- [ ] Docker builds on Ubuntu 20.04
- [ ] Docker builds on Ubuntu 22.04
- [ ] Docker builds on Ubuntu 24.04
- [ ] Docker runs on macOS (Apple Silicon)
- [ ] Docker runs on macOS (Intel)
- [ ] Viser accessible from different device on LAN
- [ ] Policy dropdown shows available policies
- [ ] Camera view switching works (dropdown)
- [ ] Velocity command sliders work

**[Metal]:**
- [ ] `uv venv && uv pip install -e ".[dev]"` succeeds on macOS ARM64
- [ ] `scripts/validate_macos.py` passes all checks
- [ ] MuJoCo viewer opens and renders robot
- [ ] WASD keys adjust velocity commands
- [ ] Camera view switching works (1/2/3 keys)
- [ ] `--headless --duration 10` runs and exits
- [ ] `--headless --steps 500` runs correct number of steps
- [ ] Headless stdout status prints at 1 Hz
- [ ] CycloneDDS works over `lo0` on macOS

---

## 13. Example Workflow

### 13.1 [Docker] Development Session (Simulation)

```bash
# 1. Start the container with volume mounts
docker-compose up -d

# 2. Enter the container
docker exec -it unitree_launcher bash

# 3. Run simulation with a policy
python -m src.main sim --policy policies/walk_v1.onnx

# 4. Open browser to http://localhost:8080
#    - Use velocity sliders to command robot
#    - Monitor telemetry
#    - Test E-stop

# 5. Stop and review logs
#    Logs saved to logs/2024-01-15_14-30-22_sim_walk_v1/

# 6. (On host) Analyze logs
python scripts/replay_log.py logs/2024-01-15_14-30-22_sim_walk_v1/ --format csv
```

### 13.1M [Metal] Development Session (Simulation)

```bash
# 1. Set up environment (first time only)
uv venv
uv pip install -e ".[dev]"

# 2. Validate macOS compatibility (first time only)
python scripts/validate_macos.py

# 3. Run simulation with a policy (opens MuJoCo viewer)
python -m src.main sim --policy policies/walk_v1.onnx

# 4. In the MuJoCo viewer window:
#    - Press Space to start the policy
#    - Use W/S/A/D/Q/Z to adjust velocity commands
#    - Press X to zero velocity
#    - Press E for E-stop, C to clear
#    - Press M to toggle mesh/collision
#    - Press R to reset simulation
#    - Close window or Ctrl+C to exit

# 5. Review logs
python scripts/replay_log.py logs/2024-01-15_14-30-22_sim_walk_v1/ --format csv
```

### 13.1H [Metal] Headless Batch Evaluation

```bash
# Run a 60-second headless evaluation
python -m src.main sim \
    --policy policies/walk_v1.onnx \
    --headless --duration 60

# Output (1 Hz to stdout):
# [00:01.0] step=50  state=RUNNING vel=[0.0, 0.0, 0.0] hz=50.1
# [00:02.0] step=100 state=RUNNING vel=[0.0, 0.0, 0.0] hz=50.0
# ...
# [01:00.0] step=3000 state=RUNNING vel=[0.0, 0.0, 0.0] hz=50.0
# Session complete. Logs: logs/2024-01-15_14-30-22_sim_walk_v1/

# Analyze logs
python scripts/replay_log.py logs/2024-01-15_14-30-22_sim_walk_v1/ --format csv
```

### 13.2 [Docker] Real Robot Deployment

```bash
# 1. Connect laptop to G1 via ethernet
#    Configure network interface (e.g., eth0) with appropriate IP

# 2. Start container with host networking
docker run --network host -v ./policies:/app/policies unitree_launcher \
    python -m src.main real --interface eth0 --policy policies/walk_v1.onnx

# 3. Open browser to http://<laptop-ip>:8080
#    - Verify robot state in telemetry
#    - Start with small velocity commands
#    - Keep finger on E-stop!

# 4. Stop and review logs
```

### 13.2M [Metal] Real Robot Deployment

```bash
# 1. Connect laptop to G1 via ethernet
#    Configure network interface (e.g., en0) with appropriate IP

# 2. Run with MuJoCo viewer (for visual feedback)
python -m src.main real --interface en0 --policy policies/walk_v1.onnx

# 3. In the MuJoCo viewer window:
#    - Verify robot state displayed (viewer shows real-time robot pose)
#    - Press Space to start policy
#    - Use W/S/A/D for velocity (start small: single tap = 0.1 m/s)
#    - Keep finger on E key for E-stop!

# 4. Stop and review logs
```

---

## 14. References

### 14.1 Primary References (In This Repo)

**`reference/unitree_mujoco/`** - Unitree's official MuJoCo simulator
- Contains G1 MJCF models (`unitree_robots/g1/g1_29dof.xml`, `g1_23dof.xml`)
- Python simulation example (`simulate_python/unitree_mujoco.py`)
- DDS bridge implementation (`simulate_python/unitree_sdk2py_bridge.py`)
- Joint index documentation (`unitree_robots/g1/g1_joint_index_dds.md`)
- **Key file to study**: `simulate_python/unitree_sdk2py_bridge.py` shows how to interface MuJoCo with DDS

### 14.2 Unitree Official Repositories

**https://github.com/unitreerobotics/unitree_sdk2_python** - Python SDK
- Primary interface for robot communication
- DDS-based pub/sub for `LowCmd`/`LowState` messages
- Required dependency: CycloneDDS
- **Study**: Examples in `/example/` directory

**https://github.com/unitreerobotics/unitree_sdk2** - C++ SDK
- Reference for understanding DDS message formats
- IDL definitions in `/idl/` directory
- Useful for understanding low-level protocol

**https://github.com/unitreerobotics/unitree_ros** - ROS1 packages (reference only)
**https://github.com/unitreerobotics/unitree_ros2** - ROS2 packages
- Alternative integration path if ROS2 is chosen
- Contains launch files and message definitions

**https://support.unitree.com/home/zh/developer** - Official documentation
- Hardware specs, safety guidelines
- Network configuration for robot connection

### 14.3 Visualization References

**https://github.com/nerfstudio-project/viser** - Viser visualization library
- Browser-based 3D visualization
- WebSocket server architecture
- **Key features**: Scene graph API, GUI panels, camera controls

**https://viser.studio/main/** - Viser documentation
- API reference and examples
- GUI element types (buttons, sliders, dropdowns)

**https://github.com/mujocolab/mjlab** - Viser + MuJoCo integration
- **Critical reference** for implementing robot visualization
- Shows how to render MuJoCo scenes in Viser
- Interactive viewer implementation

### 14.4 BeyondMimic / Motion Tracking References

**https://github.com/HybridRobotics/whole_body_tracking** - BeyondMimic training code
- Policy training for motion tracking
- **Key file**: `utils/exporter.py` - ONNX export with embedded metadata
- **Key file**: `tasks/tracking/mdp/observations.py` - Observation definitions
- Observation terms include body-relative states and motion targets

**https://github.com/HybridRobotics/motion_tracking_controller** - BeyondMimic deployment
- **Critical reference** for BeyondMimic policy format
- C++ ONNX inference with metadata extraction
- ROS-based control architecture
- MuJoCo simulation and real robot deployment
- Shows how to extract joint names, gains, trajectories from ONNX metadata

### 14.5 Community Examples

**https://github.com/catachiii/crl-humanoid-ros** - ROS-based humanoid setup
- Alternative ROS integration approach
- Humanoid-specific considerations

**https://github.com/amazon-far/holosoma** - Python SDK usage
- Clean Python interface patterns
- Policy deployment examples

**https://github.com/boston-dynamics/spot-rl-example** - Spot RL interface
- Not Unitree, but excellent interface design reference
- Policy abstraction patterns
- Safety system design

### 14.6 Documentation Links

| Library | Documentation | Notes |
|---------|---------------|-------|
| MuJoCo | https://mujoco.readthedocs.io/ | Physics, MJCF format, Python bindings |
| Viser | https://viser.studio/main/ | Scene API, GUI elements |
| ONNX Runtime | https://onnxruntime.ai/docs/ | Python API, inference sessions |
| CycloneDDS | https://cyclonedds.io/docs/ | DDS configuration, networking |
| IsaacLab | https://isaac-sim.github.io/IsaacLab/ | Policy training, export format |
| BeyondMimic | https://beyond-mimic.github.io/ | Motion tracking paper and method |

---

## 15. Open Questions / Future Work

Items explicitly deferred from this version:

1. **Dexterous hands**: Dex3-1 hand control integration
2. **WiFi connectivity**: Currently ethernet-only
3. **GPU inference**: Not needed for small policies
4. **Terrain visualization**: Currently flat ground only
5. **ROS2 integration**: May add if ecosystem benefits justify complexity
6. **Watchdog safety**: Auto E-stop on communication timeout
7. **Joint limit safety**: Auto E-stop on limit violation
8. **Recording playback in Viser**: Replay visualization in browser **[Docker]**
9. **[Metal] Runtime policy switching**: Currently only `--policy` CLI arg; future work to add `--policy-dir` based selection via keyboard shortcut
10. **[Metal] Secondary IMU**: Not currently in `RobotState`; may be needed for ankle IMU data on real robot
11. **[Metal] WiFi DDS**: CycloneDDS over WiFi interface (currently loopback/ethernet only)
