# Unitree G1 Deployment Stack

Control stack for the Unitree G1 humanoid robot. Supports both MuJoCo simulation (macOS/Linux) and real robot deployment (Linux + Ethernet). Runs ONNX neural network policies at 50 Hz with safety enforcement.

## Supported Policy Formats

- **IsaacLab** -- Standard locomotion policies trained in Isaac Lab. Velocity-commanded (arrow key controls).
- **BeyondMimic** -- Motion-tracking policies with trajectory playback. Includes ONNX metadata for per-joint gains.

## Requirements

- Python 3.10 (hard ceiling -- CycloneDDS ARM64 wheel limit)
- macOS (Apple Silicon) or Linux (x86_64 / ARM64), or Docker
- Real robot deployment requires Linux + Ethernet connection to the G1

## Hardware Setup (Real Robot)

### Network Configuration

The G1 communicates over Ethernet using DDS (CycloneDDS). Connect your machine directly to the robot's Ethernet port.

| Parameter | Value |
|-----------|-------|
| Robot IP | `192.168.123.161` |
| Subnet | `192.168.123.x/24` |
| Host IP | Any unused address on `192.168.123.x` (e.g., `.100`) |
| DDS domain | `0` |

Configure a static IP on your Ethernet interface:

```bash
# Linux (replace enp3s0 with your interface name)
sudo ip addr add 192.168.123.100/24 dev enp3s0
sudo ip link set enp3s0 up

# Verify connectivity
ping 192.168.123.161
```

### DDS Topics

| Topic | Direction | Message type | Rate |
|-------|-----------|-------------|------|
| `rt/lowstate` | Robot -> Host | `LowState_` | ~500 Hz |
| `rt/lowcmd` | Host -> Robot | `LowCmd_` | 500 Hz (required) |

### Protocol Fields

- **`mode_machine`**: Echoed from `LowState_` into every `LowCmd_`. Must match the robot's current mode (`5` = 29-DOF).
- **`mode_pr`**: Set to `0` in every `LowCmd_` (position control for pitch/roll ankles).
- **Motor slots**: All 35 IDL motor slots are filled. Controlled joints (0-28) get `mode=0x01` (PMSM servo). Non-controlled slots (29-34) get `mode=0` with zeroed fields.

### 500 Hz Command Publishing

The robot's onboard controller expects commands at ~500 Hz. A dedicated `RecurrentThread` re-publishes the latest `LowCmd_` every 2 ms. The control loop (50 Hz) updates the command contents; between updates, the publish thread re-sends the most recent command to keep the robot's communication watchdog satisfied.

### Verification

Confirm DDS communication before running policies:

```python
from unitree_launcher.compat import patch_unitree_threading, resolve_network_interface
patch_unitree_threading()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

ChannelFactoryInitialize(0, "enp3s0")  # your interface
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(handler=lambda msg: print(
    f"mode_machine={msg.mode_machine}, "
    f"q[0]={msg.motor_state[0].q:.3f}"
), queueLen=1)

import time; time.sleep(5)
```

You should see `mode_machine=5` and streaming joint positions.

## Installation

```bash
# Create virtual environment
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"
```

**Note:** The Unitree SDK is installed from GitHub source. If you encounter an import error about a missing `b2` submodule, patch `.venv/.../unitree_sdk2py/__init__.py` to wrap the `b2` import in a try/except. See LOG.md Phase 0 for details.

## Docker

Docker provides a portable way to run simulations, evaluations, and tests without installing dependencies locally. The image supports both headless (EGL) and GUI (GLX/X11) rendering. Docker files live in `docker/`.

### Build

```bash
docker build -f docker/Dockerfile -t unitree-launcher .
```

### Headless Simulation (Any Platform)

```bash
docker run --rm \
    -v ./logs:/app/logs \
    -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher sim --headless --policy assets/policies/stance_29dof.onnx --duration 10
```

Or with Docker Compose:

```bash
docker compose -f docker/docker-compose.yml --profile headless run --rm sim-headless \
    sim --headless --policy assets/policies/stance_29dof.onnx --duration 10
```

### GUI via X11 (Linux Only)

```bash
# Allow Docker to connect to X11
xhost +local:docker

docker run --rm \
    -e DISPLAY=$DISPLAY \
    -e MUJOCO_GL=glx \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ./logs:/app/logs \
    -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher sim --policy assets/policies/stance_29dof.onnx
```

### Real Robot in Docker (Linux Only)

Requires host networking so DDS can reach the robot:

```bash
docker run --rm --network host \
    -v ./logs:/app/logs \
    -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher real --policy assets/policies/stance_29dof.onnx --interface eth0
```

### Run Tests in Docker

```bash
docker compose -f docker/docker-compose.yml --profile test run --rm test
```

## Quick Start

### Simulation with Viewer

```bash
# macOS requires mjpython (ships with the mujoco package)
mjpython -m unitree_launcher.main sim --policy path/to/policy.onnx

# Linux — standard python works
python -m unitree_launcher.main sim --policy path/to/policy.onnx
```

A MuJoCo viewer window opens with the G1 robot. Use keyboard controls to operate (see [Keybindings](#keybindings) below). On macOS, `mjpython` is required because MuJoCo's passive viewer must run on the main thread.

### Gantry Mode (Simulation)

```bash
# With viewer (use mjpython on macOS)
mjpython -m unitree_launcher.main sim --gantry

# Headless
python -m unitree_launcher.main sim --gantry --headless
```

Runs a gantry hang sequence: DAMPING (5s) -> INTERPOLATE (5s) -> HOLD (5s) -> DAMPING (5s). No policy needed. The robot hangs from an elastic band, settles under gravity, smoothly interpolates to home position with IsaacLab gains, holds, then damps to rest.

### Headless Simulation

```bash
python -m unitree_launcher.main sim --policy path/to/policy.onnx --headless --duration 30
```

Runs without a viewer. Prints status to stdout at 1 Hz. Auto-terminates after 30 seconds.

### Real Robot

```bash
python -m unitree_launcher.main real --policy path/to/policy.onnx --interface eth0
```

Connects to the G1 over Ethernet via DDS. Runs in headless mode (no viewer). The robot must be powered on and reachable on the specified network interface.

### Shell Scripts

```bash
./scripts/run_sim.sh  --policy path/to/policy.onnx            # Viewer mode
./scripts/run_eval.sh --policy path/to/policy.onnx --steps 500 # Headless eval
./scripts/run_real.sh --policy path/to/policy.onnx --interface eth0
```

## CLI Reference

```
python -m unitree_launcher.main {sim,real} [options]
```

### Sub-commands

| Sub-command | Description |
|-------------|-------------|
| `sim`       | MuJoCo simulation (viewer or headless) |
| `real`      | Real robot via DDS over Ethernet |

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--policy PATH` | *(required unless `--gantry`)* | Path to ONNX policy file |
| `--config PATH` | `configs/default.yaml` | YAML configuration file |
| `--robot VARIANT` | from config | Robot variant: `g1_29dof` or `g1_23dof` |
| `--policy-dir DIR` | none | Directory of ONNX files for runtime switching (`=`/`-` keys) |
| `--domain-id INT` | 1 (sim) / 0 (real) | DDS domain ID |
| `--log-dir DIR` | `logs/` | Log output directory |
| `--no-log` | false | Disable logging |
| `--no-est` | false | Omit base linear velocity from observations |

### Sim-only Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gantry` | false | Gantry mode: damping -> interpolate -> hold (no `--policy` needed) |
| `--headless` | false | Run without MuJoCo viewer |
| `--duration SECS` | none | Auto-stop after N seconds (headless) |
| `--steps N` | none | Auto-stop after N policy steps (headless) |

### Real-only Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--interface NAME` | *(required)* | Network interface (e.g. `eth0`, `enp3s0`) |

## Keybindings

These work in the MuJoCo viewer window. Mouse controls (orbit, pan, zoom) work as normal. Keys are chosen to avoid conflicts with MuJoCo's built-in viewer shortcuts (which use most letter keys for rendering toggles like wireframe, shadow, etc.).

| Key | Action |
|-----|--------|
| `Space` | Start / stop the policy (toggle) |
| `Backspace` | Emergency stop (latching) |
| `Enter` | Clear E-stop (transitions to STOPPED) |
| `Delete` (Fn+Backspace on Mac) | Reset robot to home position |
| `Up` / `Down` | Increase / decrease forward velocity (+/- 0.1) |
| `Left` / `Right` | Increase / decrease lateral velocity (+/- 0.1) |
| `,` / `.` | Increase / decrease yaw rate (+/- 0.1) |
| `/` | Zero all velocity commands |
| `=` / `-` | Next / previous policy in `--policy-dir` |

### State Machine

```
IDLE --(Space)--> RUNNING --(Space)--> STOPPED --(Space)--> RUNNING
                    |                     |
                    +--(Backspace)--> ESTOP <--(Backspace)--+
                                       |
                                    (Enter)
                                       |
                                    STOPPED
```

## Configuration

Configuration is loaded from YAML files in `configs/`. CLI arguments override config values.

### `configs/default.yaml`

```yaml
robot:
  variant: g1_29dof       # g1_29dof or g1_23dof

policy:
  use_onnx_metadata: true # Use gains from ONNX metadata (BeyondMimic)
  use_estimator: true     # Include base_lin_vel in observations

control:
  policy_frequency: 50    # Hz -- policy inference rate
  sim_frequency: 200      # Hz -- MuJoCo physics rate
  kp: 100.0               # Position gain (scalar or per-joint list)
  kd: 10.0                # Velocity gain
  ka: 0.5                 # Action scale
  kd_damp: 5.0            # Damping gain for non-controlled joints

safety:
  joint_position_limits: true
  joint_velocity_limits: true
  torque_limits: true

network:
  interface: "auto"       # "auto" resolves to loopback
  domain_id: 1            # DDS domain (sim=1, real=0)

logging:
  format: hdf5            # hdf5 or npz
  compression: gzip
```

### Variant Configs

- `configs/g1_29dof.yaml` -- 29-DOF (full body including wrists). Includes BeyondMimic reference gains as comments.
- `configs/g1_23dof.yaml` -- 23-DOF (no wrist pitch/yaw, no waist roll/pitch). 6 unused joints receive passive damping.

## Logging

Each run creates a timestamped directory under `--log-dir`:

```
logs/20260211_143022_sim_walk_policy/
  metadata.yaml    # Full config snapshot
  data.hdf5        # Time-series data (gzip compressed)
  events.json      # Discrete events (start, stop, estop)
```

### Logged Data

18 channels at policy frequency: joint positions, velocities, torques, IMU quaternion/gyro/accel, base position/velocity, observations, actions, commanded positions, gains, system state, velocity commands, and timing.

### Replay and Export

```bash
# Print summary
python scripts/replay_log.py logs/<run>/

# Export to CSV
python scripts/replay_log.py logs/<run>/ --format csv --output data.csv
```

## Project Structure

```
unitree_launcher/
  src/unitree_launcher/           # Installable Python package
    main.py                       # CLI entry point, viewer, headless runner
    config.py                     # Constants, dataclasses, YAML loading
    compat.py                     # macOS compatibility (RecurrentThread shim)
    gantry.py                     # Gantry harness utilities (elastic band)
    mirror.py                     # Real robot state mirroring
    robot/
      base.py                     # RobotState, RobotCommand, RobotInterface ABC
      sim_robot.py                # MuJoCo simulation backend
      real_robot.py               # DDS communication with physical robot
    policy/
      base.py                     # PolicyInterface ABC, format auto-detection
      joint_mapper.py             # Joint ordering between robot/policy/MuJoCo
      observations.py             # IsaacLab observation vector construction
      isaaclab_policy.py          # IsaacLab ONNX policy wrapper
      beyondmimic_policy.py       # BeyondMimic ONNX policy wrapper
    control/
      controller.py               # Main control loop (50 Hz), command building
      safety.py                   # State machine, E-stop, command clamping
    logging/
      logger.py                   # HDF5/NPZ time-series logging
      replay.py                   # Log loading, CSV export, summary
  configs/
    default.yaml                  # Default configuration
    g1_29dof.yaml                 # 29-DOF variant
    g1_23dof.yaml                 # 23-DOF variant
  assets/robots/g1/
    g1_29dof.xml                  # MuJoCo robot model (29 actuators)
    g1_23dof.xml                  # MuJoCo robot model (23 controlled)
    scene_29dof.xml               # Scene with ground plane and lighting
    scene_23dof.xml
    meshes/                       # 64 STL mesh files
  scripts/
    run_sim.sh                    # Simulation launcher
    run_real.sh                   # Real robot launcher
    run_eval.sh                   # Headless evaluation launcher
    replay_log.py                 # Log replay CLI
    gantry_sim.py                 # Gantry hang test (sim & real)
    wrist_validation.py           # Wrist sinusoid tracking test
    mirror_real_robot.py          # Mirror real robot state in MuJoCo
  tests/                          # 393 tests
    conftest.py                   # Shared fixtures, ONNX model helpers
    test_config.py                # Constants, config loading, validation
    test_compat.py                # RecurrentThread, SDK patching
    test_gantry.py                # Gantry harness hanging test
    test_joint_mapper.py          # Joint ordering, reordering roundtrips
    test_observations.py          # Gravity projection, velocity transforms
    test_safety.py                # State machine, clamping, thread safety
    test_isaaclab_policy.py       # Load, inference, dimension validation
    test_beyondmimic_policy.py    # Geometry helpers, observation construction
    test_sim_robot.py             # Impedance control, sensor mapping, DDS
    test_controller.py            # Command building, key handling, lifecycle
    test_real_robot.py            # DDS command/state, watchdog, CRC
    test_logger.py                # HDF5/NPZ roundtrip, events, replay
    test_viewer.py                # GLFW key map, viewer/headless runners
    test_main.py                  # CLI parsing, config integration, wiring
    test_integration.py           # End-to-end headless pipeline tests
    test_scaffolding.py           # Testing utilities
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/unitree_launcher --cov-report=term-missing

# Skip slow tests
pytest tests/ -m "not slow"

# Run a specific phase
pytest tests/test_controller.py -v
```

## Architecture

```
                    +-----------+
  Keyboard ------->| Controller|-------> DataLogger
  (GLFW keys)      |  (50 Hz)  |         (HDF5/NPZ)
                    +-----+-----+
                          |
              +-----------+-----------+
              |                       |
        +-----+------+        +------+------+
        |   Policy   |        |   Safety    |
        | (ONNX inf) |        | (clamp/estop)|
        +-----+------+        +------+------+
              |                       |
              +-----------+-----------+
                          |
                    +-----+-----+
                    |   Robot   |
                    | (sim/real)|
                    +-----------+
                          |
              +-----------+-----------+
              |                       |
        MuJoCo (sim)           DDS (real)
```

### Threading Model

Three threads cooperate when the viewer is active:

| Thread | Role | Frequency |
|--------|------|-----------|
| **Main** | Drain key queue, `viewer.sync()` under lock | ~60 Hz |
| **Control** | Read state, policy inference, PD command, `mj_step()` under lock | 50 Hz |
| **Viewer** | GLFW rendering, fires key callback into queue | vsync |

A `threading.Lock` on `SimRobot` protects `mjData` — held only during `mj_step()` (control thread) and `viewer.sync()` (main thread). The key callback fires on the viewer thread and must never acquire the lock; instead it enqueues key names into a `queue.SimpleQueue`, which the main thread drains outside the lock to avoid cross-thread deadlocks.

For BeyondMimic policies, the joint ordering is automatically extracted from the ONNX model's embedded metadata at startup, ensuring correct mapping between the policy's joint order and the robot's native order.

## Safety

- **Joint position limits** -- Commands clamped to physical joint ranges
- **Joint velocity limits** -- Velocity targets clamped per motor specification
- **Torque limits** -- Torque commands clamped per motor rating
- **Orientation check** -- Detects if robot has fallen (gravity vector > 35 deg from vertical)
- **Watchdog (real robot)** -- E-stops if no state message received within 100ms
- **Exception handling** -- Any control loop exception triggers immediate E-stop
- **E-stop latching** -- E-stop state persists until explicitly cleared

All safety limits are sourced from the Unitree RL Lab motor specifications and MuJoCo model files.
