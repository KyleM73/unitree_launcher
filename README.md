# Unitree G1 Deployment Stack

Control stack for the Unitree G1 humanoid robot. Supports MuJoCo simulation (macOS/Linux) and onboard real robot deployment (G1 Jetson Orin). Runs ONNX neural network policies at 50 Hz with safety enforcement.

## Modes

| Mode | Command | Description |
|------|---------|-------------|
| **sim** | `uv run sim` | MuJoCo simulation (GUI, Viser, or headless) |
| **eval** | `uv run eval` | Accurate evaluation (1000 Hz physics, headless) |
| **real** | `uv run real` | Onboard G1 deployment (C++ DDS backend) |
| **mirror** | `uv run mirror` | Read-only DDS visualization of the real robot |
| **replay** | `uv run replay` | Play back logged data (GUI, Viser, or summary/CSV) |

## Supported Policies

- **IsaacLab** — Velocity-tracking locomotion (arrow key / stick controls)
- **BeyondMimic** — Motion-tracking with trajectory playback and ONNX metadata gains

## Requirements

- Python 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- macOS (Apple Silicon) or Linux (x86_64 / aarch64)
- Real robot: Linux + C++ unitree_interface binding

## Quick Start

```bash
# Install
uv sync

# Simulation with MuJoCo GUI (macOS needs mjpython)
uv run sim --gui --policy assets/policies/stance_29dof.onnx

# Simulation with web viewer
uv run sim --viser --policy assets/policies/beyondmimic_29dof.onnx
# Open http://localhost:8080

# Headless evaluation
uv run eval --steps 500 --policy assets/policies/stance_29dof.onnx

# Replay logged data
uv run replay logs/run_name/ --gui
uv run replay logs/run_name/ --viser --speed 0.5 --loop

# Run tests
uv run pytest tests/ -x
```

## Gantry Arm Test (Sim2Real)

The `--gantry` flag runs a right shoulder pitch sinusoid while the robot hangs from a gantry. Used for sim2real comparison — the same test runs identically in sim and on real hardware.

```bash
# Sim with GUI viewer
uv run sim --gantry --gui --duration 40

# Sim with viser
uv run sim --gantry --viser --duration 40

# Real robot (on G1, after deploy)
uv run real --gantry -c configs/g1_deploy.yaml --duration 40

# Compare logged data
uv run python scripts/compare_sim2real.py logs/<sim_run>/ logs/<real_run>/
```

The test sequence:
1. **Prepare** (10-20s): Smooth blend from current pose to home position
2. **Sinusoid**: Right shoulder pitch sweeps through quarter ROM (negative direction only, 0.2 Hz)

All infrastructure is active: wireless E-stop (A button), gamepad, keyboard, data logging, video recording (`--record`).

## Real Robot Deployment

### Network

| Node | IP |
|------|----|
| Motor control board | `192.168.123.161` |
| G1 PC (SSH) | `192.168.123.164` (user: `unitree`, pass: `123`) |
| Dev machine | `192.168.123.100` (configure with `./scripts/setup_robot_network.sh`) |

### Deploy to Robot

```bash
# Sync code and run preflight checks (auto-installs uv if needed)
./scripts/deploy_to_robot.sh

# SSH in and build C++ backend (first time only)
ssh unitree@192.168.123.164
cd ~/unitree_launcher
./scripts/build_cpp_backend.sh

# Run
uv run real -c configs/g1_deploy.yaml --policy assets/policies/stance_29dof.onnx
```

### Wireless Controller (Real Robot)

| Button | Action |
|--------|--------|
| **A** | E-stop (software, checked every tick) |
| **L2+B** | Hardware damping mode (firmware-level, always works) |
| **B** | Stop policy (return to stance) |
| **X** | Start / re-activate policy |
| **Y** | Reset policy (stop + reset + re-activate) |
| **Start** | Start policy |
| **Select** | Stop policy |
| **Left stick** | Forward/back (Y) and strafe (X) |
| **Right stick X** | Yaw rate |
| **R1 + DPad Up/Down** | Next / previous policy |

### Prepare Phase

On real hardware, a 20-second prepare phase blends from the current pose to the default stance:
- 300-step linear ramp (6s) from current motor positions to home pose
- Policies warm up (ONNX session + observation history) during prepare
- Wireless A-button E-stop is active throughout
- At 90%, policy state is reset for clean activation

## Keyboard Controls (Simulation)

| Key | Action |
|-----|--------|
| `Space` | Toggle policy (start / stop) |
| `Backspace` | E-stop (latching) |
| `Enter` | Clear E-stop |
| `Delete` | Reset robot and policy |
| `Up` / `Down` | Forward / back velocity (±0.1) |
| `Left` / `Right` | Strafe velocity (±0.1) |
| `,` / `.` | Yaw rate (±0.1) |
| `/` | Zero all velocity |
| `=` / `-` | Next / previous policy |

## Web Viewer (Viser)

```bash
uv run sim --viser --policy path/to/policy.onnx
# Open http://localhost:8080
```

Sidebar controls: Start/Stop, E-stop, Reset, policy selector, velocity sliders, telemetry panel (Hz, inference time, height, step count).

## Configuration

YAML configs in `configs/`. CLI arguments override config values.

| Config | Use |
|--------|-----|
| `default.yaml` | Base config for sim |
| `g1_deploy.yaml` | Onboard real deployment |
| `g1_sim_bm.yaml` | BeyondMimic simulation |
| `g1_real_bm.yaml` | BeyondMimic on real robot |
| `g1_29dof.yaml` | 29-DOF variant defaults |
| `g1_23dof.yaml` | 23-DOF variant defaults |

Key settings:
```yaml
control:
  policy_frequency: 50    # Policy inference rate (Hz)
  sim_frequency: 500      # MuJoCo physics rate (Hz)
  kd_damp: 8.0            # Damping gain for safety/non-controlled joints
  transition_steps: 5     # Steps to interpolate to policy starting pose

safety:
  tilt_check: true        # E-stop on >57° tilt
  frame_drop_check: true  # E-stop on >200ms frame drop
```

## Policy Transitions

- **Activation**: Cosine-interpolates from current position to `default_pos` over `transition_steps` (default 5). Policy `warmup()` runs during transition (ONNX + obs history).
- **Return to stance**: Instant (stance policy needs full authority immediately).
- **BeyondMimic**: Holds at first reference frame for 5 steps before advancing. ONNX metadata `start_timestep` / `end_timestep` trims unstable trajectory edges.

## Docker

Build the image from the repo root:

```bash
docker build -f docker/Dockerfile -t unitree-launcher .
```

Run directly:

Policy files are gitignored and not baked into the image — mount them with `-v`:

```bash
# Headless simulation
docker run --rm -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher sim --policy assets/policies/stance_29dof.onnx --duration 10

# Headless evaluation (1000 Hz physics)
docker run --rm -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher eval --steps 500 --policy assets/policies/stance_29dof.onnx

# Viser web viewer (open http://localhost:8080)
docker run --rm -p 8080:8080 -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher sim --viser --play

# Mirror real robot via viser (Linux, host networking for DDS)
docker run --rm --network host unitree-launcher mirror --viser --interface eth0

# Real robot (Linux, host networking for DDS, C++ backend)
docker build -f docker/Dockerfile --build-arg BUILD_CPP_BACKEND=1 -t unitree-launcher-real .
docker run --rm --network host -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher-real real --policy assets/policies/stance_29dof.onnx

# X11 GUI (Linux only)
xhost +local:docker
docker run --rm -e DISPLAY=$DISPLAY -e MUJOCO_GL=glx \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v ./assets/policies:/app/assets/policies:ro \
    unitree-launcher sim --gui --policy assets/policies/stance_29dof.onnx
```

### Docker Compose Profiles

| Profile | Services | Description |
|---------|----------|-------------|
| `headless` | `sim-headless`, `eval` | EGL rendering, no display needed |
| `gui` | `sim-gui` | X11 forwarding (Linux only) |
| `viser` | `sim-viser`, `mirror` | Viser web viewer on port 8080 |
| `real` | `real-robot` | Host networking + C++ backend |
| `test` | `test` | pytest runner |

```bash
# Headless sim
docker compose -f docker/docker-compose.yml --profile headless run --rm sim-headless \
    sim --policy assets/policies/stance_29dof.onnx --duration 10

# Evaluation
docker compose -f docker/docker-compose.yml --profile headless run --rm eval \
    eval --steps 500 --policy assets/policies/stance_29dof.onnx

# Viser sim
docker compose -f docker/docker-compose.yml --profile viser run --rm sim-viser \
    sim --viser --policy assets/policies/stance_29dof.onnx

# Tests
docker compose -f docker/docker-compose.yml --profile test run --rm test
```

**Note:** X11 GUI forwarding requires Linux with an X server. macOS does not support X11 forwarding to Docker containers natively — use `--viser` instead.

## Project Structure

```
src/unitree_launcher/
  main.py                     # CLI entry point, viewer/headless runners
  config.py                   # Joint constants, dataclasses, YAML loading
  mirror.py                   # Mirror mode entry point (DDS → MuJoCo viewer)
  replay.py                   # Replay mode entry point (logged data → viewer)
  gantry.py                   # Elastic band + gantry simulation utilities
  trajectory.py               # Collision-aware IK trajectory planning
  recording.py                # MuJoCo video recording (MP4)
  compat.py                   # unitree_sdk2py patches, cross-platform helpers
  script_utils.py             # Shared helpers for diagnostic scripts
  control/
    runtime.py                # Step-based control loop, transitions, state machine
    safety.py                 # Safety controller, E-stop, command clamping
    gamepad.py                # Gamepad monitor (E-stop via USB HID)
  policy/
    base.py                   # Policy ABC, action smoothing, warmup()
    isaaclab_policy.py        # IsaacLab velocity-tracking policy
    beyondmimic_policy.py     # BeyondMimic motion-tracking policy
    hold_policy.py            # Static PD hold at home pose
    sinusoid_policy.py        # Joint sinusoid for gantry testing
    joint_mapper.py           # Robot ↔ policy joint ordering
    factory.py                # Policy loading, gain overrides, preloading
  robot/
    base.py                   # RobotState, RobotCommand, RobotInterface ABC
    sim_robot.py              # MuJoCo simulation backend
    real_robot.py             # C++ unitree_interface backend (onboard)
    mirror_robot.py           # Read-only Python DDS backend
  controller/
    input.py                  # InputManager (merges all controllers)
    keyboard.py               # Keyboard input (GLFW keys)
    wireless.py               # Unitree wireless gamepad (real robot)
    gamepad_input.py          # USB HID gamepad (sim/real)
    viser_input.py            # Viser web UI input
  estimation/
    state_estimator.py        # InEKF + contact detection + FK
    inekf.py                  # Invariant Extended Kalman Filter
    contact.py                # Contact detection (GRF thresholding)
    kinematics.py             # Leg forward kinematics (Jacobian)
    lie_group.py              # SO(3)/SE(3) Lie group operations
  datalog/
    logger.py                 # HDF5/NPZ time-series logging
    replay.py                 # Log loading, state reconstruction, CSV export
  viz/
    viser_viewer.py           # Web-based 3D viewer
    conversions.py            # MuJoCo geom → trimesh

configs/                      # YAML configuration presets
assets/robots/g1/             # MuJoCo XML models + meshes
assets/policies/              # ONNX policy files
scripts/                      # Shell helpers (deploy, build, network setup)
tests/                        # 505 automated tests
```

## Testing

```bash
uv run pytest tests/ -x              # All tests
uv run pytest tests/ -k transition   # Specific tests
uv run pytest tests/ -m "not slow"   # Skip slow tests
```

## State Estimator

An InEKF state estimator fuses IMU predictions with contact-foot kinematics (leg Jacobian).

- **Real mode**: Always on — the estimator is the only source of base state.
- **Sim mode**: Opt-in with `--estimator` to validate estimator-in-the-loop before hardware.
- **Tuning**: Add `--estimator-verbose` for diagnostic output. See [`docs/estimator_tuning.md`](docs/estimator_tuning.md).

Two estimation modes:

| Mode | Flag | Estimates | Default for |
|------|------|-----------|-------------|
| **pos+vel** | _(default)_ | `base_position`, `base_velocity` | Real and sim |
| **pos+vel+imu** | `--estimate-imu` | Above + smoothed `imu_quaternion`, bias-corrected `imu_angular_velocity` | Policies trained with filtered IMU |

Most policies are trained with raw IMU in Isaac Lab — use the default mode. Only add `--estimate-imu` for policies explicitly trained with filtered IMU inputs.

```bash
# Sim: test estimator against MuJoCo ground truth
uv run sim --estimator --policy assets/policies/beyondmimic_29dof.onnx

# Real: estimator is automatic, verbose for tuning
uv run real --estimator-verbose --policy assets/policies/stance_29dof.onnx

# Full estimation (pos+vel+imu) for policies that expect filtered IMU
uv run real --estimate-imu --policy assets/policies/filtered_imu_policy.onnx
```

## Safety

- **Joint limits**: Commands clamped to physical joint position/velocity/torque ranges
- **Tilt detection**: E-stop on >57° tilt from vertical (every tick)
- **Frame drop**: E-stop on >200ms control loop stall
- **Wireless E-stop**: A-button checked in `get_state()` (tightest Python loop)
- **Hardware fallback**: L2+B on wireless controller triggers firmware-level damping
- **Exception handling**: Any control loop exception triggers immediate E-stop
- **E-stop latching**: Persists until explicitly cleared
