# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Code Style

- **No historical notes in code**: Comments, docstrings, and variable names describe what the code currently is and does. Never reference what it used to be, what phase it was built in, what it was renamed from, or what was removed. That is what git history is for.
- **No backward-compatibility aliases**: One name for each thing. No duplicate properties, no shim methods, no "formerly known as" aliases. If something is renamed, rename it everywhere.
- **No legacy fallback code paths**: Code should fail clearly in failure modes, not silently fall back to deprecated behavior. No try/except blocks that swallow errors for backward compat.
- **No sed for file edits**: Use the Edit tool directly with explicit old_string/new_string. sed is error-prone (double replacements, wrong matches, whitespace issues).

## Architecture

- **Five modes**: `sim`, `eval`, `real`, `mirror`, `replay` via `uv run <mode>`
- **Three robot backends**: `SimRobot` (MuJoCo), `RealRobot` (C++ onboard), `MirrorRobot` (read-only DDS)
- **Runtime** (`control/runtime.py`): Step-based control. `step()` is the atomic unit (no sleep, no thread). `start_threaded()` for gui/viser/real modes.
- **Policies own everything**: Each policy implements `step(state, vel_cmd) -> RobotCommand` with its own observation building, control law, gains, and action smoothing.
- **JointMapper**: Maps between robot-native and policy joint orderings. Properties: `n_policy`, `policy_joints`, `policy_indices`, `n_robot`.
- **InputController** (`controller/`): All input sources implement `InputController` (keyboard, viser, gamepad, wireless). `InputManager` merges them — first non-zero velocity wins, union of commands. Wireless has highest priority on real.
- **Variable naming**: `runtime` (not `pipeline` or `controller`), `rt` in tests, `s.runtime` for integration test setup.
- **Policy transitions**: Runtime interpolates from current position to `default_pos` over `transition_steps` (default 5). Return to default/stance is always instant. BeyondMimic holds at start frame for `hold_steps` (5) before advancing. `warmup()` runs ONNX + obs history during transition without advancing trajectory counters.
- **BM trajectory trimming**: ONNX metadata `start_timestep` / `end_timestep` skip unstable edges. `stiffness`/`damping` properties return robot-native order (consistent with base class).

## Common Commands

```bash
uv run sim --gui --policy assets/policies/stance_29dof.onnx
uv run sim --viser --policy assets/policies/beyondmimic_29dof.onnx
uv run eval --steps 500 --policy assets/policies/stance_29dof.onnx
uv run replay logs/run_name/ --gui
uv run pytest tests/ -x
```

## Testing

- Tests use `step()` directly (no `time.sleep` polling)
- Mock policies return `RobotCommand` from `step()`
- Integration tests disable `tilt_check` for dummy zero-output policies
- Headless tests run at max speed (no sleep)
