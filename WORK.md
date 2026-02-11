# WORK.md — Execution Guide for Engineers

> **DO NOT DELETE this file.** It accompanies SPEC.md, PLAN_METAL.md, and PLAN_DOCKER.md.

## Overview

You have two implementation plans — **PLAN_METAL.md** (native macOS / MuJoCo viewer) and **PLAN_DOCKER.md** (Docker / Viser browser UI). They share **~83% of source code** (20 of 24 `src/` files are identical). This document tells you how to build both without doing the work twice.

**Strategy: build the shared core through PLAN_METAL first, then add the Docker/Viser layer on top.**

---

## Why Metal First

1. **Faster feedback loop** — native execution, no container rebuild cycle. You see results in the MuJoCo viewer immediately.
2. **Environment validation** — Metal Phase 0 catches dependency problems (CycloneDDS, MuJoCo, SDK2) before any code is written. Docker hides these behind the container.
3. **Simpler integration surface** — keyboard input is easier to debug than a Viser web UI. Get the control loop right first, then wire up the fancier interface.

---

## Execution Order

### Pass 1 — Shared Core (via PLAN_METAL)

Follow PLAN_METAL.md for these phases. Everything you build here is reused by the Docker plan.

| Step | Metal Phase | Module(s) Built | Ref Lines | Notes |
|------|-------------|----------------|-----------|-------|
| 1 | **Phase 0** | Environment validation | M:230 | Metal-only. Confirms MuJoCo, CycloneDDS, SDK2 work on your machine. |
| 2 | **Phase 1** | Scaffolding, `pyproject.toml`, assets | M:376 | Creates the shared directory tree. Use Metal's `pyproject.toml` (has Python 3.10 ceiling). |
| 3 | **Phase 2** | `src/config.py`, dataclasses, YAML configs | M:530 | **Shared.** 6 tasks, 13+ tests. |
| 4 | **Phase 3** | `src/compat.py` (RecurrentThread shim) | M:808 | **Shared.** Can run in parallel with Phase 2. |
| 5 | **Phase 4** | `src/policy/joint_mapper.py`, `src/policy/observations.py` | M:958 | **Shared.** 40+ tests. |
| 6 | **Phase 5** | `src/control/safety.py` | M:1105 | **Shared.** 26+ tests incl. `clamp_command()`. Can parallelize with Phases 4, 6, 7. |
| 7 | **Phase 6** | `src/policy/isaaclab_policy.py`, `src/policy/beyondmimic_policy.py` | M:1189 | **Shared.** Depends on Phase 4 (joint mapper). |
| 8 | **Phase 7** | `src/robot/sim_robot.py`, DDS bridge | M:1318 | **Shared core.** Depends on Phases 2, 3. |
| 9 | **Phase 8** | `src/control/controller.py` | M:1467 | **Shared core logic.** Depends on Phases 5, 6, 7. See "Key Divergence Points" below. |
| 10 | **Phase 10** | `src/logging/logger.py`, `src/logging/replay.py` | M:1879 | **Shared.** Can be built any time after Phase 2. |
| 11 | **Phase 11** | `src/robot/real_robot.py` | M:1997 | **Shared.** Can parallelize with Phases 9, 10. |

**After Step 11, all shared modules are complete.** Run the full shared test suite (~200+ tests).

### Pass 2 — Metal-Specific Integration

Continue in PLAN_METAL.md for the platform-specific pieces.

| Step | Metal Phase | What You Build | Ref Lines |
|------|-------------|---------------|-----------|
| 12 | **Phase 9** | MuJoCo native viewer + GLFW keyboard callbacks | M:1728 |
| 13 | **Phase 12** | `src/main.py` (Metal variant), shell scripts | M:2094 |
| 14 | **Phase 13** | Integration tests, manual end-to-end validation | M:2314 |

**Metal build is now complete.**

### Pass 3 — Docker/Viser Layer

Switch to PLAN_DOCKER.md. You only need to build the **divergent pieces** — skip all shared phases.

| Step | Docker Phase | What You Build | Ref Lines | Notes |
|------|-------------|---------------|-----------|-------|
| 15 | **Phase 7** | `src/viz/server.py`, `src/viz/robot_viz.py`, `src/viz/ui.py` | D:1380 | The main new code. Viser browser UI. |
| 16 | **Phase 10** | `src/main.py` (Docker variant) | D:1749 | Different CLI structure and wiring than Metal. |
| 17 | **Phase 11** | `docker/Dockerfile`, `docker-compose.yml` | D:1917 | Container setup with EGL for headless rendering. |
| 18 | **Phase 12** | Integration tests (Docker-specific) | D:2005 | End-to-end through browser UI. |
| 19 | **Phase 13** | Documentation | D:2129 | README, inline docs. |

**Docker build is now complete.**

---

## Key Divergence Points

Three files exist in both plans but have different implementations. Design these carefully during Pass 1 so Pass 3 is clean.

### `src/control/controller.py`

| Concern | Metal (PLAN_METAL) | Docker (PLAN_DOCKER) |
|---------|--------------------|--------------------|
| User input | `handle_key()` with GLFW key codes (Space, WASD, QZ, E, R, N, P, C, X) | `handle_key()` called from Viser keyboard events; policy selection via UI dropdown |
| Status output | Periodic stdout print at 1 Hz (for headless monitoring) | Viser telemetry panel |
| Policy cycling | N/P keys navigate policy directory | UI dropdown in `src/viz/ui.py` |
| Sim stepping when stopped | Calls `robot.step()` even when STOPPED (keeps sim alive for viewer) | Skips `robot.step()` when not RUNNING |
| Headless termination | `_max_steps` / `_max_duration` auto-stop | Not applicable (always has Viser) |

**Recommendation:** Build `controller.py` with the Metal `handle_key()` interface first. The core control loop, PD law, safety integration, and policy reload logic are identical. In Pass 3, the Docker variant only needs to swap the input dispatch and add Viser-specific wiring.

### `src/main.py`

| Concern | Metal | Docker |
|---------|-------|--------|
| Viewer | `mujoco.viewer.launch_passive()` | Viser server (runs as web server) |
| CLI args | `--headless`, `--duration`, `--steps` | `sim`/`real` subparsers, `--no-viser`, `--viser-port` |
| Event loop | GLFW viewer loop or headless tick loop | `while True` blocking on Ctrl+C |

**Recommendation:** These are essentially two different files that happen to share a name. Build Metal's `main.py` in Pass 2 (Step 13), then write Docker's `main.py` from scratch in Pass 3 (Step 16). Don't try to make one file serve both.

### `src/robot/sim_robot.py`

The core simulation logic (MuJoCo stepping, impedance control, DDS bridge, sensor reading) is identical. The Metal plan adds MuJoCo viewer integration hooks (exposing `model`/`data` for the passive viewer).

**Recommendation:** Build the shared core in Pass 1 (Step 8). Add viewer accessors as simple properties — the Docker plan doesn't use them but they're harmless.

---

## Parallelization Opportunities

If you have multiple engineers, these groups are independent after Phase 2 (config/dataclasses) completes:

```
                    ┌─ Phase 4: Joint Mapping ──┐
                    │                            ├─ Phase 6: Policies ─┐
Phase 0 → 1 → 2 ──┤─ Phase 5: Safety           │                     ├─ Phase 8: Control Loop
                    │                            │                     │
                    ├─ Phase 7: MuJoCo Sim ──────┘                     │
                    │                                                   │
                    ├─ Phase 10: Logging (independent) ─────────────────┤
                    │                                                   │
                    └─ Phase 3: Compat ─────────────────────────────────┘

After Phase 8:
  Phase 9 (Viewer) ─────┐
  Phase 11 (Real Robot) ─┼─ Phase 12 (CLI) → Phase 13 (Integration)
  Phase 10 (Logging) ────┘
```

**Critical path:** Phase 0 → 1 → 2 → 4 → 6 → 7 → 8 → 9 → 12 → 13

---

## File Inventory

### Shared Source (build once, used by both plans)

```
src/__init__.py
src/config.py                    # Phase 2
src/compat.py                    # Phase 3
src/robot/__init__.py
src/robot/base.py                # Phase 2 (dataclasses + abstract interface)
src/robot/sim_robot.py           # Phase 7 (minor Metal viewer additions)
src/robot/real_robot.py          # Phase 11
src/policy/__init__.py
src/policy/base.py               # Phase 2 (abstract interface)
src/policy/joint_mapper.py       # Phase 4
src/policy/observations.py       # Phase 4
src/policy/isaaclab_policy.py    # Phase 6
src/policy/beyondmimic_policy.py # Phase 6
src/control/__init__.py
src/control/controller.py        # Phase 8 (diverges in input handling)
src/control/safety.py            # Phase 5
src/logging/__init__.py
src/logging/logger.py            # Phase 10
src/logging/replay.py            # Phase 10
```

### Metal-Only Files

```
.python-version                  # Phase 1 (contains "3.10")
scripts/validate_macos.py        # Phase 0
scripts/run_eval.sh              # Phase 12 (headless batch eval)
src/main.py                      # Phase 12 (Metal variant)
tests/test_viewer.py             # Phase 9 (GLFW key handling tests)
```

### Docker-Only Files

```
docker/Dockerfile                # Phase 11
docker/docker-compose.yml        # Phase 11
src/viz/__init__.py              # Phase 7
src/viz/server.py                # Phase 7
src/viz/robot_viz.py             # Phase 7
src/viz/ui.py                    # Phase 7
src/main.py                      # Phase 10 (Docker variant)
requirements.txt                 # Phase 0 (mirrors pyproject.toml for Docker)
tests/test_viz.py                # Phase 7 (Viser UI tests)
```

### Shared Non-Source Files

```
configs/default.yaml             # Phase 2
configs/g1_29dof.yaml            # Phase 2
configs/g1_23dof.yaml            # Phase 2
assets/robots/g1/g1_29dof.xml   # Phase 1
assets/robots/g1/g1_23dof.xml   # Phase 1
assets/robots/g1/scene_29dof.xml # Phase 7
assets/robots/g1/scene_23dof.xml # Phase 7
assets/robots/g1/meshes/*.stl   # Phase 1
scripts/run_sim.sh               # Phase 12
scripts/run_real.sh              # Phase 12
scripts/replay_log.py            # Phase 10
tests/conftest.py                # Phase 1
tests/test_config.py             # Phase 2
tests/test_joint_mapper.py       # Phase 4
tests/test_observations.py       # Phase 4
tests/test_safety.py             # Phase 5
tests/test_isaaclab_policy.py    # Phase 6
tests/test_beyondmimic_policy.py # Phase 6
tests/test_logger.py             # Phase 10
tests/test_sim_robot.py          # Phase 7
tests/test_controller.py         # Phase 8
tests/test_compat.py             # Phase 3
tests/test_real_robot.py         # Phase 11
tests/test_main.py               # Phase 12 (different per plan)
tests/test_integration.py        # Phase 13
pyproject.toml                   # Phase 1 (minor deps differ per plan)
```

---

## Config Schema Note

The YAML config schema differs slightly between plans:

- **Metal:** No `viser:` section. Includes `headless:` options.
- **Docker:** Includes `viser:` section (`port`, `share`). No headless options.

Build the shared config schema in Phase 2. Add plan-specific sections when you reach the divergent phases. The config loader should ignore unknown keys so both variants coexist cleanly.

---

## Testing Checkpoints

Run the test suite at these milestones to catch regressions early:

| After Step | Expected Tests | What's Covered |
|-----------|---------------|----------------|
| Step 3 (Phase 2) | ~35 | Config, constants, dataclasses |
| Step 7 (Phase 6) | ~120 | + joint mapping, observations, safety, policies |
| Step 9 (Phase 8) | ~170 | + sim robot, control loop |
| Step 11 (Phase 11) | ~200 | + logging, real robot |
| Step 14 (Phase 13) | ~245 | + viewer, CLI, integration (full Metal suite) |
| Step 18 (Phase 12) | ~245 | Full Docker suite (replaces viewer tests with viz tests) |

Both plans target **~233–272 total tests** and **>80% coverage**.

---

## Safety-Critical Code — Test Values, Not Just Behavior

The test appendix in both plans (Appendix A) calls out specific formulas that require **value-level verification** — not just "it runs without crashing" but "given these exact inputs, the output matches this exact number." These are:

1. **Impedance control law** in `SimRobot.send_command()` — PD torque computation
2. **IsaacLab command building** — `target_pos = q_home + Ka * action`
3. **BeyondMimic command building** — `target_pos = target_q + Ka * action`, `dq_target = target_dq`
4. **Projected gravity** — quaternion rotation math
5. **Body-relative transforms** — used in BeyondMimic observations
6. **Safety limit clamping** — `clamp_command()` must enforce joint position, velocity, and torque limits

Write tests with hand-computed expected values for these. If the math is wrong, the robot could damage itself.

---

## Quick Reference

| Question | Answer |
|----------|--------|
| Which plan do I start with? | PLAN_METAL.md |
| When do I switch to PLAN_DOCKER? | After Metal Phase 13 (integration tests pass) |
| What's the Python version? | 3.10 (CycloneDDS ARM64 wheel ceiling) |
| macOS or Linux? | macOS for sim/dev; Linux for sim + real robot |
| What's the critical path? | Phases 0→1→2→4→6→7→8→9→12→13 |
| Where do the plans diverge? | `src/viz/`, `src/main.py`, `src/control/controller.py`, `docker/` |
| What's the test target? | ~245 tests, >80% coverage, value-level checks on safety math |
