# State Estimator Tuning Guide

This document covers how to bring up and tune the InEKF state estimator on the real G1 robot. Follow the sections in order.

## Prerequisites

- Robot on gantry with Ethernet connected
- `--estimator-verbose` flag added to your run command for diagnostic output
- Terminal visible for log output

The state estimator is **always enabled in real mode** — no `--estimator` flag needed. The `--estimator` flag exists only for sim mode (to test estimator-in-the-loop against MuJoCo ground truth). On real hardware there is no ground truth; the estimator provides `base_position` and `base_velocity` to the policy.

By default, raw IMU quaternion and angular velocity are passed through unmodified — policies trained with raw IMU in Isaac Lab destabilize when fed filtered IMU values (confirmed with proj_g_angvel policy). Add `--estimate-imu` to also estimate smoothed orientation and bias-corrected angular velocity, but only for policies explicitly trained with filtered IMU inputs.

All tunable parameters live in `src/unitree_launcher/estimation/state_estimator.py` unless noted otherwise. Defaults are in the `__init__` method and the `_DEFAULT_NOISE` dict at the top of the file.

---

## 1. Verify IMU Convention

**What to check:** The Unitree SDK provides IMU quaternion as wxyz. Our code assumes this. If wrong, orientation will be inverted and the robot will immediately diverge.

**How to test:**
```bash
# Stand robot upright on gantry, run:
uv run real --interface enp3s0 --estimator-verbose --no-log
```

Look at the first `Accel bias calibrated` log line. The accel bias should be small — each component under ±2 m/s². If you see values like 15-20 m/s², the quaternion convention is wrong (gravity is being projected into the wrong frame).

**Fix:** If the quaternion is xyzw instead of wxyz, swap the ordering in `real_robot.py` where `imu_quaternion` is assigned.

---

## 2. Verify Gyro Bias Calibration

**What to check:** During the 0.5s warmup, the estimator averages raw gyro readings to estimate bias. The robot must be stationary during this window.

**How to test:** Look for the `Gyro bias calibrated` log line. Healthy values are under ±0.03 rad/s per axis. If much larger, either:
- The robot was moving during startup (wait for it to settle)
- The gyro has a large offset (normal for some IMUs — the EKF will track it)

**What to tune:**
- `_WARMUP_TICKS` (default 25 = 0.5s): increase to 50 (1s) if the robot takes longer to settle after powering on. Longer warmup = better bias estimate but slower startup.

---

## 3. Verify Contact Detection

This is the most critical subsystem. Bad contact detection causes velocity spikes and position drift.

**Parameters** (in `estimation/contact.py`, `ContactDetector.__init__`):

| Parameter | Default | What it does |
|---|---|---|
| `high_threshold` | 40 N | GRF above this → foot is in contact |
| `low_threshold` | 15 N | GRF below this → foot has lifted |
| `high_time` | 0.02 s | Debounce for contact onset |
| `low_time` | 0.04 s | Debounce for contact break |
| `lever_arm` | 0.025 m | Ankle-to-ground distance for torque→force conversion |

**How to test:**
1. Stand the robot on flat ground (off gantry). Run with `--estimator-verbose`.
2. Watch the periodic log: `contacts=L1/R1`. Both should read 1 while standing.
3. Manually lift one foot (or have a helper tilt the robot). The lifted foot should switch to 0 within ~50ms.
4. Set it back down. Should return to 1 within ~30ms.

**What to look for:**
- **Flickering (rapid 0/1/0/1):** `high_threshold` too low or `low_threshold` too close to `high_threshold`. Increase separation (e.g., high=60, low=20).
- **Slow detection:** Increase `high_time`/`low_time` debounce is too long, or thresholds are too far apart.
- **Never detects contact:** Torque readings may be in different units. Print raw `abs(joint_torques[4]) / 0.025` and check the range. Adjust `lever_arm` if the foot geometry differs.
- **Contact detected in the air:** `low_threshold` too low. The ankle pitch torque from gravity/inertia alone exceeds the threshold. Raise it.

**Tuning procedure:**
1. Stand still → both feet L1/R1, stable
2. Walk in place slowly → contacts alternate cleanly, no flickering
3. Walk forward → contacts track gait cycle (one foot 0 during swing phase)

If contacts flicker during walking, increase `low_time` to 0.06-0.08s first. This is the least disruptive change.

---

## 4. Verify Velocity Estimation

The velocity output comes from leg kinematics (Jacobian × joint velocity), not IMU integration. It's accurate when the contact foot is truly stationary, but spiky during foot strike/liftoff.

**Parameters** (in `state_estimator.py`, `__init__`):

| Parameter | Default | What it does |
|---|---|---|
| `_leg_vel_alpha` | 0.3 | EMA smoothing (lower = smoother, more lag) |
| `_leg_vel_outlier` | 1.0 m/s | Reject samples deviating more than this from current estimate |

**How to test:**
1. Walk the robot forward at normal speed with `--estimator-verbose`.
2. Watch `|vel|` in the periodic log. Should be 0.0-0.1 when standing, 0.3-0.8 when walking.
3. If `|vel|` shows sudden spikes to 2-5 m/s, the outlier rejection isn't catching all bad samples.

**Tuning procedure:**
- **Velocity too spiky:** Lower `_leg_vel_alpha` to 0.2 or 0.15. Trade-off: more smoothing = more latency (~1-2 ticks at 50 Hz).
- **Velocity lags behind actual motion:** Raise `_leg_vel_alpha` toward 0.4-0.5. Only do this if contacts are clean.
- **Spurious spikes still getting through:** Raise `_leg_vel_outlier` to 1.5 m/s if legitimate walking accelerations are being rejected. Lower to 0.5 m/s if only small motions are expected.

---

## 5. Verify Position Estimation

Position is maintained by the EKF, corrected by foot contact kinematics. The main failure mode is drift.

**How to test:**
1. Walk the robot in a loop (forward 2m, turn, return to start).
2. Watch `pos=[x,y,z]` in the periodic log.
3. After returning to start, position should be within 0.1-0.2m of [0,0,0.79].

**What to look for:**
- **Large z-drift (height):** FK foot positions don't match real foot positions. Measure the real ankle-to-ground distance and update `lever_arm` in contact.py and the foot offset in `kinematics.py`.
- **Large xy-drift:** Contact transitions are noisy (see section 3). Each false contact break/make cycle introduces ~1-5mm of position error.
- **Position jumps:** The EKF covariance grew too large between corrections (no contacts detected for multiple ticks). Reduce `accel_noise` to tighten the prediction, or fix contact detection.

---

## 6. EKF Noise Parameters

These rarely need changing. Only adjust if the above steps don't resolve issues.

**Parameters** (in `state_estimator.py`, `_DEFAULT_NOISE` dict):

| Parameter | Default | What it controls |
|---|---|---|
| `gyro_noise` | 0.01 | Trust in gyro for rotation prediction |
| `accel_noise` | 1.0 | Trust in accelerometer for velocity/position prediction. Intentionally loose — lets FK corrections dominate. |
| `gyro_bias_noise` | 0.001 | How fast gyro bias is allowed to change |
| `accel_bias_noise` | 0.1 | How fast accel bias is allowed to change |
| `contact_noise` | 0.05 | FK measurement uncertainty (meters). Represents combined error from joint encoder noise + kinematic model mismatch. |

**When to change:**
- **Position drifts even with good contacts:** Reduce `contact_noise` (to 0.02-0.03) so the EKF trusts FK more. Only if FK accuracy has been verified.
- **Velocity oscillates at correction boundaries:** Increase `accel_noise` further (to 2.0-5.0). This makes the EKF rely more on corrections and less on IMU integration.
- **Heading slowly rotates:** Reduce `gyro_bias_noise` so the EKF adapts gyro bias faster. Or increase warmup time for a better initial calibration.

---

## 7. Sanity Bounds

The estimator falls back to defaults if the estimate exceeds these bounds.

| Parameter | Default | Location |
|---|---|---|
| `_MAX_VELOCITY` | 3.0 m/s | `state_estimator.py` top of file |
| `_MAX_POS_DELTA` | 2.0 m | `state_estimator.py` top of file |

If the robot is on a treadmill or in a large arena, increase `_MAX_POS_DELTA`. If the robot should never exceed 1 m/s, tighten `_MAX_VELOCITY` to 2.0.

When the fallback triggers, the policy sees a discontinuity (position jumps to initial, velocity goes to zero). This is a last resort. If it triggers during normal operation, fix the upstream issue (contacts, noise params) rather than widening the bounds.

---

## Quick Reference: First Hardware Run

```bash
# 1. Gantry, standing still, verbose logging
uv run real --interface enp3s0 --estimator-verbose --no-log

# Check logs for:
#   - Gyro bias < ±0.03 rad/s
#   - Accel bias < ±2 m/s²
#   - contacts=L1/R1 (both feet detected)

# 2. Walk in place (manually move legs while on gantry)
#   - Contacts should alternate cleanly
#   - |vel| should stay < 0.5

# 3. Free walk (off gantry, with policy)
uv run real --interface enp3s0 --estimator-verbose \
    --policy assets/policies/beyondmimic_29dof.onnx

# Watch for:
#   - Smooth velocity (no spikes > 2 m/s)
#   - Position tracks forward motion
#   - No fallback triggers (position doesn't jump to [0,0,0.79])
```

---

## Parameter Summary Table

| Parameter | File | Default | Adjust if... |
|---|---|---|---|
| `_WARMUP_TICKS` | state_estimator.py | 25 (0.5s) | Robot doesn't settle in 0.5s → increase |
| `_BLEND_TICKS` | state_estimator.py | 25 (0.5s) | Output discontinuity at blend end → increase |
| `high_threshold` | contact.py | 40 N | False contacts in air → raise |
| `low_threshold` | contact.py | 15 N | Contacts flicker → lower gap or raise |
| `high_time` | contact.py | 0.02 s | Contact onset too sensitive → increase |
| `low_time` | contact.py | 0.04 s | Contact break flickers → increase |
| `lever_arm` | contact.py | 0.025 m | GRF values wrong scale → measure real foot offset |
| `_leg_vel_alpha` | state_estimator.py | 0.3 | Velocity spiky → lower; laggy → raise |
| `_leg_vel_outlier` | state_estimator.py | 1.0 m/s | False rejections → raise; spikes leaking → lower |
| `accel_noise` | state_estimator.py | 1.0 | Position drifts a lot → lower; oscillates → raise |
| `contact_noise` | state_estimator.py | 0.05 m | FK trusted + verified → lower to 0.02 |
| `_MAX_VELOCITY` | state_estimator.py | 3.0 m/s | False fallbacks during fast walking → raise |
| `_MAX_POS_DELTA` | state_estimator.py | 2.0 m | Robot walks far from start → raise |
