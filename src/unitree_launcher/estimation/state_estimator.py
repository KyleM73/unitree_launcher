"""High-level state estimator facade.

Wires together the InEKF, ContactDetector, and G1Kinematics into a
single ``update()`` / ``populate_robot_state()`` interface that the
control loop can call each tick.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from unitree_launcher.estimation.contact import ContactDetector
from unitree_launcher.estimation.inekf import RightInvariantEKF
from unitree_launcher.estimation.kinematics import G1Kinematics
from unitree_launcher.estimation.lie_group import (
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
)
from unitree_launcher.robot.base import RobotState


# Default noise parameters (tuned for G1 at 50 Hz).
# accel_noise is high because the accelerometer sees real dynamic forces
# (PD settling, ground impacts) that look like noise from the filter's
# perspective.  This prevents velocity overconfidence and lets FK
# corrections dominate position accuracy.
_DEFAULT_NOISE = {
    "gyro_noise": 0.01,
    "accel_noise": 1.0,
    "gyro_bias_noise": 0.001,
    "accel_bias_noise": 0.1,
    "contact_noise": 0.05,
}

# During warmup the robot settles on the ground and the accelerometer
# transitions from free-fall (~0) to specific-force (~[0,0,9.81]).
# The InEKF is NOT run during warmup because the transient IMU data
# causes velocity to diverge (the FK correction Jacobian has no
# velocity term, so corrections cannot fix velocity errors).
#
# At the END of warmup (= tick _WARMUP_TICKS) the EKF is freshly
# initialized from current sensor readings and immediately starts
# producing output — no gap between init and first use.
_WARMUP_TICKS = 25  # 0.5 s at 50 Hz

# Number of ticks over which to blend from raw state → estimator
# output after warmup.  This avoids a hard discontinuity when the
# estimator kicks in.
_BLEND_TICKS = 25  # 0.5 s linear ramp

# Sanity bounds: if the estimated state exceeds these, fall back to
# the raw robot state to prevent NaN/Inf from reaching the policy.
_MAX_VELOCITY = 5.0    # m/s
_MAX_POS_DELTA = 1.0   # m from initial position


class StateEstimator:
    """Facade that wires InEKF + contact detection + FK.

    Usage in the control loop::

        estimator = StateEstimator(config)
        # each tick:
        estimator.update(robot_state)
        state = estimator.populate_robot_state(robot_state)

    Args:
        config: Full Config object (uses control.policy_frequency).
        noise_params: Override noise parameters for the InEKF.
    """

    def __init__(self, config: Any, noise_params: Optional[Dict[str, float]] = None):
        self._noise_params = {**_DEFAULT_NOISE, **(noise_params or {})}
        self._ekf = RightInvariantEKF(**self._noise_params)
        self._contacts = ContactDetector()
        self._fk = G1Kinematics()

        self._dt = 1.0 / config.control.policy_frequency
        self._initialized = False
        self._tick = 0
        self._p0 = np.zeros(3)  # initial position for sanity check

        # Track contact IDs in the filter
        self._left_contact_id: Optional[int] = None
        self._right_contact_id: Optional[int] = None
        self._prev_left: bool = False
        self._prev_right: bool = False
        self._leg_velocity: np.ndarray = np.zeros(3)  # smoothed leg-derived velocity
        self._leg_vel_alpha: float = 0.3  # EMA smoothing factor (0..1, lower = smoother)
        self._leg_vel_outlier: float = 1.0  # reject samples deviating more than this (m/s)

    def _init_ekf(self, robot_state: RobotState) -> None:
        """(Re-)initialize the EKF and augment both feet as contacts.

        Calibrates the initial accelerometer bias from the known-standing
        condition: the expected specific force is ``R0^T @ [0,0,9.81]``,
        so any deviation in the actual reading is bias.
        """
        R0 = quat_to_rotation_matrix(robot_state.imu_quaternion)
        if not np.all(np.isfinite(R0)):
            R0 = np.eye(3)

        p0 = robot_state.base_position.copy()
        v0 = robot_state.base_velocity.copy()

        if np.any(np.isnan(p0)):
            p0 = np.array([0.0, 0.0, 0.793])
        if np.any(np.isnan(v0)):
            v0 = np.zeros(3)

        # Calibrate accelerometer bias from known-standing condition.
        # Expected specific force when upright and stationary: R^T @ g_up
        # where g_up = [0, 0, 9.81] (reaction to gravity in world frame).
        expected_accel = R0.T @ np.array([0.0, 0.0, 9.81])
        actual_accel = robot_state.imu_linear_acceleration
        ba0 = actual_accel - expected_accel
        if not np.all(np.isfinite(ba0)):
            ba0 = np.zeros(3)

        self._p0 = p0.copy()
        self._ekf.initialize(R0, v0, p0, ba0=ba0)

        q = robot_state.joint_positions
        left_body = self._fk.left_foot_position(q)
        right_body = self._fk.right_foot_position(q)

        self._left_contact_id = self._ekf.augment_state(left_body)
        self._right_contact_id = self._ekf.augment_state(right_body)
        self._prev_left = True
        self._prev_right = True

    def initialize(self, robot_state: RobotState) -> None:
        """Initialize from first robot state reading."""
        self._init_ekf(robot_state)
        self._contacts.reset(both_in_contact=True)
        self._tick = 0
        self._initialized = True

    def update(self, robot_state: RobotState, dt: Optional[float] = None) -> None:
        """Run one estimator tick.

        Lifecycle:
            tick 0             — initialize (first call)
            tick 1..WARMUP-1   — warmup: only update contact detector
                                 (EKF skipped — transient IMU would
                                 destroy velocity estimates)
            tick WARMUP        — re-init EKF from settled sensor data,
                                 then immediately run first predict+correct
            tick WARMUP+1..    — normal InEKF: predict + detect + correct
            tick WARMUP..+BLEND — output blended with raw state (linear ramp)
            tick WARMUP+BLEND+ — full estimator output

        Args:
            robot_state: Current sensor readings.
            dt: Override timestep (default: 1/policy_frequency).
        """
        if not self._initialized:
            self.initialize(robot_state)
            return

        dt = dt or self._dt
        self._tick += 1

        # --- Warmup phase: let the robot settle, keep contact detector warm ---
        if self._tick < _WARMUP_TICKS:
            self._contacts.update(robot_state.joint_torques, dt)
            return

        # --- Re-initialize exactly at warmup end, then fall through to normal ---
        if self._tick == _WARMUP_TICKS:
            self._init_ekf(robot_state)
            self._contacts.reset(both_in_contact=True)
            # Fall through: run the first predict+correct immediately so
            # the EKF has one tick of real output before populate uses it.

        # --- Normal operation ---

        # 1. IMU prediction
        self._ekf.predict(
            robot_state.imu_angular_velocity,
            robot_state.imu_linear_acceleration,
            dt,
        )

        q = robot_state.joint_positions

        # 2. Contact detection
        left_contact, right_contact = self._contacts.update(
            robot_state.joint_torques, dt
        )

        # 3. Handle contact transitions
        if left_contact and not self._prev_left:
            left_body = self._fk.left_foot_position(q)
            self._left_contact_id = self._ekf.augment_state(left_body)
        elif not left_contact and self._prev_left:
            if self._left_contact_id is not None:
                self._ekf.marginalize_contact(self._left_contact_id)
                self._left_contact_id = None

        if right_contact and not self._prev_right:
            right_body = self._fk.right_foot_position(q)
            self._right_contact_id = self._ekf.augment_state(right_body)
        elif not right_contact and self._prev_right:
            if self._right_contact_id is not None:
                self._ekf.marginalize_contact(self._right_contact_id)
                self._right_contact_id = None

        self._prev_left = left_contact
        self._prev_right = right_contact

        # 4. Kinematic corrections for active contacts
        if left_contact and self._left_contact_id is not None:
            left_body = self._fk.left_foot_position(q)
            self._ekf.correct_kinematics(self._left_contact_id, left_body)

        if right_contact and self._right_contact_id is not None:
            right_body = self._fk.right_foot_position(q)
            self._ekf.correct_kinematics(self._right_contact_id, right_body)

        # 5. Contact-foot velocity measurement.
        #    When a foot is in contact it should be stationary, so the
        #    pelvis velocity can be inferred from leg joint velocities:
        #        v_pelvis = -R @ (J_foot @ dq_leg + omega × p_foot)
        #    This is joint-encoder accurate (~0.5 mm/s error) and is used
        #    both as an EKF correction and as the direct velocity output
        #    (the EKF velocity state accumulates IMU integration drift
        #    between corrections, so the direct measurement is more accurate).
        R = self._ekf.rotation
        dq = robot_state.joint_velocities
        omega = robot_state.imu_angular_velocity
        v_samples = []

        if left_contact and self._left_contact_id is not None:
            J = self._fk.left_foot_jacobian(q)
            dq_leg = dq[self._fk.LEFT_LEG_INDICES]
            p_foot = self._fk.left_foot_position(q)
            v_from_leg = -R @ (J @ dq_leg + np.cross(omega, p_foot))
            # Reject if measurement deviates too far from current estimate
            # (likely a foot that's lifting but still marked in-contact).
            if np.linalg.norm(v_from_leg - self._leg_velocity) < self._leg_vel_outlier:
                self._ekf.correct_velocity(v_from_leg, noise_std=0.01)
                v_samples.append(v_from_leg)

        if right_contact and self._right_contact_id is not None:
            J = self._fk.right_foot_jacobian(q)
            dq_leg = dq[self._fk.RIGHT_LEG_INDICES]
            p_foot = self._fk.right_foot_position(q)
            v_from_leg = -R @ (J @ dq_leg + np.cross(omega, p_foot))
            if np.linalg.norm(v_from_leg - self._leg_velocity) < self._leg_vel_outlier:
                self._ekf.correct_velocity(v_from_leg, noise_std=0.01)
                v_samples.append(v_from_leg)

        if v_samples:
            v_raw = np.mean(v_samples, axis=0)
            # Exponential moving average to smooth per-tick spikes from
            # joint velocity noise at foot strike/liftoff.
            a = self._leg_vel_alpha
            self._leg_velocity = a * v_raw + (1.0 - a) * self._leg_velocity
        # else: keep previous _leg_velocity (brief single-support phase)

    @property
    def settled(self) -> bool:
        """True once warmup is complete and EKF is running."""
        return self._tick >= _WARMUP_TICKS

    def _blend_alpha(self) -> float:
        """Blending weight: 0 = raw state, 1 = full estimator output."""
        if self._tick < _WARMUP_TICKS:
            return 0.0
        ticks_since = self._tick - _WARMUP_TICKS
        if ticks_since >= _BLEND_TICKS:
            return 1.0
        return ticks_since / _BLEND_TICKS

    def _estimate_is_usable(self) -> bool:
        """Check if current estimate is finite and within sanity bounds."""
        if not self._ekf.initialized:
            return False
        if not self.settled:
            return False
        pos = self._ekf.position
        vel = self._ekf.velocity
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            return False
        if np.linalg.norm(vel) > _MAX_VELOCITY:
            return False
        if np.linalg.norm(pos - self._p0) > _MAX_POS_DELTA:
            return False
        return True

    def populate_robot_state(self, robot_state: RobotState) -> RobotState:
        """Fill base_position and base_velocity from estimator.

        During warmup, returns the original state unmodified (or fills
        NaN fields with defaults for the real robot).

        After warmup, blends from raw → estimator over ``_BLEND_TICKS``
        to avoid a hard discontinuity.  Falls back to raw state if the
        estimate is NaN, Inf, or exceeds sanity bounds.

        Returns a modified copy of the input state.
        """
        state = robot_state.copy()

        raw_pos_nan = np.any(np.isnan(robot_state.base_position))
        raw_vel_nan = np.any(np.isnan(robot_state.base_velocity))

        if not self._initialized or not self._estimate_is_usable():
            # Not ready — fill NaN fields with defaults, pass non-NaN through
            if raw_pos_nan:
                state.base_position = self._p0.copy()
            if raw_vel_nan:
                state.base_velocity = np.zeros(3)
            return state

        alpha = self._blend_alpha()
        if alpha <= 0.0:
            if raw_pos_nan:
                state.base_position = self._p0.copy()
            if raw_vel_nan:
                state.base_velocity = np.zeros(3)
            return state

        ekf_pos = self._ekf.position

        # For velocity, use the leg-derived measurement directly rather
        # than the EKF velocity state.  The EKF velocity accumulates IMU
        # integration errors between corrections, but the leg Jacobian
        # measurement is joint-encoder accurate (~0.5 mm/s error).
        ekf_vel = self._leg_velocity

        # Blend raw → estimator (raw = ground truth in sim, defaults if NaN)
        raw_pos = robot_state.base_position if not raw_pos_nan else self._p0
        raw_vel = robot_state.base_velocity if not raw_vel_nan else np.zeros(3)

        state.base_position = (1.0 - alpha) * raw_pos + alpha * ekf_pos
        state.base_velocity = (1.0 - alpha) * raw_vel + alpha * ekf_vel

        # Smoothed orientation from fused EKF rotation (IMU + kinematics).
        ekf_quat = rotation_matrix_to_quat(self._ekf.rotation)
        raw_quat = robot_state.imu_quaternion
        state.imu_quaternion = (1.0 - alpha) * raw_quat + alpha * ekf_quat
        # Re-normalize after linear blend (SLERP not needed for small alpha steps).
        state.imu_quaternion /= np.linalg.norm(state.imu_quaternion)

        # Bias-corrected angular velocity.
        state.imu_angular_velocity = (
            robot_state.imu_angular_velocity - alpha * self._ekf.gyro_bias
        )

        return state

    @property
    def base_position(self) -> np.ndarray:
        return self._ekf.position

    @property
    def base_velocity(self) -> np.ndarray:
        return self._ekf.velocity

    @property
    def base_rotation(self) -> np.ndarray:
        return self._ekf.rotation

    @property
    def left_contact(self) -> bool:
        return self._contacts.left_contact

    @property
    def right_contact(self) -> bool:
        return self._contacts.right_contact

    @property
    def initialized(self) -> bool:
        return self._initialized
