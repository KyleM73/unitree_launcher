"""Right-Invariant Extended Kalman Filter on SE_k(3).

Implements contact-aided InEKF following Hartley et al. (IJRR 2020).
The state lives on an extended special Euclidean group SE_k(3) where k
is the number of active contacts. IMU biases are tracked in a separate
Euclidean vector (not part of the group).

State matrix X (5+k × 5+k):
    [[R, v, p, d_1, ..., d_k],
     [0, 1, 0,  0,  ...,  0 ],
     [0, 0, 1,  0,  ...,  0 ],
     [0, 0, 0,  1,  ...,  0 ],
     ...
     [0, 0, 0,  0,  ...,  1 ]]

where d_i = world-frame position of contact i.

Bias vector theta = [bg(3), ba(3)] (gyro bias, accel bias).

Covariance layout:
    [rot(3), vel(3), pos(3), contact_1(3), ..., contact_k(3), bg(3), ba(3)]
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from unitree_launcher.estimation.lie_group import (
    adjoint_se2_3,
    se2_3_exp,
    skew,
)

# Gravity in world frame (NED: [0,0,-g] or NWU: [0,0,-g])
_GRAVITY = np.array([0.0, 0.0, -9.81])


class RightInvariantEKF:
    """Right-invariant EKF for legged robot state estimation.

    Args:
        gyro_noise: Gyroscope noise std (rad/s).
        accel_noise: Accelerometer noise std (m/s^2).
        gyro_bias_noise: Gyroscope bias random walk std.
        accel_bias_noise: Accelerometer bias random walk std.
        contact_noise: Contact position measurement noise std (m).
    """

    def __init__(
        self,
        gyro_noise: float = 0.01,
        accel_noise: float = 0.1,
        gyro_bias_noise: float = 0.001,
        accel_bias_noise: float = 0.01,
        contact_noise: float = 0.02,
    ):
        self._gyro_noise = gyro_noise
        self._accel_noise = accel_noise
        self._gyro_bias_noise = gyro_bias_noise
        self._accel_bias_noise = accel_bias_noise
        self._contact_noise = contact_noise

        # State
        self._dim = 5  # base SE_2(3): R, v, p
        self._X: Optional[np.ndarray] = None  # (5+k, 5+k)
        self._theta = np.zeros(6)  # [bg(3), ba(3)]
        self._P: Optional[np.ndarray] = None  # covariance

        # Contact bookkeeping: maps contact_id -> column index in X
        self._contacts: dict[int, int] = {}
        self._next_contact_id = 0

    def initialize(
        self,
        R0: np.ndarray,
        v0: np.ndarray,
        p0: np.ndarray,
        bg0: Optional[np.ndarray] = None,
        ba0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """Set initial state.

        Args:
            R0: Initial rotation matrix (3x3), body->world.
            v0: Initial velocity (3,) in world frame.
            p0: Initial position (3,) in world frame.
            bg0: Initial gyro bias (3,). Defaults to zeros.
            ba0: Initial accel bias (3,). Defaults to zeros.
            P0: Initial covariance (15x15). Defaults to diagonal.
        """
        self._dim = 5
        self._contacts = {}
        self._next_contact_id = 0

        self._X = np.eye(5)
        self._X[0:3, 0:3] = R0
        self._X[0:3, 3] = v0
        self._X[0:3, 4] = p0

        self._theta = np.zeros(6)
        if bg0 is not None:
            self._theta[0:3] = bg0
        if ba0 is not None:
            self._theta[3:6] = ba0

        if P0 is not None:
            self._P = P0.copy()
        else:
            # Default: moderate uncertainty in position/velocity,
            # small in rotation, moderate in biases
            P_diag = np.array([
                0.01, 0.01, 0.01,    # rotation
                0.1, 0.1, 0.1,       # velocity
                0.01, 0.01, 0.01,    # position
                0.001, 0.001, 0.001, # gyro bias
                0.01, 0.01, 0.01,    # accel bias
            ])
            self._P = np.diag(P_diag)

    @property
    def initialized(self) -> bool:
        return self._X is not None

    @property
    def rotation(self) -> np.ndarray:
        """Current rotation estimate (3x3), body->world."""
        return self._X[0:3, 0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate (3,) in world frame."""
        return self._X[0:3, 3].copy()

    @property
    def position(self) -> np.ndarray:
        """Current position estimate (3,) in world frame."""
        return self._X[0:3, 4].copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        return self._theta[0:3].copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self._theta[3:6].copy()

    @property
    def n_contacts(self) -> int:
        return len(self._contacts)

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    # ------------------------------------------------------------------
    # Predict (IMU propagation)
    # ------------------------------------------------------------------

    def predict(self, omega: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """Propagate state using IMU measurements.

        Args:
            omega: Gyroscope measurement (3,) in body frame (rad/s).
            accel: Accelerometer measurement (3,) in body frame (m/s^2).
            dt: Time step (s).
        """
        R = self._X[0:3, 0:3]
        v = self._X[0:3, 3]
        p = self._X[0:3, 4]

        bg = self._theta[0:3]
        ba = self._theta[3:6]

        # Bias-corrected measurements
        omega_c = omega - bg
        accel_c = accel - ba

        # Strapdown integration (first-order)
        R_new = R @ _rodrigues_increment(omega_c * dt)
        v_new = v + (R @ accel_c + _GRAVITY) * dt
        p_new = p + v * dt + 0.5 * (R @ accel_c + _GRAVITY) * dt * dt

        # Guard: if integration produced NaN/Inf, skip this prediction
        if not (np.all(np.isfinite(R_new)) and np.all(np.isfinite(v_new))
                and np.all(np.isfinite(p_new))):
            return

        # Update state matrix
        self._X[0:3, 0:3] = R_new
        self._X[0:3, 3] = v_new
        self._X[0:3, 4] = p_new
        # Contact positions in X are static (world frame) — no update needed.

        # --- Covariance propagation ---
        n_c = len(self._contacts)
        state_dim = 9 + 3 * n_c + 6  # rot + vel + pos + contacts + biases

        # Discrete-time state transition (Φ)
        Phi = np.eye(state_dim)

        # Rotation block derivatives
        Phi[0:3, 0:3] = _rodrigues_increment(-omega_c * dt)  # rotation block

        # Velocity block
        Phi[3:6, 0:3] = -R @ skew(accel_c) * dt
        # Position block
        Phi[6:9, 0:3] = -0.5 * R @ skew(accel_c) * dt * dt
        Phi[6:9, 3:6] = np.eye(3) * dt

        # Bias coupling
        bias_start = 9 + 3 * n_c
        Phi[0:3, bias_start:bias_start+3] = -R * dt  # d(rot)/d(bg)
        Phi[3:6, bias_start+3:bias_start+6] = -R * dt  # d(vel)/d(ba)
        Phi[6:9, bias_start+3:bias_start+6] = -0.5 * R * dt * dt  # d(pos)/d(ba)

        # Process noise
        Q = np.zeros((state_dim, state_dim))
        Q[0:3, 0:3] = np.eye(3) * (self._gyro_noise * dt) ** 2
        Q[3:6, 3:6] = np.eye(3) * (self._accel_noise * dt) ** 2
        # Position process noise: use accel noise scaled by dt (not dt^2)
        # to prevent overconfidence that causes correction instability.
        Q[6:9, 6:9] = np.eye(3) * (self._accel_noise * dt) ** 2 * 0.1
        # Contact process noise (small — contacts should be stationary)
        for i in range(n_c):
            idx = 9 + 3 * i
            Q[idx:idx+3, idx:idx+3] = np.eye(3) * 1e-6
        # Bias random walk
        Q[bias_start:bias_start+3, bias_start:bias_start+3] = (
            np.eye(3) * (self._gyro_bias_noise * dt) ** 2
        )
        Q[bias_start+3:bias_start+6, bias_start+3:bias_start+6] = (
            np.eye(3) * (self._accel_bias_noise * dt) ** 2
        )

        self._P = Phi @ self._P @ Phi.T + Q
        self._P = 0.5 * (self._P + self._P.T)  # enforce symmetry

    # ------------------------------------------------------------------
    # Correct (kinematic measurement from FK)
    # ------------------------------------------------------------------

    def correct_kinematics(
        self, contact_id: int, p_foot_body: np.ndarray
    ) -> None:
        """Apply right-invariant correction from forward kinematics.

        Uses the constant measurement Jacobian from Hartley et al.
        (IJRR 2020, Eq. 35).  In the right-invariant formulation, H is
        state-independent — this is what makes the filter robust and
        prevents the cross-coupling divergence that occurs with standard
        EKF linearization.

        Args:
            contact_id: ID returned by augment_state.
            p_foot_body: Foot position in body/pelvis frame from FK (3,).
        """
        if contact_id not in self._contacts:
            return

        col = self._contacts[contact_id]
        R = self._X[0:3, 0:3]
        p = self._X[0:3, 4]
        d = self._X[0:3, col]  # world-frame contact position

        # Right-invariant innovation (Hartley 2020, Sec. IV-C)
        innovation = p_foot_body - R.T @ (d - p)

        # Right-invariant measurement Jacobian (Hartley 2020, Eq. 35)
        # H is CONSTANT — the key property of invariant filtering.
        n_c = len(self._contacts)
        state_dim = 9 + 3 * n_c + 6

        H = np.zeros((3, state_dim))
        # No rotation coupling (0), no velocity coupling (0)
        H[0:3, 6:9] = -np.eye(3)   # position block
        contact_idx_in_cov = 9 + list(self._contacts.keys()).index(contact_id) * 3
        H[0:3, contact_idx_in_cov:contact_idx_in_cov+3] = np.eye(3)  # contact block

        # Measurement noise
        R_meas = np.eye(3) * self._contact_noise ** 2

        # Kalman gain
        S = H @ self._P @ H.T + R_meas

        # Guard: skip correction if S is ill-conditioned or contains NaN
        if not np.all(np.isfinite(S)):
            return
        try:
            S_inv = np.linalg.solve(S, np.eye(3))
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(S_inv)):
            return

        K = self._P @ H.T @ S_inv

        # Guard: skip if Kalman gain is not finite
        if not np.all(np.isfinite(K)):
            return

        # State correction
        delta = K @ innovation

        # Apply correction to group state via right perturbation
        # delta_X = exp(delta_group), delta_bias = delta_bias_part
        n_group = 9 + 3 * n_c
        delta_group = delta[:n_group]
        delta_bias = delta[n_group:]

        # Build the correction matrix on SE_k(3)
        X_corr = self._build_correction(delta_group, n_c)
        self._X = X_corr @ self._X

        # Bias correction
        self._theta += delta_bias

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(state_dim) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_meas @ K.T
        self._P = 0.5 * (self._P + self._P.T)

        # Re-normalize rotation via SVD
        self._renormalize_rotation()

    # ------------------------------------------------------------------
    # Correct (direct velocity measurement / ZUPT)
    # ------------------------------------------------------------------

    def correct_velocity(self, v_measured: np.ndarray, noise_std: float = 0.1) -> None:
        """Apply a direct velocity measurement correction.

        Used for zero-velocity updates (ZUPT) when both feet are in
        contact and the robot is approximately stationary, or for
        velocity estimates from other sources.

        Args:
            v_measured: Measured velocity (3,) in world frame.
            noise_std: Measurement noise standard deviation (m/s).
        """
        n_c = len(self._contacts)
        state_dim = 9 + 3 * n_c + 6

        v = self._X[0:3, 3]
        innovation = v_measured - v

        # H maps velocity state to measurement: H[0:3, 3:6] = I
        H = np.zeros((3, state_dim))
        H[0:3, 3:6] = np.eye(3)

        R_meas = np.eye(3) * noise_std ** 2

        S = H @ self._P @ H.T + R_meas
        if not np.all(np.isfinite(S)):
            return
        try:
            S_inv = np.linalg.solve(S, np.eye(3))
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(S_inv)):
            return

        K = self._P @ H.T @ S_inv
        if not np.all(np.isfinite(K)):
            return

        delta = K @ innovation

        # Apply correction: velocity is at index [3:6] in the group
        n_group = 9 + 3 * n_c
        delta_group = delta[:n_group]
        delta_bias = delta[n_group:]

        X_corr = self._build_correction(delta_group, n_c)
        self._X = X_corr @ self._X
        self._theta += delta_bias

        I_KH = np.eye(state_dim) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_meas @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    # ------------------------------------------------------------------
    # Contact augmentation / marginalization
    # ------------------------------------------------------------------

    def augment_state(self, p_foot_body: np.ndarray) -> int:
        """Add a new contact to the state.

        Args:
            p_foot_body: Foot position in body frame from FK (3,).

        Returns:
            contact_id for use with correct_kinematics / marginalize_contact.
        """
        R = self._X[0:3, 0:3]
        p = self._X[0:3, 4]

        # World-frame contact position
        d_world = R @ p_foot_body + p

        # Grow X matrix
        old_dim = self._dim
        new_dim = old_dim + 1
        X_new = np.eye(new_dim)
        X_new[0:old_dim, 0:old_dim] = self._X
        X_new[0:3, old_dim - 1] = d_world  # new column for contact
        # Shift the last two rows/columns (the 1-padding) if needed:
        # Actually X is (3+2+k) but we store as (5+k, 5+k) with the identity
        # padding. Let's be more careful:

        # Rebuild X properly
        n_c_old = len(self._contacts)
        n_c_new = n_c_old + 1
        size = 5 + n_c_new
        X_new = np.eye(size)
        X_new[0:3, 0:3] = self._X[0:3, 0:3]  # R
        X_new[0:3, 3] = self._X[0:3, 3]  # v
        X_new[0:3, 4] = self._X[0:3, 4]  # p
        # Copy existing contacts
        for cid, col in self._contacts.items():
            # old col -> new col (same relative position)
            X_new[0:3, col] = self._X[0:3, col]
        # New contact goes in the next column
        new_col = 5 + n_c_old
        X_new[0:3, new_col] = d_world

        self._dim = size
        self._X = X_new

        # Grow covariance
        # New contact position uncertainty = R * contact_noise * R^T + pos_uncertainty
        state_dim_old = 9 + 3 * n_c_old + 6
        state_dim_new = 9 + 3 * n_c_new + 6

        P_new = np.zeros((state_dim_new, state_dim_new))

        # Copy existing blocks: [0:9+3*n_c_old] and biases [last 6]
        group_old = 9 + 3 * n_c_old
        group_new = 9 + 3 * n_c_new

        # Top-left: existing group states
        P_new[0:group_old, 0:group_old] = self._P[0:group_old, 0:group_old]
        # Cross terms: group <-> bias
        P_new[0:group_old, group_new:group_new+6] = self._P[0:group_old, group_old:group_old+6]
        P_new[group_new:group_new+6, 0:group_old] = self._P[group_old:group_old+6, 0:group_old]
        # Bias-bias
        P_new[group_new:group_new+6, group_new:group_new+6] = self._P[group_old:group_old+6, group_old:group_old+6]

        # New contact covariance block: derived from current R, p uncertainty
        # The contact position d = R * p_body + p, so its uncertainty
        # comes from rotation and position uncertainty, plus measurement noise.
        new_idx = 9 + 3 * n_c_old  # index of new contact in covariance

        # Cross-correlations: new contact is correlated with rotation and position
        # d = R * p_body + p  =>  J_rot = -R * [p_body]×,  J_pos = I
        J_rot = -R @ skew(p_foot_body)
        J_pos = np.eye(3)

        # Augmentation Jacobian (maps existing state to new contact)
        F = np.zeros((3, state_dim_old))
        F[0:3, 0:3] = J_rot
        F[0:3, 6:9] = J_pos

        P_dd = F @ self._P @ F.T + np.eye(3) * self._contact_noise ** 2
        P_dx = F @ self._P[:state_dim_old, :group_old]
        P_db = F @ self._P[:state_dim_old, group_old:group_old+6]

        P_new[new_idx:new_idx+3, new_idx:new_idx+3] = P_dd
        P_new[new_idx:new_idx+3, 0:group_old] = P_dx
        P_new[0:group_old, new_idx:new_idx+3] = P_dx.T
        P_new[new_idx:new_idx+3, group_new:group_new+6] = P_db
        P_new[group_new:group_new+6, new_idx:new_idx+3] = P_db.T

        self._P = 0.5 * (P_new + P_new.T)

        # Register contact
        contact_id = self._next_contact_id
        self._next_contact_id += 1
        self._contacts[contact_id] = new_col

        return contact_id

    def marginalize_contact(self, contact_id: int) -> None:
        """Remove a contact from the state.

        Args:
            contact_id: ID returned by augment_state.
        """
        if contact_id not in self._contacts:
            return

        col = self._contacts[contact_id]
        n_c = len(self._contacts)

        # Remove column from X
        size_old = 5 + n_c
        size_new = size_old - 1
        X_new = np.eye(size_new)
        X_new[0:3, 0:3] = self._X[0:3, 0:3]
        X_new[0:3, 3] = self._X[0:3, 3]
        X_new[0:3, 4] = self._X[0:3, 4]

        # Copy remaining contacts (skip the removed one)
        new_col_idx = 5
        new_contacts = {}
        for cid, old_col in sorted(self._contacts.items(), key=lambda x: x[1]):
            if cid == contact_id:
                continue
            X_new[0:3, new_col_idx] = self._X[0:3, old_col]
            new_contacts[cid] = new_col_idx
            new_col_idx += 1

        self._X = X_new
        self._dim = size_new

        # Remove rows/cols from covariance
        # Find the covariance index for this contact
        sorted_contacts = sorted(self._contacts.items(), key=lambda x: x[1])
        contact_order = [cid for cid, _ in sorted_contacts]
        cov_idx = 9 + contact_order.index(contact_id) * 3

        state_dim = 9 + 3 * n_c + 6
        keep = list(range(cov_idx)) + list(range(cov_idx + 3, state_dim))
        self._P = self._P[np.ix_(keep, keep)]
        self._P = 0.5 * (self._P + self._P.T)

        self._contacts = new_contacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_correction(self, delta_group: np.ndarray, n_c: int) -> np.ndarray:
        """Build correction matrix on SE_k(3) from tangent vector.

        The tangent vector has 9 + 3*n_c components:
        [rot(3), vel(3), pos(3), contact_1(3), ..., contact_k(3)]
        """
        size = 5 + n_c
        delta_X = np.eye(size)

        # Core SE_2(3) part
        xi = delta_group[0:9]
        core = se2_3_exp(xi)
        delta_X[0:5, 0:5] = core

        # Contact corrections: small translations
        for i in range(n_c):
            dp = delta_group[9 + 3*i : 9 + 3*(i+1)]
            delta_X[0:3, 5+i] = core[0:3, 0:3] @ dp  # rotated perturbation

        return delta_X

    def _renormalize_rotation(self) -> None:
        """Re-normalize rotation matrix via SVD."""
        R = self._X[0:3, 0:3]
        U, _, Vt = np.linalg.svd(R)
        # Ensure proper rotation (det = +1)
        S = np.eye(3)
        S[2, 2] = np.linalg.det(U @ Vt)
        self._X[0:3, 0:3] = U @ S @ Vt


def _rodrigues_increment(phi: np.ndarray) -> np.ndarray:
    """Small rotation: phi -> rotation matrix (same as so3_exp but inlined)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + skew(phi)
    K = skew(phi / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
