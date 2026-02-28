"""Analytical forward kinematics for G1 legs: pelvis -> foot site.

Hardcoded kinematic chains extracted from g1_29dof.xml body pos/quat
attributes. Each leg has 6 joints: hip_pitch(Y), hip_roll(X), hip_yaw(Z),
knee(Y), ankle_pitch(Y), ankle_roll(X).

Joint ordering (robot-native):
    Left leg:  indices 0-5  [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    Right leg: indices 6-11 [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
"""
from __future__ import annotations

import numpy as np


def _rotx(a: float) -> np.ndarray:
    """Rotation about X axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(a: float) -> np.ndarray:
    """Rotation about Y axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotz(a: float) -> np.ndarray:
    """Rotation about Z axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


# ======================================================================
# Kinematic chain parameters from g1_29dof.xml
# ======================================================================

# Left leg: pelvis -> hip_pitch_link -> hip_roll_link -> hip_yaw_link
#           -> knee_link -> ankle_pitch_link -> ankle_roll_link (foot site)
#
# Each entry: (translation, pre-rotation quaternion wxyz, joint axis)
# Pre-rotation is the quat= attribute on the child body.

_LEFT_CHAIN = [
    # pelvis -> left_hip_pitch_link
    {"pos": np.array([0.0, 0.064452, -0.1027]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "Y"},
    # -> left_hip_roll_link
    {"pos": np.array([0.0, 0.052, -0.030465]),
     "quat": np.array([0.996179, 0.0, -0.0873386, 0.0]),
     "axis": "X"},
    # -> left_hip_yaw_link
    {"pos": np.array([0.025001, 0.0, -0.12412]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "Z"},
    # -> left_knee_link
    {"pos": np.array([-0.078273, 0.0021489, -0.17734]),
     "quat": np.array([0.996179, 0.0, 0.0873386, 0.0]),
     "axis": "Y"},
    # -> left_ankle_pitch_link
    {"pos": np.array([0.0, -9.4445e-05, -0.30001]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "Y"},
    # -> left_ankle_roll_link (foot site at origin)
    {"pos": np.array([0.0, 0.0, -0.017558]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "X"},
]

_RIGHT_CHAIN = [
    # pelvis -> right_hip_pitch_link
    {"pos": np.array([0.0, -0.064452, -0.1027]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "Y"},
    # -> right_hip_roll_link
    {"pos": np.array([0.0, -0.052, -0.030465]),
     "quat": np.array([0.996179, 0.0, -0.0873386, 0.0]),
     "axis": "X"},
    # -> right_hip_yaw_link
    {"pos": np.array([0.025001, 0.0, -0.12412]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "Z"},
    # -> right_knee_link
    {"pos": np.array([-0.078273, -0.0021489, -0.17734]),
     "quat": np.array([0.996179, 0.0, 0.0873386, 0.0]),
     "axis": "Y"},
    # -> right_ankle_pitch_link
    {"pos": np.array([0.0, 9.4445e-05, -0.30001]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "Y"},
    # -> right_ankle_roll_link (foot site at origin)
    {"pos": np.array([0.0, 0.0, -0.017558]),
     "quat": np.array([1.0, 0.0, 0.0, 0.0]),
     "axis": "X"},
]

_JOINT_ROTATION = {"X": _rotx, "Y": _roty, "Z": _rotz}


def _fk_chain(chain: list, q: np.ndarray) -> np.ndarray:
    """Compute foot position in pelvis frame for a 6-joint leg chain.

    Args:
        chain: List of link dicts with pos, quat, axis.
        q: Joint angles (6,) in order [hip_pitch, hip_roll, hip_yaw,
           knee, ankle_pitch, ankle_roll].

    Returns:
        Foot position (3,) in pelvis frame.
    """
    R = np.eye(3)
    p = np.zeros(3)

    for i, link in enumerate(chain):
        # Apply link transform: translate then rotate by pre-rotation quat
        p = p + R @ link["pos"]
        R_pre = _quat_to_rot(link["quat"])
        R = R @ R_pre

        # Apply joint rotation
        R_joint = _JOINT_ROTATION[link["axis"]](q[i])
        R = R @ R_joint

    return p


class G1Kinematics:
    """Forward kinematics for G1 legs.

    Computes foot positions in the pelvis (body) frame using the
    analytical kinematic chain from the MJCF model.
    """

    # Robot-native joint indices
    LEFT_LEG_INDICES = list(range(0, 6))
    RIGHT_LEG_INDICES = list(range(6, 12))

    def left_foot_position(self, q_full: np.ndarray) -> np.ndarray:
        """Left foot position in pelvis frame.

        Args:
            q_full: Full robot joint positions (29,) or left leg subset (6,).

        Returns:
            Position (3,) in pelvis frame.
        """
        q = q_full[self.LEFT_LEG_INDICES] if q_full.shape[0] > 6 else q_full
        return _fk_chain(_LEFT_CHAIN, q)

    def right_foot_position(self, q_full: np.ndarray) -> np.ndarray:
        """Right foot position in pelvis frame.

        Args:
            q_full: Full robot joint positions (29,) or right leg subset (6,).

        Returns:
            Position (3,) in pelvis frame.
        """
        q = q_full[self.RIGHT_LEG_INDICES] if q_full.shape[0] > 6 else q_full
        return _fk_chain(_RIGHT_CHAIN, q)

    def left_foot_jacobian(self, q_full: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Positional Jacobian of left foot w.r.t. left leg joints.

        Computed via central finite differences (12 FK evaluations).

        Args:
            q_full: Full joint positions (29,).
            eps: Perturbation size for finite differences.

        Returns:
            Jacobian (3, 6) mapping left leg joint velocities to foot velocity.
        """
        q_leg = q_full[self.LEFT_LEG_INDICES].copy()
        J = np.zeros((3, 6))
        for i in range(6):
            q_plus = q_leg.copy()
            q_minus = q_leg.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            J[:, i] = (_fk_chain(_LEFT_CHAIN, q_plus) - _fk_chain(_LEFT_CHAIN, q_minus)) / (2 * eps)
        return J

    def right_foot_jacobian(self, q_full: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Positional Jacobian of right foot w.r.t. right leg joints.

        Args:
            q_full: Full joint positions (29,).
            eps: Perturbation size for finite differences.

        Returns:
            Jacobian (3, 6) mapping right leg joint velocities to foot velocity.
        """
        q_leg = q_full[self.RIGHT_LEG_INDICES].copy()
        J = np.zeros((3, 6))
        for i in range(6):
            q_plus = q_leg.copy()
            q_minus = q_leg.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            J[:, i] = (_fk_chain(_RIGHT_CHAIN, q_plus) - _fk_chain(_RIGHT_CHAIN, q_minus)) / (2 * eps)
        return J
