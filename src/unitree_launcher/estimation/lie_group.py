"""SO(3) and SE_2(3) Lie group operations for the InEKF.

Implements the minimum set of operations needed for right-invariant EKF
on SE_2(3): exponential/logarithmic maps, adjoint representation, and
skew-symmetric utilities.

References:
    Hartley et al., "Contact-Aided Invariant Extended Kalman Filtering
    for Robot State Estimation", IJRR 2020.
"""
from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def unskew(S: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix -> 3-vector."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map SO(3): 3-vector -> 3x3 rotation matrix.

    Uses Rodrigues formula with small-angle guard at ||phi|| < 1e-7.
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-7:
        # First-order Taylor: R ≈ I + [phi]×
        return np.eye(3) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3): 3x3 rotation matrix -> 3-vector."""
    cos_angle = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-7:
        # Small-angle: phi ≈ unskew(R - I)
        return unskew(R - np.eye(3))
    return (angle / (2.0 * np.sin(angle))) * unskew(R - R.T)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3), needed for SE_2(3) exponential map.

    J_l(phi) = I + (1-cos||phi||)/||phi||^2 * [phi]× +
               (||phi||-sin||phi||)/||phi||^3 * [phi]×^2
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-7:
        return np.eye(3) + 0.5 * skew(phi)
    K = skew(phi)
    a2 = angle * angle
    return (
        np.eye(3)
        + ((1.0 - np.cos(angle)) / a2) * K
        + ((angle - np.sin(angle)) / (a2 * angle)) * (K @ K)
    )


def se2_3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map SE_2(3): 9-vector -> 5x5 matrix.

    xi = [phi(3), rho_v(3), rho_p(3)] where:
        phi   = rotation tangent vector
        rho_v = velocity tangent vector
        rho_p = position tangent vector

    Returns 5x5 matrix:
        [[R, v, p],
         [0, 1, 0],
         [0, 0, 1]]
    """
    phi = xi[0:3]
    rho_v = xi[3:6]
    rho_p = xi[6:9]

    R = so3_exp(phi)
    J = so3_left_jacobian(phi)

    X = np.eye(5)
    X[0:3, 0:3] = R
    X[0:3, 3] = J @ rho_v
    X[0:3, 4] = J @ rho_p
    return X


def adjoint_se2_3(X: np.ndarray) -> np.ndarray:
    """Adjoint representation of SE_2(3): 5x5 matrix -> 9x9 matrix.

    For X = [[R, v, p], [0,1,0], [0,0,1]], the adjoint is:
        [[R,    0,    0  ],
         [[v]×R, R,    0  ],
         [[p]×R, 0,    R  ]]
    """
    R = X[0:3, 0:3]
    v = X[0:3, 3]
    p = X[0:3, 4]

    Ad = np.zeros((9, 9))
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = skew(v) @ R
    Ad[3:6, 3:6] = R
    Ad[6:9, 0:3] = skew(p) @ R
    Ad[6:9, 6:9] = R
    return Ad


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix (body->world).

    Same convention as observations._quat_to_rotation_matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion (Shepperd's method)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)  # normalize
