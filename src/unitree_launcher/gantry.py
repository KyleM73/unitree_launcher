"""Gantry simulation utilities.

Elastic band, gantry positioning, and shared helpers for scripts that
run the G1 on a virtual or real gantry harness.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    ISAACLAB_KP_29DOF,
    ISAACLAB_KD_29DOF,
    Q_HOME_29DOF,
    STANDBY_KP_29DOF,
    STANDBY_KD_29DOF,
    UNITREE_KP_29DOF,
    UNITREE_KD_29DOF,
)

ANCHOR_POINT = np.array([0.0, 0.0, 1.2])


# ---------------------------------------------------------------------------
# Elastic band (matching unitree_mujoco)
# ---------------------------------------------------------------------------

class ElasticBand:
    """Virtual elastic band — spring-damper pulling torso toward an anchor.

    Matching the implementation in unitree_mujoco (simulate_python/
    unitree_sdk2py_bridge.py, ElasticBand class). The force is:

        F = (K * (distance - rest_length) - D * radial_velocity) * direction

    Applied to ``xfrc_applied`` on the torso body each physics substep.
    """

    def __init__(
        self,
        stiffness: float = 500.0,
        damping: float = 300.0,
        point: np.ndarray = ANCHOR_POINT,
        length: float = 0.0,
    ):
        self.stiffness = stiffness
        self.damping = damping
        self.point = point.copy()
        self.length = length
        self.enable = True

    def advance(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Compute spring-damper force given position and velocity."""
        delta_x = self.point - x
        distance = np.linalg.norm(delta_x)
        if distance < 1e-8:
            return np.zeros(3)
        direction = delta_x / distance
        v = np.dot(dx, direction)
        return (self.stiffness * (distance - self.length) - self.damping * v) * direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_torso_body_id(model: mujoco.MjModel) -> int:
    """Get torso_link body id (matching unitree_mujoco's body selection)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if body_id < 0:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return body_id


def smooth_alpha(t: float, duration: float) -> float:
    """Cosine ease-in-out: 0->1 over [0, duration]."""
    frac = max(0.0, min(1.0, t / duration))
    return 0.5 * (1.0 - math.cos(math.pi * frac))


def build_gain_arrays(kind: str = "isaaclab") -> tuple[np.ndarray, np.ndarray]:
    """Build Kp/Kd arrays in G1_29DOF_JOINTS order.

    Args:
        kind: "isaaclab" for training gains, "standby" for MTC hold gains,
            "unitree" for official Unitree deploy gains (unitree_rl_lab).
    """
    if kind == "isaaclab":
        kp_dict, kd_dict = ISAACLAB_KP_29DOF, ISAACLAB_KD_29DOF
    elif kind == "unitree":
        kp_dict, kd_dict = UNITREE_KP_29DOF, UNITREE_KD_29DOF
    else:
        kp_dict, kd_dict = STANDBY_KP_29DOF, STANDBY_KD_29DOF
    kp = np.array([kp_dict[j] for j in G1_29DOF_JOINTS], dtype=np.float64)
    kd = np.array([kd_dict[j] for j in G1_29DOF_JOINTS], dtype=np.float64)
    return kp, kd


def build_home_positions() -> np.ndarray:
    """Build home position array in G1_29DOF_JOINTS order."""
    return np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS], dtype=np.float64)


def apply_band(data: mujoco.MjData, band: ElasticBand, torso_id: int) -> None:
    """Apply elastic band force to torso body via xfrc_applied."""
    if band.enable:
        f = band.advance(data.qpos[:3], data.qvel[:3])
        data.xfrc_applied[torso_id, :3] = f
    else:
        data.xfrc_applied[torso_id, :3] = 0.0


def enable_gantry(sim_robot) -> None:
    """Position the robot below the elastic band anchor point.

    Sets base at anchor point and joints to home pose. The elastic band
    (with rest_length=0) will hold the robot at the anchor height.
    """
    sim_robot.mj_data.qpos[0:3] = ANCHOR_POINT
    sim_robot.mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    sim_robot.mj_data.qvel[:] = 0.0

    home_q = build_home_positions()
    for cfg_i, joint_name in enumerate(G1_29DOF_JOINTS):
        jnt_name = joint_name + "_joint"
        jnt_id = mujoco.mj_name2id(
            sim_robot.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name
        )
        if jnt_id >= 0:
            qadr = sim_robot.mj_model.jnt_qposadr[jnt_id]
            sim_robot.mj_data.qpos[qadr] = home_q[cfg_i]

    mujoco.mj_forward(sim_robot.mj_model, sim_robot.mj_data)


def setup_gantry_band(robot, band: ElasticBand, torso_id: int) -> None:
    """Register elastic band as a substep callback on *robot*.

    After calling this, ``robot.step()`` automatically applies the band
    force before each physics substep — no need for a custom step function.
    """
    def callback(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        apply_band(data, band, torso_id)

    robot.set_substep_callback(callback)


# ---------------------------------------------------------------------------
# Collision-free interpolation trajectory (optional mink dependency)
# ---------------------------------------------------------------------------

def plan_interpolation_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray | None = None,
    duration: float = 5.0,
    policy_hz: float = 50.0,
) -> np.ndarray:
    """Plan a collision-free interpolation trajectory via mink.

    Wraps :func:`trajectory.plan_trajectory` and
    :func:`trajectory.resample_trajectory` with standard defaults.

    Args:
        q_start: Current joint positions, shape ``(29,)``, config order.
        q_goal: Target joint positions. Defaults to home pose.
        duration: Execution duration in seconds.
        policy_hz: Execution frequency in Hz.

    Returns:
        Waypoints array of shape ``(N_steps, 29)``.

    Raises:
        ImportError: If mink is not installed.
        RuntimeError: If trajectory planning fails.
    """
    from unitree_launcher.trajectory import plan_trajectory, resample_trajectory

    if q_goal is None:
        q_goal = build_home_positions()

    plan = plan_trajectory(q_start, q_goal)
    return resample_trajectory(plan, duration, 1.0 / policy_hz)
