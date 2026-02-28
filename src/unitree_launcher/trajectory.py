"""Collision-free trajectory planning via mink (differential IK).

Pre-computes a collision-free joint trajectory from q_start to q_goal
using mink's QP-based IK solver with collision avoidance constraints.
The result can be replayed during execution instead of naive linear
interpolation, which can cause arm-torso self-collisions.

Requires ``mink>=1.0`` (optional dependency).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Tuple

import numpy as np

from unitree_launcher.config import G1_29DOF_JOINTS, G1_29DOF_MUJOCO_JOINTS

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

# ---------------------------------------------------------------------------
# Collision geom pairs for self-collision avoidance
# ---------------------------------------------------------------------------

# Each entry is (geoms_A, geoms_B) — mink checks all A×B distances.
# Covers: arms vs torso, arms vs head, arms vs legs, arm vs arm, arms vs pelvis.
# Foot geoms excluded (irrelevant for arm interpolation).

_LEFT_ARM_GEOMS = [
    "left_shoulder_yaw_collision",
    "left_elbow_collision",
    "left_hand_collision",
]

_RIGHT_ARM_GEOMS = [
    "right_shoulder_yaw_collision",
    "right_elbow_collision",
    "right_hand_collision",
]

_TORSO_GEOMS = [
    "torso_collision1",
    "torso_collision2",
    "torso_collision3",
]

G1_SELF_COLLISION_PAIRS: List[Tuple[List[str], List[str]]] = [
    # Arms vs torso
    (_LEFT_ARM_GEOMS, _TORSO_GEOMS),
    (_RIGHT_ARM_GEOMS, _TORSO_GEOMS),
    # Arms vs head
    (_LEFT_ARM_GEOMS, ["head_collision"]),
    (_RIGHT_ARM_GEOMS, ["head_collision"]),
    # Arms vs same-side leg
    (_LEFT_ARM_GEOMS, ["left_thigh", "left_shin"]),
    (_RIGHT_ARM_GEOMS, ["right_thigh", "right_shin"]),
    # Cross-body arm vs opposite leg
    (["left_elbow_collision", "left_hand_collision"], ["right_thigh"]),
    (["right_elbow_collision", "right_hand_collision"], ["left_thigh"]),
    # Arm vs arm
    (_LEFT_ARM_GEOMS, _RIGHT_ARM_GEOMS),
    # Arms vs pelvis
    (
        ["left_elbow_collision", "left_hand_collision",
         "right_elbow_collision", "right_hand_collision"],
        ["pelvis_collision"],
    ),
]


# ---------------------------------------------------------------------------
# TrajectoryPlan dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrajectoryPlan:
    """Result of a collision-free trajectory plan.

    Attributes:
        waypoints: Joint positions at each planner step, shape ``(T, 29)``,
            in config-native joint order (``G1_29DOF_JOINTS``).
        dt_plan: Planner timestep (seconds).
        converged: Whether the planner reached the goal within threshold.
    """
    waypoints: np.ndarray
    dt_plan: float
    converged: bool


# ---------------------------------------------------------------------------
# Trajectory planner
# ---------------------------------------------------------------------------

def plan_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    dt: float = 0.02,
    max_steps: int = 500,
    threshold: float = 0.01,
    posture_cost: float = 1.0,
    posture_gain: float = 0.1,
    collision_min_dist: float = 0.01,
    collision_detect_dist: float = 0.15,
    solver: str = "daqp",
    damping: float = 1e-1,
    model_path: str | None = None,
) -> TrajectoryPlan:
    """Plan a collision-free trajectory from q_start to q_goal.

    Loads a fresh MuJoCo model, creates a mink Configuration with the
    freejoint base locked, and iterates posture-task IK with collision
    avoidance until convergence or max_steps.

    Args:
        q_start: Start joint positions, shape ``(29,)``, config order.
        q_goal: Goal joint positions, shape ``(29,)``, config order.
        dt: Planner integration timestep.
        max_steps: Maximum planner iterations.
        threshold: Convergence threshold (max absolute joint error, rad).
        posture_cost: PostureTask cost weight.
        posture_gain: PostureTask convergence gain (0–1).  Lower values
            make the planner take smaller steps per iteration, producing
            smoother trajectories.  Default 0.1 = reduce ~10% of error
            per step.
        collision_min_dist: Minimum distance between collision geoms (m).
        collision_detect_dist: Distance at which collision avoidance activates (m).
        solver: QP solver name (default: "daqp").
        damping: Regularization damping for solve_ik.
        model_path: Path to MuJoCo XML. Defaults to scene_29dof.xml.

    Returns:
        TrajectoryPlan with waypoints and convergence status.
    """
    import mink
    import mujoco

    if model_path is None:
        model_path = str(_ASSETS_DIR / "robots" / "g1" / "scene_29dof.xml")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Build mapping: config joint index -> MuJoCo qpos address
    joint_qpos_adr = []
    for cfg_name in G1_29DOF_JOINTS:
        mj_name = G1_29DOF_MUJOCO_JOINTS[cfg_name]
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
        if jnt_id < 0:
            raise ValueError(f"Joint {mj_name!r} not found in model")
        joint_qpos_adr.append(model.jnt_qposadr[jnt_id])

    # Set initial configuration
    data.qpos[:] = 0.0
    # Set freejoint quaternion to identity
    data.qpos[3] = 1.0
    for cfg_i, qadr in enumerate(joint_qpos_adr):
        data.qpos[qadr] = q_start[cfg_i]
    mujoco.mj_forward(model, data)

    # Create mink configuration
    configuration = mink.Configuration(model, data.qpos.copy())

    # Create posture task targeting q_goal
    posture_task = mink.PostureTask(model=model, cost=posture_cost, gain=posture_gain)
    target_q = data.qpos.copy()
    for cfg_i, qadr in enumerate(joint_qpos_adr):
        target_q[qadr] = q_goal[cfg_i]
    posture_task.set_target(target_q)

    tasks = [posture_task]

    # Create collision avoidance limit
    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=G1_SELF_COLLISION_PAIRS,
        minimum_distance_from_collisions=collision_min_dist,
        collision_detection_distance=collision_detect_dist,
    )
    limits = [
        mink.ConfigurationLimit(model),
        collision_limit,
    ]

    # Freejoint DOF indices (first 6 velocity DOFs)
    n_base_dof = 6

    # Iterate IK — start with q_start so the trajectory begins exactly
    # at the current robot configuration (no first-frame snap).
    waypoints = [q_start.copy()]
    converged = False

    for step in range(max_steps):
        vel = mink.solve_ik(
            configuration, tasks, dt, solver, damping=damping, limits=limits,
        )
        # Zero out base DOFs so the robot doesn't translate/rotate
        vel[:n_base_dof] = 0.0
        configuration.integrate_inplace(vel, dt)

        # Extract current joint positions in config order
        current_q = np.array([
            configuration.q[qadr] for qadr in joint_qpos_adr
        ])
        waypoints.append(current_q)

        # Check convergence
        if np.max(np.abs(current_q - q_goal)) < threshold:
            converged = True
            break

    return TrajectoryPlan(
        waypoints=np.array(waypoints),
        dt_plan=dt,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_trajectory(
    plan: TrajectoryPlan,
    target_duration: float,
    target_dt: float,
) -> np.ndarray:
    """Resample planner output to execution frequency.

    Uses linear interpolation per joint to map from planner timesteps
    to the target duration and frequency.

    Args:
        plan: TrajectoryPlan from ``plan_trajectory()``.
        target_duration: Desired execution duration in seconds.
        target_dt: Execution timestep (e.g. 1/50 for 50 Hz).

    Returns:
        Resampled waypoints, shape ``(N_steps, 29)``.
    """
    n_plan = len(plan.waypoints)
    if n_plan == 0:
        raise ValueError("Empty trajectory plan")

    plan_times = np.arange(n_plan) * plan.dt_plan
    plan_total = plan_times[-1]

    n_exec = int(target_duration / target_dt)
    exec_times = np.arange(n_exec) * target_dt

    # Scale execution times to fit the planner timeline
    if plan_total > 0:
        scaled_times = exec_times * (plan_total / target_duration)
    else:
        scaled_times = np.zeros(n_exec)

    n_joints = plan.waypoints.shape[1]
    resampled = np.zeros((n_exec, n_joints))
    for j in range(n_joints):
        resampled[:, j] = np.interp(scaled_times, plan_times, plan.waypoints[:, j])

    return resampled
