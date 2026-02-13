"""Robot constants and configuration for the Unitree G1 humanoid.

This module defines:
- Joint name lists for 29-DOF and 23-DOF variants
- MuJoCo joint name mappings
- IsaacLab joint ordering and index mappings
- DDS/IDL name mappings
- Joint limits (position, torque)
- Home positions
- Configuration dataclasses and YAML loading
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


# ============================================================================
# Section 1: Robot Joint Constants (from SPEC sections 2.2–2.4)
# ============================================================================

# ---------------------------------------------------------------------------
# 29-DOF joint names in robot-native order
# ---------------------------------------------------------------------------
G1_29DOF_JOINTS: List[str] = [
    # Left leg (0–5)
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    # Right leg (6–11)
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    # Waist (12–14)
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    # Left arm (15–21)
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    # Right arm (22–28)
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

# ---------------------------------------------------------------------------
# 23-DOF joint names in robot-native order
# ---------------------------------------------------------------------------
G1_23DOF_JOINTS: List[str] = [
    # Left leg (0–5) — same as 29-DOF
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    # Right leg (6–11) — same as 29-DOF
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    # Torso (12) — single joint replaces waist_yaw/roll/pitch
    "torso",
    # Left arm (13–17) — 5-DOF, no wrist joints
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_roll",
    # Right arm (18–22) — 5-DOF, no wrist joints
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_elbow_roll",
]

# ---------------------------------------------------------------------------
# Config name -> MuJoCo joint name mapping (29-DOF)
# ---------------------------------------------------------------------------
G1_29DOF_MUJOCO_JOINTS: Dict[str, str] = {
    "left_hip_pitch": "left_hip_pitch_joint",
    "left_hip_roll": "left_hip_roll_joint",
    "left_hip_yaw": "left_hip_yaw_joint",
    "left_knee": "left_knee_joint",
    "left_ankle_pitch": "left_ankle_pitch_joint",
    "left_ankle_roll": "left_ankle_roll_joint",
    "right_hip_pitch": "right_hip_pitch_joint",
    "right_hip_roll": "right_hip_roll_joint",
    "right_hip_yaw": "right_hip_yaw_joint",
    "right_knee": "right_knee_joint",
    "right_ankle_pitch": "right_ankle_pitch_joint",
    "right_ankle_roll": "right_ankle_roll_joint",
    "waist_yaw": "waist_yaw_joint",
    "waist_roll": "waist_roll_joint",
    "waist_pitch": "waist_pitch_joint",
    "left_shoulder_pitch": "left_shoulder_pitch_joint",
    "left_shoulder_roll": "left_shoulder_roll_joint",
    "left_shoulder_yaw": "left_shoulder_yaw_joint",
    "left_elbow": "left_elbow_joint",
    "left_wrist_roll": "left_wrist_roll_joint",
    "left_wrist_pitch": "left_wrist_pitch_joint",
    "left_wrist_yaw": "left_wrist_yaw_joint",
    "right_shoulder_pitch": "right_shoulder_pitch_joint",
    "right_shoulder_roll": "right_shoulder_roll_joint",
    "right_shoulder_yaw": "right_shoulder_yaw_joint",
    "right_elbow": "right_elbow_joint",
    "right_wrist_roll": "right_wrist_roll_joint",
    "right_wrist_pitch": "right_wrist_pitch_joint",
    "right_wrist_yaw": "right_wrist_yaw_joint",
}

# Config name -> MuJoCo joint name mapping (23-DOF)
# Note: some config names differ from MuJoCo names (torso->waist_yaw_joint,
# elbow_pitch->elbow_joint, elbow_roll->wrist_roll_joint)
G1_23DOF_MUJOCO_JOINTS: Dict[str, str] = {
    "left_hip_pitch": "left_hip_pitch_joint",
    "left_hip_roll": "left_hip_roll_joint",
    "left_hip_yaw": "left_hip_yaw_joint",
    "left_knee": "left_knee_joint",
    "left_ankle_pitch": "left_ankle_pitch_joint",
    "left_ankle_roll": "left_ankle_roll_joint",
    "right_hip_pitch": "right_hip_pitch_joint",
    "right_hip_roll": "right_hip_roll_joint",
    "right_hip_yaw": "right_hip_yaw_joint",
    "right_knee": "right_knee_joint",
    "right_ankle_pitch": "right_ankle_pitch_joint",
    "right_ankle_roll": "right_ankle_roll_joint",
    "torso": "waist_yaw_joint",
    "left_shoulder_pitch": "left_shoulder_pitch_joint",
    "left_shoulder_roll": "left_shoulder_roll_joint",
    "left_shoulder_yaw": "left_shoulder_yaw_joint",
    "left_elbow_pitch": "left_elbow_joint",
    "left_elbow_roll": "left_wrist_roll_joint",
    "right_shoulder_pitch": "right_shoulder_pitch_joint",
    "right_shoulder_roll": "right_shoulder_roll_joint",
    "right_shoulder_yaw": "right_shoulder_yaw_joint",
    "right_elbow_pitch": "right_elbow_joint",
    "right_elbow_roll": "right_wrist_roll_joint",
}

# ---------------------------------------------------------------------------
# IsaacLab joint ordering (29-DOF, MuJoCo joint names)
# ---------------------------------------------------------------------------
ISAACLAB_G1_29DOF_JOINTS: List[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# IsaacLab index -> robot-native index. E.g. ISAACLAB_TO_NATIVE_INDICES[0] = 0
# means IsaacLab joint 0 (left_hip_pitch_joint) maps to native index 0.
ISAACLAB_TO_NATIVE_INDICES: List[int] = [
    0,   # left_hip_pitch_joint   -> native 0
    6,   # right_hip_pitch_joint  -> native 6
    12,  # waist_yaw_joint        -> native 12
    1,   # left_hip_roll_joint    -> native 1
    7,   # right_hip_roll_joint   -> native 7
    13,  # waist_roll_joint       -> native 13
    2,   # left_hip_yaw_joint     -> native 2
    8,   # right_hip_yaw_joint    -> native 8
    14,  # waist_pitch_joint      -> native 14
    3,   # left_knee_joint        -> native 3
    9,   # right_knee_joint       -> native 9
    15,  # left_shoulder_pitch    -> native 15
    22,  # right_shoulder_pitch   -> native 22
    4,   # left_ankle_pitch       -> native 4
    10,  # right_ankle_pitch      -> native 10
    16,  # left_shoulder_roll     -> native 16
    23,  # right_shoulder_roll    -> native 23
    5,   # left_ankle_roll        -> native 5
    11,  # right_ankle_roll       -> native 11
    17,  # left_shoulder_yaw      -> native 17
    24,  # right_shoulder_yaw     -> native 24
    18,  # left_elbow             -> native 18
    25,  # right_elbow            -> native 25
    19,  # left_wrist_roll        -> native 19
    26,  # right_wrist_roll       -> native 26
    20,  # left_wrist_pitch       -> native 20
    27,  # right_wrist_pitch      -> native 27
    21,  # left_wrist_yaw         -> native 21
    28,  # right_wrist_yaw        -> native 28
]

# ---------------------------------------------------------------------------
# Home positions (SPEC section 2.4)
# ---------------------------------------------------------------------------
Q_HOME_29DOF: Dict[str, float] = {
    "left_hip_pitch": -0.312,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.669,
    "left_ankle_pitch": -0.33,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.312,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.669,
    "right_ankle_pitch": -0.33,
    "right_ankle_roll": 0.0,
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "waist_pitch": 0.0,
    "left_shoulder_pitch": 0.2,
    "left_shoulder_roll": 0.2,
    "left_shoulder_yaw": 0.0,
    "left_elbow": 0.6,
    "left_wrist_roll": 0.0,
    "left_wrist_pitch": 0.0,
    "left_wrist_yaw": 0.0,
    "right_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_yaw": 0.0,
    "right_elbow": 0.6,
    "right_wrist_roll": 0.0,
    "right_wrist_pitch": 0.0,
    "right_wrist_yaw": 0.0,
}

Q_HOME_23DOF: Dict[str, float] = {
    "left_hip_pitch": -0.312,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.669,
    "left_ankle_pitch": -0.33,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.312,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.669,
    "right_ankle_pitch": -0.33,
    "right_ankle_roll": 0.0,
    "torso": 0.0,
    "left_shoulder_pitch": 0.2,
    "left_shoulder_roll": 0.2,
    "left_shoulder_yaw": 0.0,
    "left_elbow_pitch": 0.6,
    "left_elbow_roll": 0.0,
    "right_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": 0.6,
    "right_elbow_roll": 0.0,
}

# ---------------------------------------------------------------------------
# Joint position limits: config-name -> (min, max) in radians (SPEC 2.2)
# ---------------------------------------------------------------------------
JOINT_LIMITS_29DOF: Dict[str, Tuple[float, float]] = {
    "left_hip_pitch": (-2.53, 2.88),
    "left_hip_roll": (-0.52, 2.97),
    "left_hip_yaw": (-2.76, 2.76),
    "left_knee": (-0.09, 2.88),
    "left_ankle_pitch": (-0.87, 0.52),
    "left_ankle_roll": (-0.26, 0.26),
    "right_hip_pitch": (-2.53, 2.88),
    "right_hip_roll": (-2.97, 0.52),
    "right_hip_yaw": (-2.76, 2.76),
    "right_knee": (-0.09, 2.88),
    "right_ankle_pitch": (-0.87, 0.52),
    "right_ankle_roll": (-0.26, 0.26),
    "waist_yaw": (-2.62, 2.62),
    "waist_roll": (-0.52, 0.52),
    "waist_pitch": (-0.52, 0.52),
    "left_shoulder_pitch": (-3.09, 2.67),
    "left_shoulder_roll": (-1.59, 2.25),
    "left_shoulder_yaw": (-2.62, 2.62),
    "left_elbow": (-1.05, 2.09),
    "left_wrist_roll": (-1.97, 1.97),
    "left_wrist_pitch": (-1.61, 1.61),
    "left_wrist_yaw": (-1.61, 1.61),
    "right_shoulder_pitch": (-3.09, 2.67),
    "right_shoulder_roll": (-2.25, 1.59),
    "right_shoulder_yaw": (-2.62, 2.62),
    "right_elbow": (-1.05, 2.09),
    "right_wrist_roll": (-1.97, 1.97),
    "right_wrist_pitch": (-1.61, 1.61),
    "right_wrist_yaw": (-1.61, 1.61),
}

JOINT_LIMITS_23DOF: Dict[str, Tuple[float, float]] = {
    "left_hip_pitch": (-2.53, 2.88),
    "left_hip_roll": (-0.52, 2.97),
    "left_hip_yaw": (-2.76, 2.76),
    "left_knee": (-0.09, 2.88),
    "left_ankle_pitch": (-0.87, 0.52),
    "left_ankle_roll": (-0.26, 0.26),
    "right_hip_pitch": (-2.53, 2.88),
    "right_hip_roll": (-2.97, 0.52),
    "right_hip_yaw": (-2.76, 2.76),
    "right_knee": (-0.09, 2.88),
    "right_ankle_pitch": (-0.87, 0.52),
    "right_ankle_roll": (-0.26, 0.26),
    "torso": (-2.62, 2.62),
    "left_shoulder_pitch": (-3.09, 2.67),
    "left_shoulder_roll": (-1.59, 2.25),
    "left_shoulder_yaw": (-2.62, 2.62),
    "left_elbow_pitch": (-1.05, 2.09),
    "left_elbow_roll": (-1.97, 1.97),
    "right_shoulder_pitch": (-3.09, 2.67),
    "right_shoulder_roll": (-2.25, 1.59),
    "right_shoulder_yaw": (-2.62, 2.62),
    "right_elbow_pitch": (-1.05, 2.09),
    "right_elbow_roll": (-1.97, 1.97),
}

# ---------------------------------------------------------------------------
# Standby PD gains for holding the home pose (from BeyondMimic deployment).
# These are much higher than the walking policy gains to hold against gravity.
# Source: https://github.com/HybridRobotics/motion_tracking_controller
# ---------------------------------------------------------------------------
STANDBY_KP_29DOF: Dict[str, float] = {
    "left_hip_pitch": 350.0, "left_hip_roll": 200.0, "left_hip_yaw": 200.0,
    "left_knee": 300.0, "left_ankle_pitch": 300.0, "left_ankle_roll": 150.0,
    "right_hip_pitch": 350.0, "right_hip_roll": 200.0, "right_hip_yaw": 200.0,
    "right_knee": 300.0, "right_ankle_pitch": 300.0, "right_ankle_roll": 150.0,
    "waist_yaw": 200.0, "waist_roll": 200.0, "waist_pitch": 200.0,
    "left_shoulder_pitch": 40.0, "left_shoulder_roll": 40.0,
    "left_shoulder_yaw": 40.0, "left_elbow": 40.0,
    "left_wrist_roll": 40.0, "left_wrist_pitch": 40.0, "left_wrist_yaw": 40.0,
    "right_shoulder_pitch": 40.0, "right_shoulder_roll": 40.0,
    "right_shoulder_yaw": 40.0, "right_elbow": 40.0,
    "right_wrist_roll": 40.0, "right_wrist_pitch": 40.0, "right_wrist_yaw": 40.0,
}
STANDBY_KD_29DOF: Dict[str, float] = {
    "left_hip_pitch": 15.0, "left_hip_roll": 15.0, "left_hip_yaw": 15.0,
    "left_knee": 20.0, "left_ankle_pitch": 15.0, "left_ankle_roll": 15.0,
    "right_hip_pitch": 15.0, "right_hip_roll": 15.0, "right_hip_yaw": 15.0,
    "right_knee": 20.0, "right_ankle_pitch": 15.0, "right_ankle_roll": 15.0,
    "waist_yaw": 10.0, "waist_roll": 10.0, "waist_pitch": 10.0,
    "left_shoulder_pitch": 3.0, "left_shoulder_roll": 3.0,
    "left_shoulder_yaw": 3.0, "left_elbow": 3.0,
    "left_wrist_roll": 3.0, "left_wrist_pitch": 3.0, "left_wrist_yaw": 3.0,
    "right_shoulder_pitch": 3.0, "right_shoulder_roll": 3.0,
    "right_shoulder_yaw": 3.0, "right_elbow": 3.0,
    "right_wrist_roll": 3.0, "right_wrist_pitch": 3.0, "right_wrist_yaw": 3.0,
}

# ---------------------------------------------------------------------------
# IsaacLab G1 Cylinder training gains (per-joint stiffness/damping).
# Source: G1_CYLINDER_CFG from IsaacLab g1_cylinder.py
#   stiffness = armature * (10 * 2π)²
#   damping   = 2 * 2.0 * armature * (10 * 2π)
# Motor types: 7520_14 (hip_pitch/yaw, waist_yaw), 7520_22 (hip_roll, knee),
#   5020 (ankle, shoulder, elbow, wrist_roll), 4010 (wrist_pitch/yaw),
#   2×5020 (ankle, waist_roll/pitch)
# ---------------------------------------------------------------------------
ISAACLAB_KP_29DOF: Dict[str, float] = {
    "left_hip_pitch": 40.18, "left_hip_roll": 99.10, "left_hip_yaw": 40.18,
    "left_knee": 99.10, "left_ankle_pitch": 28.50, "left_ankle_roll": 28.50,
    "right_hip_pitch": 40.18, "right_hip_roll": 99.10, "right_hip_yaw": 40.18,
    "right_knee": 99.10, "right_ankle_pitch": 28.50, "right_ankle_roll": 28.50,
    "waist_yaw": 40.18, "waist_roll": 28.50, "waist_pitch": 28.50,
    "left_shoulder_pitch": 14.25, "left_shoulder_roll": 14.25,
    "left_shoulder_yaw": 14.25, "left_elbow": 14.25,
    "left_wrist_roll": 14.25, "left_wrist_pitch": 16.78, "left_wrist_yaw": 16.78,
    "right_shoulder_pitch": 14.25, "right_shoulder_roll": 14.25,
    "right_shoulder_yaw": 14.25, "right_elbow": 14.25,
    "right_wrist_roll": 14.25, "right_wrist_pitch": 16.78, "right_wrist_yaw": 16.78,
}
ISAACLAB_KD_29DOF: Dict[str, float] = {
    "left_hip_pitch": 2.558, "left_hip_roll": 6.309, "left_hip_yaw": 2.558,
    "left_knee": 6.309, "left_ankle_pitch": 1.814, "left_ankle_roll": 1.814,
    "right_hip_pitch": 2.558, "right_hip_roll": 6.309, "right_hip_yaw": 2.558,
    "right_knee": 6.309, "right_ankle_pitch": 1.814, "right_ankle_roll": 1.814,
    "waist_yaw": 2.558, "waist_roll": 1.814, "waist_pitch": 1.814,
    "left_shoulder_pitch": 0.907, "left_shoulder_roll": 0.907,
    "left_shoulder_yaw": 0.907, "left_elbow": 0.907,
    "left_wrist_roll": 0.907, "left_wrist_pitch": 1.068, "left_wrist_yaw": 1.068,
    "right_shoulder_pitch": 0.907, "right_shoulder_roll": 0.907,
    "right_shoulder_yaw": 0.907, "right_elbow": 0.907,
    "right_wrist_roll": 0.907, "right_wrist_pitch": 1.068, "right_wrist_yaw": 1.068,
}

# Torque limits: config-name -> max torque in Nm (SPEC 2.2)
# ---------------------------------------------------------------------------
TORQUE_LIMITS_29DOF: Dict[str, float] = {
    "left_hip_pitch": 88.0,
    "left_hip_roll": 139.0,
    "left_hip_yaw": 88.0,
    "left_knee": 139.0,
    "left_ankle_pitch": 50.0,
    "left_ankle_roll": 50.0,
    "right_hip_pitch": 88.0,
    "right_hip_roll": 139.0,
    "right_hip_yaw": 88.0,
    "right_knee": 139.0,
    "right_ankle_pitch": 50.0,
    "right_ankle_roll": 50.0,
    "waist_yaw": 88.0,
    "waist_roll": 50.0,
    "waist_pitch": 50.0,
    "left_shoulder_pitch": 25.0,
    "left_shoulder_roll": 25.0,
    "left_shoulder_yaw": 25.0,
    "left_elbow": 25.0,
    "left_wrist_roll": 25.0,
    "left_wrist_pitch": 5.0,
    "left_wrist_yaw": 5.0,
    "right_shoulder_pitch": 25.0,
    "right_shoulder_roll": 25.0,
    "right_shoulder_yaw": 25.0,
    "right_elbow": 25.0,
    "right_wrist_roll": 25.0,
    "right_wrist_pitch": 5.0,
    "right_wrist_yaw": 5.0,
}

TORQUE_LIMITS_23DOF: Dict[str, float] = {
    "left_hip_pitch": 88.0,
    "left_hip_roll": 139.0,
    "left_hip_yaw": 88.0,
    "left_knee": 139.0,
    "left_ankle_pitch": 50.0,
    "left_ankle_roll": 50.0,
    "right_hip_pitch": 88.0,
    "right_hip_roll": 139.0,
    "right_hip_yaw": 88.0,
    "right_knee": 139.0,
    "right_ankle_pitch": 50.0,
    "right_ankle_roll": 50.0,
    "torso": 88.0,
    "left_shoulder_pitch": 25.0,
    "left_shoulder_roll": 25.0,
    "left_shoulder_yaw": 25.0,
    "left_elbow_pitch": 25.0,
    "left_elbow_roll": 25.0,
    "right_shoulder_pitch": 25.0,
    "right_shoulder_roll": 25.0,
    "right_shoulder_yaw": 25.0,
    "right_elbow_pitch": 25.0,
    "right_elbow_roll": 25.0,
}

# ---------------------------------------------------------------------------
# Velocity limits: config-name -> max velocity in rad/s
# Source: unitree_rl_lab ImplicitActuatorCfg velocity_limit_sim values
#   N7520-14.3 (hip_pitch, hip_yaw, waist_yaw): 32 rad/s
#   N7520-22.5 (hip_roll, knee): 20 rad/s
#   N5020-16   (shoulder, elbow, wrist_roll, ankle, waist_roll/pitch): 37 rad/s
#   N5020-16-parallel (ankle, 23-DOF only): 30 rad/s
#   W4010-25   (wrist_pitch, wrist_yaw, 29-DOF only): 22 rad/s
# ---------------------------------------------------------------------------
VELOCITY_LIMITS_29DOF: Dict[str, float] = {
    "left_hip_pitch": 32.0,
    "left_hip_roll": 20.0,
    "left_hip_yaw": 32.0,
    "left_knee": 20.0,
    "left_ankle_pitch": 37.0,
    "left_ankle_roll": 37.0,
    "right_hip_pitch": 32.0,
    "right_hip_roll": 20.0,
    "right_hip_yaw": 32.0,
    "right_knee": 20.0,
    "right_ankle_pitch": 37.0,
    "right_ankle_roll": 37.0,
    "waist_yaw": 32.0,
    "waist_roll": 37.0,
    "waist_pitch": 37.0,
    "left_shoulder_pitch": 37.0,
    "left_shoulder_roll": 37.0,
    "left_shoulder_yaw": 37.0,
    "left_elbow": 37.0,
    "left_wrist_roll": 37.0,
    "left_wrist_pitch": 22.0,
    "left_wrist_yaw": 22.0,
    "right_shoulder_pitch": 37.0,
    "right_shoulder_roll": 37.0,
    "right_shoulder_yaw": 37.0,
    "right_elbow": 37.0,
    "right_wrist_roll": 37.0,
    "right_wrist_pitch": 22.0,
    "right_wrist_yaw": 22.0,
}

VELOCITY_LIMITS_23DOF: Dict[str, float] = {
    "left_hip_pitch": 32.0,
    "left_hip_roll": 20.0,
    "left_hip_yaw": 32.0,
    "left_knee": 20.0,
    "left_ankle_pitch": 30.0,
    "left_ankle_roll": 30.0,
    "right_hip_pitch": 32.0,
    "right_hip_roll": 20.0,
    "right_hip_yaw": 32.0,
    "right_knee": 20.0,
    "right_ankle_pitch": 30.0,
    "right_ankle_roll": 30.0,
    "torso": 32.0,
    "left_shoulder_pitch": 37.0,
    "left_shoulder_roll": 37.0,
    "left_shoulder_yaw": 37.0,
    "left_elbow_pitch": 37.0,
    "left_elbow_roll": 37.0,
    "right_shoulder_pitch": 37.0,
    "right_shoulder_roll": 37.0,
    "right_shoulder_yaw": 37.0,
    "right_elbow_pitch": 37.0,
    "right_elbow_roll": 37.0,
}

# ---------------------------------------------------------------------------
# DDS/IDL name <-> config name mappings (SPEC section 2.2, 2.3)
# ---------------------------------------------------------------------------
_DDS_TO_CONFIG_29DOF: Dict[str, str] = {
    "L_LEG_HIP_PITCH": "left_hip_pitch",
    "L_LEG_HIP_ROLL": "left_hip_roll",
    "L_LEG_HIP_YAW": "left_hip_yaw",
    "L_LEG_KNEE": "left_knee",
    "L_LEG_ANKLE_PITCH": "left_ankle_pitch",
    "L_LEG_ANKLE_ROLL": "left_ankle_roll",
    "R_LEG_HIP_PITCH": "right_hip_pitch",
    "R_LEG_HIP_ROLL": "right_hip_roll",
    "R_LEG_HIP_YAW": "right_hip_yaw",
    "R_LEG_KNEE": "right_knee",
    "R_LEG_ANKLE_PITCH": "right_ankle_pitch",
    "R_LEG_ANKLE_ROLL": "right_ankle_roll",
    "WAIST_YAW": "waist_yaw",
    "WAIST_ROLL": "waist_roll",
    "WAIST_PITCH": "waist_pitch",
    "L_SHOULDER_PITCH": "left_shoulder_pitch",
    "L_SHOULDER_ROLL": "left_shoulder_roll",
    "L_SHOULDER_YAW": "left_shoulder_yaw",
    "L_ELBOW": "left_elbow",
    "L_WRIST_ROLL": "left_wrist_roll",
    "L_WRIST_PITCH": "left_wrist_pitch",
    "L_WRIST_YAW": "left_wrist_yaw",
    "R_SHOULDER_PITCH": "right_shoulder_pitch",
    "R_SHOULDER_ROLL": "right_shoulder_roll",
    "R_SHOULDER_YAW": "right_shoulder_yaw",
    "R_ELBOW": "right_elbow",
    "R_WRIST_ROLL": "right_wrist_roll",
    "R_WRIST_PITCH": "right_wrist_pitch",
    "R_WRIST_YAW": "right_wrist_yaw",
}

_DDS_TO_CONFIG_23DOF: Dict[str, str] = {
    "L_LEG_HIP_PITCH": "left_hip_pitch",
    "L_LEG_HIP_ROLL": "left_hip_roll",
    "L_LEG_HIP_YAW": "left_hip_yaw",
    "L_LEG_KNEE": "left_knee",
    "L_LEG_ANKLE_PITCH": "left_ankle_pitch",
    "L_LEG_ANKLE_ROLL": "left_ankle_roll",
    "R_LEG_HIP_PITCH": "right_hip_pitch",
    "R_LEG_HIP_ROLL": "right_hip_roll",
    "R_LEG_HIP_YAW": "right_hip_yaw",
    "R_LEG_KNEE": "right_knee",
    "R_LEG_ANKLE_PITCH": "right_ankle_pitch",
    "R_LEG_ANKLE_ROLL": "right_ankle_roll",
    "TORSO": "torso",
    "L_SHOULDER_PITCH": "left_shoulder_pitch",
    "L_SHOULDER_ROLL": "left_shoulder_roll",
    "L_SHOULDER_YAW": "left_shoulder_yaw",
    "L_ELBOW_PITCH": "left_elbow_pitch",
    "L_ELBOW_ROLL": "left_elbow_roll",
    "R_SHOULDER_PITCH": "right_shoulder_pitch",
    "R_SHOULDER_ROLL": "right_shoulder_roll",
    "R_SHOULDER_YAW": "right_shoulder_yaw",
    "R_ELBOW_PITCH": "right_elbow_pitch",
    "R_ELBOW_ROLL": "right_elbow_roll",
}

# Reverse mappings (MuJoCo name -> config name)
_MUJOCO_TO_CONFIG_29DOF: Dict[str, str] = {v: k for k, v in G1_29DOF_MUJOCO_JOINTS.items()}
_MUJOCO_TO_CONFIG_23DOF: Dict[str, str] = {v: k for k, v in G1_23DOF_MUJOCO_JOINTS.items()}


def _get_joints_for_variant(variant: str) -> List[str]:
    """Return the joint name list for a given variant."""
    if variant == "g1_29dof":
        return G1_29DOF_JOINTS
    elif variant == "g1_23dof":
        return G1_23DOF_JOINTS
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Must be 'g1_29dof' or 'g1_23dof'.")


def resolve_joint_name(name: str, variant: str = "g1_29dof") -> str:
    """Resolve any joint name form to the canonical config-name.

    Accepts:
    - Config names: ``left_hip_pitch`` -> ``left_hip_pitch``
    - MuJoCo names: ``left_hip_pitch_joint`` -> ``left_hip_pitch``
    - DDS/IDL names: ``L_LEG_HIP_PITCH`` -> ``left_hip_pitch``

    Raises ``ValueError`` if the name is not recognized.
    """
    joints = _get_joints_for_variant(variant)

    # Already a config name?
    if name in joints:
        return name

    # MuJoCo name?
    mujoco_map = _MUJOCO_TO_CONFIG_29DOF if variant == "g1_29dof" else _MUJOCO_TO_CONFIG_23DOF
    if name in mujoco_map:
        return mujoco_map[name]

    # DDS/IDL name?
    dds_map = _DDS_TO_CONFIG_29DOF if variant == "g1_29dof" else _DDS_TO_CONFIG_23DOF
    if name in dds_map:
        return dds_map[name]

    raise ValueError(
        f"Unrecognized joint name {name!r} for variant {variant!r}. "
        f"Valid config names: {joints}"
    )


# ============================================================================
# Section 2: Configuration Dataclasses (PLAN_METAL Task 2.5)
# ============================================================================

@dataclass
class RobotConfig:
    variant: str = "g1_29dof"
    idl_mode: int = 0


@dataclass
class PolicyConfig:
    default_policy: Optional[str] = None  # stance/velocity tracking policy path
    default_ka: Union[float, List[float]] = 0.3  # action scale for default policy
    format: Optional[str] = None  # "isaaclab", "beyondmimic", or None (auto-detect)
    observed_joints: Optional[List[str]] = None
    controlled_joints: Optional[List[str]] = None
    use_onnx_metadata: bool = True
    use_estimator: bool = True


@dataclass
class ControlConfig:
    policy_frequency: int = 50
    sim_frequency: int = 200
    kp: Union[float, List[float]] = 100.0
    kd: Union[float, List[float]] = 10.0
    ka: Union[float, List[float]] = 0.5
    kd_damp: float = 5.0
    q_home: Optional[Dict[str, float]] = None


@dataclass
class SafetyConfig:
    joint_position_limits: bool = True
    joint_velocity_limits: bool = True
    torque_limits: bool = True


@dataclass
class NetworkConfig:
    interface: str = "auto"
    domain_id: int = 1


@dataclass
class ViewerConfig:
    enabled: bool = True
    sync: bool = True


@dataclass
class LoggingConfig:
    enabled: bool = True
    format: str = "hdf5"
    compression: str = "gzip"
    log_frequency: int = 50


@dataclass
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ============================================================================
# Section 3: Configuration Loading and Validation
# ============================================================================

_SECTION_CLASSES: Dict[str, type] = {
    "robot": RobotConfig,
    "policy": PolicyConfig,
    "control": ControlConfig,
    "safety": SafetyConfig,
    "network": NetworkConfig,
    "viewer": ViewerConfig,
    "logging": LoggingConfig,
}


def _dict_to_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    known_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return cls(**filtered)


def _validate_config(cfg: Config) -> None:
    """Validate a loaded Config, raising ValueError on problems."""
    # Variant
    if cfg.robot.variant not in ("g1_29dof", "g1_23dof"):
        raise ValueError(
            f"Invalid variant: {cfg.robot.variant!r}. Must be 'g1_29dof' or 'g1_23dof'."
        )

    # IDL mode
    if cfg.robot.idl_mode not in (0, 1):
        raise ValueError(
            f"Invalid idl_mode: {cfg.robot.idl_mode}. Must be 0 or 1."
        )

    variant = cfg.robot.variant
    joints = _get_joints_for_variant(variant)

    # Resolve and validate joint name lists
    if cfg.policy.controlled_joints is not None:
        cfg.policy.controlled_joints = [
            resolve_joint_name(n, variant) for n in cfg.policy.controlled_joints
        ]

    if cfg.policy.observed_joints is not None:
        cfg.policy.observed_joints = [
            resolve_joint_name(n, variant) for n in cfg.policy.observed_joints
        ]

    # Gain list length validation
    n_controlled = len(cfg.policy.controlled_joints) if cfg.policy.controlled_joints else len(joints)
    for gain_name in ("kp", "kd", "ka"):
        val = getattr(cfg.control, gain_name)
        if isinstance(val, list) and len(val) != n_controlled:
            raise ValueError(
                f"Length of {gain_name} list ({len(val)}) does not match "
                f"number of controlled joints ({n_controlled})."
            )

    # Frequency divisibility
    if cfg.control.sim_frequency % cfg.control.policy_frequency != 0:
        raise ValueError(
            f"sim_frequency ({cfg.control.sim_frequency}) must be evenly "
            f"divisible by policy_frequency ({cfg.control.policy_frequency})."
        )

    # Logging format
    if cfg.logging.format not in ("hdf5", "npz"):
        raise ValueError(
            f"Invalid logging format: {cfg.logging.format!r}. Must be 'hdf5' or 'npz'."
        )


def load_config(path: str) -> Config:
    """Load and validate a Config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()
    for section_name, cls in _SECTION_CLASSES.items():
        if section_name in raw and raw[section_name] is not None:
            setattr(cfg, section_name, _dict_to_dataclass(cls, raw[section_name]))

    _validate_config(cfg)
    return cfg


def merge_configs(base: Config, override: Config) -> Config:
    """Merge override into base. Non-None override values win."""
    result = copy.deepcopy(base)
    for section_name in _SECTION_CLASSES:
        base_section = getattr(result, section_name)
        override_section = getattr(override, section_name)
        for f in fields(base_section):
            override_val = getattr(override_section, f.name)
            if override_val is not None:
                setattr(base_section, f.name, override_val)
    return result


def apply_cli_overrides(config: Config, args) -> None:
    """Apply CLI argument overrides to a loaded Config (in-place).

    Supported overrides:
        --robot   -> config.robot.variant
    """
    robot_variant = getattr(args, "robot", None)
    if robot_variant is not None:
        config.robot.variant = robot_variant
        _validate_config(config)
