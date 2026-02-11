"""Robot state/command dataclasses and abstract robot interface.

Defines the shared data structures used across sim and real robot backends.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================================
# Dataclasses (PLAN_METAL Task 2.2)
# ============================================================================

@dataclass
class RobotState:
    """Snapshot of robot state at a given time."""
    timestamp: float
    joint_positions: np.ndarray       # (N_DOF,)
    joint_velocities: np.ndarray      # (N_DOF,)
    joint_torques: np.ndarray         # (N_DOF,) estimated
    imu_quaternion: np.ndarray        # (4,) wxyz
    imu_angular_velocity: np.ndarray  # (3,)
    imu_linear_acceleration: np.ndarray  # (3,)
    base_position: np.ndarray         # (3,) world frame (sim only, NaN for real)
    base_velocity: np.ndarray         # (3,) world frame (sim only, NaN for real)

    @staticmethod
    def zeros(n_dof: int) -> RobotState:
        """Create a zero-initialized state."""
        return RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(n_dof),
            joint_velocities=np.zeros(n_dof),
            joint_torques=np.zeros(n_dof),
            imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # identity
            imu_angular_velocity=np.zeros(3),
            imu_linear_acceleration=np.zeros(3),
            base_position=np.zeros(3),
            base_velocity=np.zeros(3),
        )

    def copy(self) -> RobotState:
        """Return a deep copy of this state."""
        return copy.deepcopy(self)


@dataclass
class RobotCommand:
    """Command to send to the robot."""
    joint_positions: np.ndarray   # (N_DOF,) target positions
    joint_velocities: np.ndarray  # (N_DOF,) target velocities
    joint_torques: np.ndarray     # (N_DOF,) feedforward torques
    kp: np.ndarray                # (N_DOF,) position gains
    kd: np.ndarray                # (N_DOF,) velocity gains

    @staticmethod
    def damping(n_dof: int, kd: float = 5.0) -> RobotCommand:
        """Create a damping-mode command (zero positions, specified kd)."""
        return RobotCommand(
            joint_positions=np.zeros(n_dof),
            joint_velocities=np.zeros(n_dof),
            joint_torques=np.zeros(n_dof),
            kp=np.zeros(n_dof),
            kd=np.full(n_dof, kd),
        )


# ============================================================================
# Abstract Robot Interface (PLAN_METAL Task 2.3)
# ============================================================================

class RobotInterface(ABC):
    """Abstract interface for robot backends (sim and real)."""

    @abstractmethod
    def connect(self) -> None:
        """Initialize connection to the robot."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Clean up and disconnect."""
        ...

    @abstractmethod
    def get_state(self) -> RobotState:
        """Read the current robot state."""
        ...

    @abstractmethod
    def send_command(self, cmd: RobotCommand) -> None:
        """Send a command to the robot."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance simulation by one step (no-op for real robot)."""
        ...

    @abstractmethod
    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Reset the robot to an initial state."""
        ...

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        ...
