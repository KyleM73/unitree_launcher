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
# Dataclasses
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
    def damping(n_dof: int, kd: float = 8.0) -> RobotCommand:
        """Create a damping-mode command (Kp=0, Kd=8, matching RoboJuDo)."""
        return RobotCommand(
            joint_positions=np.zeros(n_dof),
            joint_velocities=np.zeros(n_dof),
            joint_torques=np.zeros(n_dof),
            kp=np.zeros(n_dof),
            kd=np.full(n_dof, kd),
        )


# ============================================================================
# Abstract Robot Interface
# ============================================================================

class RobotInterface(ABC):
    """Abstract interface for robot backends.

    Three implementations:

    - **SimRobot** — Pure MuJoCo simulation (sim/eval modes).
    - **RealRobot** — Onboard C++ unitree_cpp (real mode).
    - **MirrorRobot** — Read-only Python DDS subscriber (mirror mode).

    The uniform control loop is::

        robot.connect()
        robot.send_command(cmd)   # queue/publish command
        robot.step()              # advance physics (sim) or no-op (real)

    +-----------------------+-------------------------------+-------------------------------+
    | Method                | SimRobot                      | RealRobot                     |
    +=======================+===============================+===============================+
    | ``connect()``         | No-op (pure MuJoCo).          | Initializes C++ binding,      |
    |                       |                               | verifies connection.          |
    +-----------------------+-------------------------------+-------------------------------+
    | ``get_state()``       | Reads qpos/qvel directly      | Reads state via C++ binding.  |
    |                       | (no sensor lag). IMU from      | ``base_position/velocity``    |
    |                       | sensordata.                   | are NaN (no world frame).     |
    +-----------------------+-------------------------------+-------------------------------+
    | ``send_command(cmd)`` | Stores cmd; applied at next   | Sends via C++ binding.        |
    |                       | ``step()`` call.              | C++ handles 500 Hz republish. |
    +-----------------------+-------------------------------+-------------------------------+
    | ``step()``            | Acquires lock, applies cmd    | **No-op.** Hardware runs its  |
    |                       | to position actuators, runs   | own PD loop.                  |
    |                       | N physics substeps.           |                               |
    +-----------------------+-------------------------------+-------------------------------+
    | ``reset()``           | Resets MuJoCo qpos/qvel/ctrl. | Logs warning. Cannot reset    |
    |                       |                               | physical hardware.            |
    +-----------------------+-------------------------------+-------------------------------+
    | ``graceful_shutdown`` | Calls ``disconnect()``.       | Sends damping commands        |
    |                       |                               | (Kp=0, Kd=8), then disconns. |
    +-----------------------+-------------------------------+-------------------------------+
    """

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
        """Advance one policy timestep.

        - **SimRobot**: acquires lock, applies pending command to position
          actuators, runs N physics substeps.
        - **RealRobot**: no-op — hardware runs its own PD.
        """
        ...

    @abstractmethod
    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Reset the robot to an initial state."""
        ...

    def graceful_shutdown(self, damping_duration: float = 0.5) -> None:
        """Shut down safely.

        - **SimRobot** (default): calls ``disconnect()``.
        - **RealRobot**: sends damping commands (Kp=0, Kd=8) for
          *damping_duration* seconds, then disconnects.
        """
        self.disconnect()

    def set_safety(self, safety) -> None:
        """Attach a safety controller for watchdog/E-stop."""
        pass

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        ...
