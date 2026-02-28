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
    """Abstract interface for robot backends (sim and real).

    The uniform control loop is::

        robot.send_command(cmd)   # queue/publish command
        robot.step()              # advance physics (sim) or no-op (real)

    Sim vs real behavioral differences:

    +-----------------------+-------------------------------+-------------------------------+
    | Method                | SimRobot                      | RealRobot                     |
    +=======================+===============================+===============================+
    | ``connect()``         | Starts DDS state publisher    | Starts DDS sub + 500 Hz cmd   |
    |                       | thread (domain_id=1).         | re-publish thread (domain=0). |
    |                       |                               | Blocks until first state msg. |
    +-----------------------+-------------------------------+-------------------------------+
    | ``get_state()``       | Reads MuJoCo sensor data      | Returns latest DDS LowState_  |
    |                       | (lags qpos by one substep).   | snapshot. Checks watchdog.    |
    |                       | ``base_position/velocity``    | ``base_position/velocity``    |
    |                       | populated from sim.           | are NaN (no world frame).     |
    +-----------------------+-------------------------------+-------------------------------+
    | ``send_command(cmd)`` | Stores cmd; applied at next   | Builds LowCmd_ IDL message,   |
    |                       | ``step()`` call.              | publishes immediately + marks |
    |                       |                               | for 500 Hz re-publish.        |
    +-----------------------+-------------------------------+-------------------------------+
    | ``step()``            | Acquires lock, applies cmd    | **No-op.** Hardware runs its  |
    |                       | to actuators, runs N physics  | own PD loop at ~500 Hz.       |
    |                       | substeps (with optional       |                               |
    |                       | substep callback).            |                               |
    +-----------------------+-------------------------------+-------------------------------+
    | ``reset()``           | Resets MuJoCo qpos/qvel/ctrl  | Logs warning. Cannot reset    |
    |                       | and runs ``mj_forward()``.    | physical hardware.            |
    +-----------------------+-------------------------------+-------------------------------+
    | ``graceful_shutdown`` | Calls ``disconnect()`` (no    | Sends damping commands for    |
    |                       | damping needed in sim).       | *damping_duration* seconds,   |
    |                       |                               | then disconnects.             |
    +-----------------------+-------------------------------+-------------------------------+
    | ``set_safety()``      | No-op (safety is sim-side).   | Stores reference for watchdog |
    |                       |                               | E-stop in ``get_state()``.    |
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

        - **SimRobot**: acquires lock, applies pending command to MuJoCo
          actuators, runs ``sim_freq // policy_freq`` physics substeps
          (calling the substep callback before each ``mj_step`` if set).
        - **RealRobot**: no-op — the hardware runs its own PD at ~500 Hz.
        """
        ...

    @abstractmethod
    def reset(self, initial_state: Optional[RobotState] = None) -> None:
        """Reset the robot to an initial state."""
        ...

    def graceful_shutdown(self, damping_duration: float = 0.5) -> None:
        """Shut down safely.

        - **SimRobot** (default): calls ``disconnect()``.
        - **RealRobot**: sends zero-torque damping commands for
          *damping_duration* seconds so the robot decelerates, then
          disconnects.
        """
        self.disconnect()

    def set_safety(self, safety) -> None:
        """Attach a safety controller for watchdog/E-stop.

        - **SimRobot** (default): no-op.
        - **RealRobot**: stores a reference used by ``get_state()`` to
          trigger E-stop if state messages go stale.
        """
        pass

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        ...
