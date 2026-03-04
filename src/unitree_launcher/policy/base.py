"""Policy base class and format detection.

Each policy subclass owns its observation format, control law, action
scaling, and gains. The Runtime calls ``policy.step(state, vel_cmd)``
and receives a complete ``RobotCommand``.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import onnxruntime as ort

from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotState


class Policy(ABC):
    """Base policy. Subclasses own observations, control law, and gains.

    The Runtime only calls:
    - ``step(state, velocity_command) -> RobotCommand``
    - ``reset()``
    - ``load(path)``
    - Properties: ``stiffness``, ``damping``, ``default_pos``
    """

    def __init__(self, mapper: JointMapper, n_dof: int):
        self._mapper = mapper
        self._n_dof = n_dof  # Total robot DOF (for building full commands)
        self._session: Optional[ort.InferenceSession] = None

        # Action state
        self._last_action = np.zeros(mapper.n_policy, dtype=np.float64)

        # Smoothing config (subclasses can override)
        self._action_beta = 1.0      # EMA factor (1.0 = no smoothing)
        self._action_clip: Optional[float] = None
        self._action_scale = np.ones(mapper.n_policy, dtype=np.float64)

        # Gains (subclasses set these from metadata or defaults)
        self._kp = np.full(mapper.n_policy, 100.0, dtype=np.float64)
        self._kd = np.full(mapper.n_policy, 10.0, dtype=np.float64)

        # Default/home positions in policy joint order
        self._default_pos_policy = np.zeros(mapper.n_policy, dtype=np.float64)
        # Default positions in full robot order (for template filling)
        self._default_pos_robot = np.zeros(n_dof, dtype=np.float64)

        # Damping for non-controlled joints
        self._kd_damp = 8.0

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from ONNX file."""
        ...

    @abstractmethod
    def step(self, state: RobotState, velocity_command: np.ndarray) -> RobotCommand:
        """One policy tick: obs -> inference -> control law -> command.

        Args:
            state: Current robot state.
            velocity_command: [vx, vy, yaw_rate] from input controller.

        Returns:
            Complete RobotCommand with positions, gains for all DOF.
        """
        ...

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._last_action[:] = 0.0

    def warmup(self, state: RobotState, velocity_command: np.ndarray) -> None:
        """Run a full step without advancing trajectory state.

        Warms up the ONNX session and initializes observation history.
        The returned command is discarded. Subclasses override to
        protect additional internal counters.
        """
        self.step(state, velocity_command)

    # ------------------------------------------------------------------
    # Gain properties (full robot DOF, for env.set_gains on policy switch)
    # ------------------------------------------------------------------

    @property
    def stiffness(self) -> np.ndarray:
        """Per-joint Kp in full robot DOF (non-controlled = 0)."""
        return self._mapper.fit_gains(self._kp, default=0.0)

    @property
    def damping(self) -> np.ndarray:
        """Per-joint Kd in full robot DOF (non-controlled = kd_damp)."""
        return self._mapper.fit_gains(self._kd, default=self._kd_damp)

    @property
    def default_pos(self) -> np.ndarray:
        """Default standing pose in full robot DOF (for prepare phase)."""
        return self._default_pos_robot.copy()

    @property
    def starting_pos(self) -> np.ndarray:
        """Position the policy expects the robot to be in at activation.

        Returns ``default_pos`` for all policies.
        """
        return self.default_pos

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        """Expected observation vector length (subclasses override)."""
        return 0

    @property
    def action_dim(self) -> int:
        """Action vector length."""
        return self._mapper.n_policy

    @property
    def last_action(self) -> np.ndarray:
        """Copy of the last action (for observation feedback)."""
        return self._last_action.copy()

    # ------------------------------------------------------------------
    # Shared infrastructure
    # ------------------------------------------------------------------

    def _run_inference(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        """Run ONNX inference."""
        if self._session is None:
            raise RuntimeError("No model loaded. Call load() first.")
        inputs = {"obs": obs.astype(np.float32).reshape(1, -1)}
        if "time_step" in kwargs:
            inputs["time_step"] = np.array(
                [[kwargs["time_step"]]], dtype=np.float32
            )
        results = self._session.run(None, inputs)
        return results[0].flatten().astype(np.float64)

    def _smooth_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing, clipping, and scaling (RoboJuDo pattern).

        Order: EMA -> clip -> scale (matching RoboJuDo base_policy.py).
        """
        # 1. EMA smoothing
        action = (
            (1.0 - self._action_beta) * self._last_action
            + self._action_beta * raw_action
        )
        self._last_action = action.copy()

        # 2. Clip
        if self._action_clip is not None:
            action = np.clip(action, -self._action_clip, self._action_clip)

        # 3. Scale
        action = action * self._action_scale

        return action

    def _build_command(
        self,
        state: RobotState,
        target_q_robot: np.ndarray,
        kp_robot: np.ndarray,
        kd_robot: np.ndarray,
    ) -> RobotCommand:
        """Build RobotCommand from full-DOF targets and gains.

        Non-controlled joints get target_pos = current_pos, kp = 0,
        kd = kd_damp (velocity damping only).

        Args:
            state: Current robot state (for non-controlled joint positions).
            target_q_robot: Shape ``(n_dof,)`` — target positions in robot order.
            kp_robot: Shape ``(n_dof,)`` — Kp gains in robot order.
            kd_robot: Shape ``(n_dof,)`` — Kd gains in robot order.
        """
        # Non-controlled joints: hold current position with damping
        non_ctrl = self._mapper.non_controlled_indices
        if len(non_ctrl) > 0:
            target_q_robot[non_ctrl] = state.joint_positions[non_ctrl]
            kp_robot[non_ctrl] = 0.0
            kd_robot[non_ctrl] = self._kd_damp

        return RobotCommand(
            joint_positions=target_q_robot,
            joint_velocities=np.zeros(self._n_dof),
            joint_torques=np.zeros(self._n_dof),
            kp=kp_robot,
            kd=kd_robot,
        )



def detect_policy_format(onnx_path: str) -> str:
    """Auto-detect policy format from ONNX model structure.

    If the model has a ``time_step`` input, it is a BeyondMimic policy.
    Otherwise it is an IsaacLab policy.
    """
    try:
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise ValueError(f"Failed to load ONNX model: {exc}") from exc
    input_names = [inp.name for inp in session.get_inputs()]
    if "time_step" in input_names:
        return "beyondmimic"
    return "isaaclab"
