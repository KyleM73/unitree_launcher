"""IsaacLab policy backend.

Loads an IsaacLab-exported ONNX model and runs inference to produce joint
position offsets (actions) from an observation vector built by
:class:`~src.policy.observations.ObservationBuilder`.

Control law (applied by the controller, not here):
    target_pos = q_home + Ka * action
    tau = Kp * (target_pos - q) - Kd * qdot
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from src.policy.base import PolicyInterface
from src.policy.joint_mapper import JointMapper


class IsaacLabPolicy(PolicyInterface):
    """ONNX-based IsaacLab locomotion policy.

    Args:
        joint_mapper: Maps between robot-native and policy joint orderings.
        obs_dim: Expected observation dimension (from ObservationBuilder).
    """

    def __init__(self, joint_mapper: JointMapper, obs_dim: int) -> None:
        self._mapper = joint_mapper
        self._obs_dim = obs_dim
        self._session: ort.InferenceSession | None = None
        self._last_action = np.zeros(joint_mapper.n_controlled, dtype=np.float32)

    def load(self, path: str) -> None:
        """Load ONNX model and validate dimensions.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If input/output dimensions do not match expectations.
        """
        try:
            session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ValueError(f"Failed to load ONNX model from {path}: {exc}") from exc

        # Validate input dimension
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"Expected 1 input, got {len(inputs)}: "
                f"{[i.name for i in inputs]}"
            )
        model_obs_dim = inputs[0].shape[1]
        if model_obs_dim != self._obs_dim:
            raise ValueError(
                f"ONNX observation dim {model_obs_dim} != expected {self._obs_dim}"
            )

        # Validate output dimension
        outputs = session.get_outputs()
        model_action_dim = outputs[0].shape[1]
        if model_action_dim != self._mapper.n_controlled:
            raise ValueError(
                f"ONNX action dim {model_action_dim} != "
                f"n_controlled {self._mapper.n_controlled}"
            )

        self._session = session
        self._last_action = np.zeros(self._mapper.n_controlled, dtype=np.float32)

    def reset(self) -> None:
        """Clear cached last action."""
        self._last_action = np.zeros(self._mapper.n_controlled, dtype=np.float32)

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Run ONNX inference.

        Args:
            observation: Shape ``(obs_dim,)`` from ObservationBuilder.

        Returns:
            Action array of shape ``(n_controlled,)``.

        Raises:
            RuntimeError: If no model has been loaded.
        """
        if self._session is None:
            raise RuntimeError("No policy loaded. Call load() first.")

        obs_input = observation.astype(np.float32).reshape(1, -1)
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        result = self._session.run([output_name], {input_name: obs_input})

        action = result[0].flatten().astype(np.float64)
        self._last_action = action.copy()
        return action

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._mapper.n_controlled

    @property
    def last_action(self) -> np.ndarray:
        """Most recent action output (for feeding back into observations)."""
        return self._last_action.copy()
