"""Abstract policy interface.

All policy backends (IsaacLab, BeyondMimic) implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import onnxruntime as ort


class PolicyInterface(ABC):
    """Abstract interface for neural network policy backends."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from file."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g., hidden states for recurrent policies)."""
        ...

    @abstractmethod
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Run inference and return action.

        Args:
            observation: Observation vector of shape ``(observation_dim,)``.
            **kwargs: Additional inputs (e.g., ``time_step`` for BeyondMimic).

        Returns:
            Action vector of shape ``(action_dim,)``.
        """
        ...

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Expected observation vector length."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action vector length."""
        ...


def detect_policy_format(onnx_path: str) -> str:
    """Auto-detect policy format from ONNX model structure.

    If the model has a ``time_step`` input, it is a BeyondMimic policy.
    Otherwise it is an IsaacLab policy.

    Raises:
        ValueError: If the file cannot be loaded.
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
