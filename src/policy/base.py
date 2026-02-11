"""Abstract policy interface.

All policy backends (IsaacLab, BeyondMimic) implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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
