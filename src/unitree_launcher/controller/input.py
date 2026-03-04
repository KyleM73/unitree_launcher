"""Input controller framework.

All input sources (keyboard, viser UI, gamepad, wireless) implement
``InputController``. An ``InputManager`` merges their outputs so
Runtime receives unified velocity commands and discrete commands.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InputController(ABC):
    """Base class for all input sources."""

    @abstractmethod
    def update(self) -> None:
        """Poll or process latest input. Called each control tick."""
        ...

    @abstractmethod
    def get_velocity(self) -> np.ndarray:
        """Current velocity command [vx, vy, yaw_rate]."""
        ...

    @abstractmethod
    def get_commands(self) -> set:
        """Pending discrete commands. Consumed on read (cleared after return)."""
        ...


class InputManager:
    """Merges multiple InputControllers.

    Velocity: first controller with a non-zero command wins.
    Commands: union of all controllers' pending commands.
    """

    def __init__(self, controllers: list[InputController] | None = None):
        self.controllers = controllers or []

    def update(self) -> None:
        """Update all controllers."""
        for ctrl in self.controllers:
            ctrl.update()

    def get_velocity(self) -> np.ndarray:
        """First non-zero velocity command wins."""
        for ctrl in self.controllers:
            vel = ctrl.get_velocity()
            if np.any(vel != 0):
                return vel
        return np.zeros(3)

    def get_commands(self) -> set:
        """Union of all controllers' pending commands."""
        all_cmds = set()
        for ctrl in self.controllers:
            all_cmds.update(ctrl.get_commands())
        return all_cmds
