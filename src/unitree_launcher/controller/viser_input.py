"""Viser web UI input controller.

Receives velocity slider values and policy selections from the viser
web viewer via push methods. Produces direct velocity commands and
policy switch commands.
"""
from __future__ import annotations

import numpy as np

from unitree_launcher.controller.input import InputController


class ViserInput(InputController):
    """Input from viser web viewer sliders and buttons.

    Data is pushed externally by the viser run loop:
    - ``push_velocity(vx, vy, yaw)`` from velocity sliders
    - ``push_policy_selection(path)`` from policy dropdown
    """

    def __init__(self):
        self._velocity = np.zeros(3)
        self._commands: set = set()

    def push_velocity(self, vx: float, vy: float, yaw: float) -> None:
        """Set velocity from viser sliders (direct, not incremental)."""
        self._velocity = np.array([vx, vy, yaw])

    def push_policy_selection(self, policy_path: str) -> None:
        """Push a policy selection from the viser dropdown."""
        self._commands.add(f"[POLICY_LOAD],{policy_path}")

    def update(self) -> None:
        """No polling needed — data is pushed externally."""
        pass

    def get_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def get_commands(self) -> set:
        cmds = self._commands.copy()
        self._commands.clear()
        return cmds
