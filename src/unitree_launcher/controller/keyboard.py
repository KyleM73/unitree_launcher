"""Keyboard input controller.

Receives key strings from GLFW viewer or viser web UI via ``push_key()``.
Produces incremental velocity commands and discrete commands.
"""
from __future__ import annotations

import numpy as np

from unitree_launcher.controller.input import InputController


class KeyboardInput(InputController):
    """Keyboard input via GLFW key callbacks or viser web buttons.

    Keys are pushed externally via ``push_key(key_name)``. The controller
    translates them into velocity adjustments and discrete commands.

    Key mapping:
        space      -> [START] or [STOP] (toggles)
        backspace  -> [SHUTDOWN]
        enter      -> [ESTOP_CLEAR]
        delete     -> [RESET]
        up/down    -> vx +/- 0.1
        left/right -> vy +/- 0.1
        comma/period -> yaw +/- 0.1
        slash      -> zero velocity
        equal      -> [POLICY_NEXT]
        minus      -> [POLICY_PREV]
    """

    def __init__(self):
        self._velocity = np.zeros(3)
        self._commands: set = set()

    def push_key(self, key: str) -> None:
        """Push a key event (called from main thread)."""
        if key == "space":
            # Toggle — Runtime._handle_commands decides START vs STOP
            # based on actual safety state
            self._commands.add("[TOGGLE]")
        elif key == "backspace":
            self._commands.add("[SHUTDOWN]")
        elif key == "enter":
            self._commands.add("[ESTOP_CLEAR]")
        elif key == "delete":
            self._commands.add("[RESET]")
        elif key == "up":
            self._velocity[0] = np.clip(self._velocity[0] + 0.1, -1.0, 1.0)
        elif key == "down":
            self._velocity[0] = np.clip(self._velocity[0] - 0.1, -1.0, 1.0)
        elif key == "left":
            self._velocity[1] = np.clip(self._velocity[1] + 0.1, -0.5, 0.5)
        elif key == "right":
            self._velocity[1] = np.clip(self._velocity[1] - 0.1, -0.5, 0.5)
        elif key == "comma":
            self._velocity[2] = np.clip(self._velocity[2] + 0.1, -1.0, 1.0)
        elif key == "period":
            self._velocity[2] = np.clip(self._velocity[2] - 0.1, -1.0, 1.0)
        elif key == "slash":
            self._velocity[:] = 0.0
        elif key == "equal":
            self._commands.add("[POLICY_NEXT]")
        elif key == "minus":
            self._commands.add("[POLICY_PREV]")

    def update(self) -> None:
        """No polling needed — keys are pushed externally."""
        pass

    def get_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def get_commands(self) -> set:
        cmds = self._commands.copy()
        self._commands.clear()
        return cmds
