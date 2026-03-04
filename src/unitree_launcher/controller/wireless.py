"""Unitree wireless controller input.

Reads the wireless controller data from the robot's DDS state. The wireless
controller has its own radio link to the G1 — it does NOT route through the
dev machine or USB. Data arrives as raw bytes in ``robot_state.wireless_remote``
and is parsed here into button events and stick axes.

Button mapping (matching RoboJuDo):
    A        -> [SHUTDOWN] (emergency stop)
    B        -> [MOTION_FADE_OUT]
    X        -> [MOTION_FADE_IN]
    Y        -> [MOTION_RESET]
    Start    -> [START]
    Select   -> [STOP]
    R1+DPadUp   -> [POLICY_NEXT]
    R1+DPadDown -> [POLICY_PREV]

Stick mapping:
    LeftY  -> vx (forward/back)
    LeftX  -> vy (strafe)
    RightX -> yaw rate
"""
from __future__ import annotations

import struct

import numpy as np

from unitree_launcher.controller.input import InputController

# Button indices in the 16-bit button field (matching RoboJuDo)
_BTN_R1 = 0
_BTN_START = 2
_BTN_SELECT = 3
_BTN_A = 8
_BTN_B = 9
_BTN_X = 10
_BTN_Y = 11
_BTN_UP = 12
_BTN_DOWN = 14


class WirelessInput(InputController):
    """Unitree wireless controller via DDS robot state.

    Call ``parse(wireless_remote_bytes)`` each tick with the raw bytes from
    the robot state's ``wireless_remote`` field. The controller parses
    button states and stick axes into velocity commands and discrete commands.

    Designed to be registered as a callback on ``RealRobot``:
        ``robot.set_wireless_handler(wireless_input.parse)``
    """

    def __init__(self, deadzone: float = 0.05):
        self._deadzone = deadzone
        self._velocity = np.zeros(3)
        self._commands: set = set()
        self._last_buttons = np.zeros(16, dtype=bool)

    def parse(self, remote_data: bytes) -> None:
        """Parse wireless controller bytes from robot state.

        Called each control tick with ``robot_state.wireless_remote`` bytes.
        Format matches RoboJuDo's ``unitreeRemoteController.parse()``.

        Args:
            remote_data: Raw bytes from the robot's wireless remote field.
        """
        if len(remote_data) < 24:
            return

        # Parse 16-bit button field (bytes 2-3)
        keys = struct.unpack("H", remote_data[2:4])[0]
        buttons = np.array(
            [((keys & (1 << i)) >> i) for i in range(16)], dtype=bool
        )

        # Edge-triggered button commands
        pressed = buttons & ~self._last_buttons  # Rising edges
        self._last_buttons = buttons.copy()

        if pressed[_BTN_A]:
            self._commands.add("[SHUTDOWN]")
        if pressed[_BTN_B]:
            self._commands.add("[MOTION_FADE_OUT]")
        if pressed[_BTN_X]:
            self._commands.add("[MOTION_FADE_IN]")
        if pressed[_BTN_Y]:
            self._commands.add("[MOTION_RESET]")
        if pressed[_BTN_START]:
            self._commands.add("[START]")
        if pressed[_BTN_SELECT]:
            self._commands.add("[STOP]")

        # Combo: R1 + DPad for policy switching
        if buttons[_BTN_R1]:
            if pressed[_BTN_UP]:
                self._commands.add("[POLICY_NEXT]")
            if pressed[_BTN_DOWN]:
                self._commands.add("[POLICY_PREV]")

        # Parse stick axes (floats at byte offsets matching RoboJuDo)
        lx = struct.unpack("<f", remote_data[4:8])[0]    # LeftX -> vy
        rx = struct.unpack("<f", remote_data[8:12])[0]    # RightX -> yaw
        ly = struct.unpack("<f", remote_data[20:24])[0]   # LeftY -> vx

        # Apply deadzone
        if abs(lx) < self._deadzone:
            lx = 0.0
        if abs(ly) < self._deadzone:
            ly = 0.0
        if abs(rx) < self._deadzone:
            rx = 0.0

        self._velocity[0] = ly   # Forward/back
        self._velocity[1] = lx   # Strafe
        self._velocity[2] = rx   # Yaw

    def update(self) -> None:
        """No-op — data is pushed via parse() callback."""
        pass

    def get_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def get_commands(self) -> set:
        cmds = self._commands.copy()
        self._commands.clear()
        return cmds
