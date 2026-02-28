"""Contact detection from joint torques using Schmitt trigger logic.

Uses ankle pitch torque as a ground reaction force proxy:
    F_z ≈ |tau_ankle_pitch| / lever_arm

where lever_arm ≈ 0.025 m (foot collision geoms sit 25mm below the
ankle roll joint in g1_29dof.xml).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SchmittTrigger:
    """Hysteresis trigger with debounce timing.

    The output goes high when the input exceeds ``high_threshold`` for at
    least ``high_time`` seconds, and goes low when the input drops below
    ``low_threshold`` for at least ``low_time`` seconds.
    """
    high_threshold: float
    low_threshold: float
    high_time: float = 0.02    # seconds to confirm contact onset
    low_time: float = 0.04     # seconds to confirm contact break

    def __post_init__(self) -> None:
        self._state: bool = False
        self._timer: float = 0.0

    def update(self, value: float, dt: float) -> bool:
        """Update trigger with new measurement.

        Returns:
            Current contact state after hysteresis + debounce.
        """
        if self._state:
            # Currently ON — check for falling edge
            if value < self.low_threshold:
                self._timer += dt
                if self._timer >= self.low_time:
                    self._state = False
                    self._timer = 0.0
            else:
                self._timer = 0.0
        else:
            # Currently OFF — check for rising edge
            if value > self.high_threshold:
                self._timer += dt
                if self._timer >= self.high_time:
                    self._state = True
                    self._timer = 0.0
            else:
                self._timer = 0.0

        return self._state

    @property
    def state(self) -> bool:
        return self._state

    @state.setter
    def state(self, value: bool) -> None:
        self._state = value
        self._timer = 0.0

    def reset(self, initial_state: bool = False) -> None:
        self._state = initial_state
        self._timer = 0.0


class ContactDetector:
    """Detect foot-ground contact from ankle pitch torques.

    Indices into the robot-native 29-DOF joint array:
        left_ankle_pitch  = 4
        right_ankle_pitch = 10

    Args:
        lever_arm: Distance from ankle roll axis to ground contact (m).
        high_threshold: GRF threshold (N) to enter contact.
        low_threshold: GRF threshold (N) to leave contact.
        high_time: Debounce time for contact onset (s).
        low_time: Debounce time for contact break (s).
    """
    # Robot-native indices for ankle pitch joints
    LEFT_ANKLE_PITCH_IDX = 4
    RIGHT_ANKLE_PITCH_IDX = 10

    def __init__(
        self,
        lever_arm: float = 0.025,
        high_threshold: float = 40.0,
        low_threshold: float = 15.0,
        high_time: float = 0.02,
        low_time: float = 0.04,
    ):
        self._lever_arm = lever_arm
        self._left = SchmittTrigger(high_threshold, low_threshold, high_time, low_time)
        self._right = SchmittTrigger(high_threshold, low_threshold, high_time, low_time)

    def update(self, joint_torques, dt: float):
        """Update contact state from joint torque measurements.

        Args:
            joint_torques: Full robot-native joint torque array (29,).
            dt: Time step in seconds.

        Returns:
            Tuple of (left_contact, right_contact) booleans.
        """
        left_grf = abs(joint_torques[self.LEFT_ANKLE_PITCH_IDX]) / self._lever_arm
        right_grf = abs(joint_torques[self.RIGHT_ANKLE_PITCH_IDX]) / self._lever_arm

        left = self._left.update(left_grf, dt)
        right = self._right.update(right_grf, dt)
        return left, right

    @property
    def left_contact(self) -> bool:
        return self._left.state

    @property
    def right_contact(self) -> bool:
        return self._right.state

    def reset(self, both_in_contact: bool = True) -> None:
        """Reset to initial state (default: both feet in contact)."""
        self._left.reset(both_in_contact)
        self._right.reset(both_in_contact)
