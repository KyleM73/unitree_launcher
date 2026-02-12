"""Joint mapping between robot-native and policy ordering.

Handles reordering, subsetting, and round-tripping between the robot's native
joint order and the policy's expected observation/action order.

Incorrect joint mapping can damage the physical robot — every method here
must be unit-tested with known numerical values.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class JointMapper:
    """Maps joints between robot-native order and policy order.

    Args:
        robot_joints: All joint names in robot-native order.
        observed_joints: Joints in observation, in policy order. Defaults resolve
            per the rules below.
        controlled_joints: Joints controlled by policy, in policy order. Defaults
            resolve per the rules below.

    Resolution when args are None:
        - Both None: observe and control all joints in robot-native order.
        - Only controlled_joints: observed_joints = controlled_joints.
        - Only observed_joints: controlled_joints = all joints in robot-native order.
        - Both specified: use as given.

    Raises:
        ValueError: Unknown joint name, duplicate, or empty controlled list.
    """

    def __init__(
        self,
        robot_joints: List[str],
        observed_joints: Optional[List[str]] = None,
        controlled_joints: Optional[List[str]] = None,
    ):
        self._robot_joints = list(robot_joints)
        self._n_total = len(self._robot_joints)
        robot_set = set(self._robot_joints)

        # --- Default resolution ---
        if controlled_joints is None and observed_joints is None:
            controlled_joints = list(self._robot_joints)
            observed_joints = list(self._robot_joints)
        elif controlled_joints is not None and observed_joints is None:
            observed_joints = list(controlled_joints)
        elif observed_joints is not None and controlled_joints is None:
            controlled_joints = list(self._robot_joints)

        # --- Validation ---
        if len(controlled_joints) == 0:
            raise ValueError("controlled_joints cannot be empty")

        if len(set(observed_joints)) != len(observed_joints):
            raise ValueError("Duplicate joint in observed_joints")
        if len(set(controlled_joints)) != len(controlled_joints):
            raise ValueError("Duplicate joint in controlled_joints")

        for name in observed_joints:
            if name not in robot_set:
                raise ValueError(f"Unknown observed joint: {name!r}")
        for name in controlled_joints:
            if name not in robot_set:
                raise ValueError(f"Unknown controlled joint: {name!r}")

        self._observed_joints = list(observed_joints)
        self._controlled_joints = list(controlled_joints)

        # --- Build index arrays ---
        robot_index = {name: i for i, name in enumerate(self._robot_joints)}
        self._observed_indices = np.array(
            [robot_index[n] for n in self._observed_joints], dtype=np.intp
        )
        self._controlled_indices = np.array(
            [robot_index[n] for n in self._controlled_joints], dtype=np.intp
        )
        controlled_set = set(self._controlled_indices)
        self._non_controlled_indices = np.array(
            [i for i in range(self._n_total) if i not in controlled_set],
            dtype=np.intp,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observed_joints(self) -> List[str]:
        """Observed joint names in policy order."""
        return self._observed_joints

    @property
    def controlled_joints(self) -> List[str]:
        """Controlled joint names in policy order."""
        return self._controlled_joints

    @property
    def observed_indices(self) -> np.ndarray:
        """Indices into robot state array for observed joints, in policy order."""
        return self._observed_indices

    @property
    def controlled_indices(self) -> np.ndarray:
        """Indices into robot state array for controlled joints, in policy order."""
        return self._controlled_indices

    @property
    def non_controlled_indices(self) -> np.ndarray:
        """Indices for joints not in the controlled set."""
        return self._non_controlled_indices

    @property
    def n_observed(self) -> int:
        return len(self._observed_joints)

    @property
    def n_controlled(self) -> int:
        return len(self._controlled_joints)

    @property
    def n_total(self) -> int:
        return self._n_total

    # ------------------------------------------------------------------
    # Mapping methods
    # ------------------------------------------------------------------

    def robot_to_observation(self, robot_values: np.ndarray) -> np.ndarray:
        """Extract observed joints from full robot array, in policy order.

        Args:
            robot_values: Shape ``(n_total,)`` in robot-native order.

        Returns:
            Shape ``(n_observed,)`` in policy observation order.
        """
        return robot_values[self._observed_indices]

    def robot_to_action(self, robot_values: np.ndarray) -> np.ndarray:
        """Extract controlled joints from full robot array, in policy order.

        Args:
            robot_values: Shape ``(n_total,)`` in robot-native order.

        Returns:
            Shape ``(n_controlled,)`` in policy action order.
        """
        return robot_values[self._controlled_indices]

    def action_to_robot(
        self, policy_action: np.ndarray, default_value: float = 0.0
    ) -> np.ndarray:
        """Map policy action back to full robot array.

        Args:
            policy_action: Shape ``(n_controlled,)`` in policy order.
            default_value: Fill value for non-controlled joints.

        Returns:
            Shape ``(n_total,)`` in robot-native order.
        """
        result = np.full(self._n_total, default_value, dtype=np.float64)
        result[self._controlled_indices] = policy_action
        return result
