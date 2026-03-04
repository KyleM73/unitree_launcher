"""Joint mapping between robot-native and policy ordering.

Handles reordering, subsetting, template-filling, and round-tripping between
the robot's native joint order and the policy's expected order.

Supports any-to-any DOF mapping: a 12-DOF policy can map onto a 29-DOF
robot, with uncontrolled joints filled from a template (e.g., default pose).

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
        policy_joints: Joints used by the policy, in policy order.
            If None, defaults to all robot joints (identity mapping).

    Raises:
        ValueError: Unknown joint name, duplicate, or empty joint list.
    """

    def __init__(
        self,
        robot_joints: List[str],
        policy_joints: Optional[List[str]] = None,
    ):
        self._robot_joints = list(robot_joints)
        self._n_robot = len(self._robot_joints)
        robot_set = set(self._robot_joints)

        # If policy_joints is None: identity mapping (all robot joints)
        if policy_joints is not None:
            self._policy_joints = list(policy_joints)
        else:
            self._policy_joints = list(self._robot_joints)

        self._n_policy = len(self._policy_joints)

        # --- Validation ---
        if self._n_policy == 0:
            raise ValueError("policy_joints cannot be empty")
        if len(set(self._policy_joints)) != self._n_policy:
            raise ValueError("Duplicate joint in policy_joints")
        for name in self._policy_joints:
            if name not in robot_set:
                raise ValueError(f"Unknown joint: {name!r}")

        # --- Build index arrays ---
        robot_index = {name: i for i, name in enumerate(self._robot_joints)}

        # policy_idx[k] = robot index of the k-th policy joint
        self._policy_indices = np.array(
            [robot_index[n] for n in self._policy_joints], dtype=np.intp
        )

        # Non-controlled: robot indices NOT in the policy set
        policy_set = set(self._policy_indices.tolist())
        self._non_controlled_indices = np.array(
            [i for i in range(self._n_robot) if i not in policy_set],
            dtype=np.intp,
        )


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def policy_joints(self) -> List[str]:
        """Policy joint names in policy order."""
        return self._policy_joints

    @property
    def robot_joints(self) -> List[str]:
        """All robot joint names in robot-native order."""
        return self._robot_joints

    @property
    def n_policy(self) -> int:
        """Number of policy joints."""
        return self._n_policy

    @property
    def n_robot(self) -> int:
        """Total robot joints."""
        return self._n_robot

    @property
    def non_controlled_indices(self) -> np.ndarray:
        """Robot indices NOT in the policy joint set."""
        return self._non_controlled_indices

    @property
    def policy_indices(self) -> np.ndarray:
        """Indices into robot state array for policy joints, in policy order."""
        return self._policy_indices

    # ------------------------------------------------------------------
    # Core mapping methods
    # ------------------------------------------------------------------

    def robot_to_policy(self, robot_data: np.ndarray) -> np.ndarray:
        """Extract policy joints from full robot array, in policy order.

        Args:
            robot_data: Shape ``(n_robot,)`` in robot-native order.

        Returns:
            Shape ``(n_policy,)`` in policy order.
        """
        return robot_data[self._policy_indices]

    def policy_to_robot(
        self,
        policy_data: np.ndarray,
        template: Optional[np.ndarray] = None,
        default: float = 0.0,
    ) -> np.ndarray:
        """Map policy data back to full robot array.

        Uncontrolled joints are filled from ``template`` (if provided) or
        with ``default`` scalar. This is the key feature: a 12-DOF policy
        action can be embedded into a 29-DOF robot command with non-controlled
        joints holding their default positions.

        Args:
            policy_data: Shape ``(n_policy,)`` in policy order.
            template: Shape ``(n_robot,)`` — values for non-policy joints.
                Typically the robot's default standing pose.
            default: Scalar fill value when template is None.

        Returns:
            Shape ``(n_robot,)`` in robot-native order.
        """
        if template is not None:
            result = template.copy()
        else:
            result = np.full(self._n_robot, default, dtype=np.float64)
        result[self._policy_indices] = policy_data
        return result

    def fit_gains(
        self,
        policy_gains: np.ndarray,
        default: float = 0.0,
    ) -> np.ndarray:
        """Map per-joint policy gains to full robot DOF array.

        Args:
            policy_gains: Shape ``(n_policy,)`` — gains for policy joints.
            default: Fill value for non-policy joints (usually 0.0).

        Returns:
            Shape ``(n_robot,)`` with policy gains at controlled indices,
            ``default`` elsewhere.
        """
        result = np.full(self._n_robot, default, dtype=np.float64)
        result[self._policy_indices] = policy_gains
        return result

