"""BeyondMimic policy backend.

Loads a BeyondMimic-exported ONNX model (with embedded metadata) and runs
inference.  Unlike IsaacLab, BeyondMimic:

- Receives a ``time_step`` input indexing into the reference trajectory.
- Outputs reference ``joint_pos`` / ``joint_vel`` and body poses from
  constant-table lookups (observation-independent) — these feed the
  ``command`` and ``motion_anchor_*`` observation terms.
- The actor network outputs ``actions`` which drive the robot through the
  same training control law as IsaacLab:
      tau = Kp * (default_q + Ka * action - q) - Kd * dq
- Builds its own observation vector (metadata-driven structure).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from unitree_launcher.config import (
    BM_ACTION_SCALE_29DOF,
    ISAACLAB_KD_29DOF,
    ISAACLAB_KP_29DOF,
    Q_HOME_29DOF,
    Q_HOME_23DOF,
)
from unitree_launcher.policy.base import Policy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotState


class BeyondMimicPolicy(Policy):
    """ONNX-based BeyondMimic motion-tracking policy.

    Args:
        joint_mapper: Maps between robot-native and policy joint orderings.
        obs_dim: Expected observation dimension.
        use_onnx_metadata: If ``True``, override config gains with ONNX metadata.
    """

    def __init__(
        self,
        joint_mapper: JointMapper,
        obs_dim: int,
        use_onnx_metadata: bool = True,
        config: Optional['Config'] = None,
    ) -> None:
        super().__init__(joint_mapper, n_dof=joint_mapper.n_robot)
        self._obs_dim = obs_dim
        self._use_onnx_metadata = use_onnx_metadata

        # Set default gains (overridden by ONNX metadata in _apply_metadata)
        policy_joints = joint_mapper.policy_joints
        self._kp = np.array(
            [ISAACLAB_KP_29DOF.get(j, 40.0) for j in policy_joints],
            dtype=np.float64,
        )
        self._kd = np.array(
            [ISAACLAB_KD_29DOF.get(j, 2.0) for j in policy_joints],
            dtype=np.float64,
        )
        self._action_scale = np.array(
            [BM_ACTION_SCALE_29DOF.get(j, 0.439) for j in policy_joints],
            dtype=np.float64,
        )

        # Home positions for full robot (for template filling)
        if config is not None:
            variant = config.robot.variant
            q_home_dict = config.control.q_home or (
                Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
            )
            self._default_pos_robot = np.array(
                [q_home_dict[j] for j in joint_mapper.robot_joints], dtype=np.float64
            )
            self._kd_damp = config.control.kd_damp

        self._start_timestep = 0  # Overridden by ONNX metadata "start_timestep"
        self._time_step = 0
        self._hold_steps = 5  # Repeat first reference frame N times before advancing
        self._hold_count = 0

        n = joint_mapper.n_policy
        self._target_q: np.ndarray = np.zeros(n)
        self._target_dq: np.ndarray = np.zeros(n)
        self._prev_target_q: np.ndarray = np.zeros(n)
        self._prev_target_dq: np.ndarray = np.zeros(n)
        self._prev_action: np.ndarray = np.zeros(n)
        self._prev_body_pos_w: Optional[np.ndarray] = None
        self._prev_body_quat_w: Optional[np.ndarray] = None

        # Metadata fields (populated by load)
        self._metadata: Dict[str, Any] = {}
        self._obs_terms: List[str] = []
        self._default_joint_pos: np.ndarray = np.zeros(n)
        self._anchor_body_name: str = ""
        self._body_names: List[str] = []
        self._anchor_body_idx: int = 0

        # Robot reference for anchor body lookups (set via set_robot)
        self._robot = None

        # Trajectory length (lazy detection)
        self._trajectory_length: Optional[int] = None

        # ONNX output name mapping (handles naming differences)
        self._output_names: List[str] = []
        self._action_idx: int = 0
        self._target_q_idx: int = 1
        self._target_dq_idx: int = 2
        self._body_pos_w_idx: Optional[int] = None
        self._body_quat_w_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Policy implementation
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load ONNX model and extract embedded metadata.

        Raises:
            ValueError: On load failure, dimension mismatch, or missing metadata.
        """
        # Load metadata first (uses onnx, not onnxruntime)
        self._metadata = self.load_metadata(path)

        # Create session
        try:
            session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ValueError(f"Failed to load ONNX model from {path}: {exc}") from exc

        # Validate inputs
        inputs = session.get_inputs()
        input_names = [i.name for i in inputs]
        if "obs" not in input_names or "time_step" not in input_names:
            raise ValueError(
                f"BeyondMimic ONNX must have 'obs' and 'time_step' inputs, "
                f"got {input_names}"
            )

        obs_input = next(i for i in inputs if i.name == "obs")
        model_obs_dim = obs_input.shape[1]
        if model_obs_dim != self._obs_dim:
            raise ValueError(
                f"ONNX observation dim {model_obs_dim} != expected {self._obs_dim}"
            )

        # Map output names (handle naming differences between real models and test fixtures)
        self._output_names = [o.name for o in session.get_outputs()]
        self._action_idx = 0  # first output is always action
        self._target_q_idx = self._find_output_index(["joint_pos", "target_q"])
        self._target_dq_idx = self._find_output_index(["joint_vel", "target_dq"])
        self._body_pos_w_idx = self._find_output_index(["body_pos_w"], required=False)
        self._body_quat_w_idx = self._find_output_index(["body_quat_w"], required=False)

        # Validate action dimension
        action_output = session.get_outputs()[self._action_idx]
        model_action_dim = action_output.shape[1]
        if model_action_dim != self._mapper.n_policy:
            raise ValueError(
                f"ONNX action dim {model_action_dim} != "
                f"n_controlled {self._mapper.n_policy}"
            )

        self._session = session

        # Apply metadata
        self._apply_metadata()
        self.reset()

    def reset(self) -> None:
        """Reset trajectory to start and clear cached state.

        Resets ``_time_step`` to 0 so the trajectory starts from the
        beginning on next activation. Initialises ``_prev_target_q`` to
        ``default_joint_pos`` for reasonable first observation.
        """
        super().reset()
        self._time_step = self._start_timestep
        self._hold_count = 0
        n = self._mapper.n_policy
        self._target_q = self._default_joint_pos.copy()
        self._target_dq = np.zeros(n)
        self._prev_target_q = self._default_joint_pos.copy()
        self._prev_target_dq = np.zeros(n)
        self._prev_action = np.zeros(n)
        self._prev_body_pos_w = None
        self._prev_body_quat_w = None

    def prefetch_reference(self, time_step: int) -> None:
        """Pre-fetch reference trajectory outputs for the given time_step.

        In training, the command observation at step N contains the reference
        for time_step N (because ``_update_command`` increments ``time_steps``
        before observations are computed).  In deployment, without pre-fetch,
        the command observation would lag by one step.

        This method runs ONNX with a zero observation to extract reference
        trajectory data (joint_pos, joint_vel, body_pos_w, body_quat_w) for
        the given time_step.  The action output is discarded.

        Only the reference caches (``_prev_target_q``, ``_prev_target_dq``,
        ``_prev_body_pos_w``, ``_prev_body_quat_w``) are updated.
        ``_prev_action`` is NOT touched so the ``actions`` observation term
        retains the correct previous-step action.
        """
        if self._session is None:
            return

        dummy_obs = np.zeros((1, self._obs_dim), dtype=np.float32)
        ts_input = np.array([[time_step]], dtype=np.float32)

        results = self._session.run(
            self._output_names,
            {"obs": dummy_obs, "time_step": ts_input},
        )

        # Update reference trajectory caches only
        self._prev_target_q = results[self._target_q_idx].flatten().astype(np.float64)
        self._prev_target_dq = results[self._target_dq_idx].flatten().astype(np.float64)

        if self._body_pos_w_idx is not None:
            self._prev_body_pos_w = results[self._body_pos_w_idx].astype(np.float64)
        if self._body_quat_w_idx is not None:
            self._prev_body_quat_w = results[self._body_quat_w_idx].astype(np.float64)

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Run ONNX inference with observation and time_step.

        Args:
            observation: Shape ``(obs_dim,)``.
            **kwargs: Must include ``time_step`` (float or ndarray).

        Returns:
            Action array of shape ``(n_controlled,)``.

        Raises:
            RuntimeError: If no model has been loaded.
            TypeError: If ``time_step`` is not provided.
        """
        if self._session is None:
            raise RuntimeError("No policy loaded. Call load() first.")

        if "time_step" not in kwargs:
            raise TypeError("BeyondMimicPolicy.get_action() requires time_step kwarg")

        time_step = kwargs["time_step"]
        obs_input = observation.astype(np.float32).reshape(1, -1)
        ts_input = np.array([[time_step]], dtype=np.float32)

        results = self._session.run(
            self._output_names,
            {"obs": obs_input, "time_step": ts_input},
        )

        action = results[self._action_idx].flatten().astype(np.float64)
        self._target_q = results[self._target_q_idx].flatten().astype(np.float64)
        self._target_dq = results[self._target_dq_idx].flatten().astype(np.float64)

        # Cache body reference outputs for next observation build
        if self._body_pos_w_idx is not None:
            self._prev_body_pos_w = results[self._body_pos_w_idx].astype(np.float64)
        if self._body_quat_w_idx is not None:
            self._prev_body_quat_w = results[self._body_quat_w_idx].astype(np.float64)

        # Cache for next observation
        self._prev_target_q = self._target_q.copy()
        self._prev_target_dq = self._target_dq.copy()
        self._prev_action = action.copy()

        return action

    def set_robot(self, robot) -> None:
        """Set robot reference for anchor body lookups (SimRobot only)."""
        self._robot = robot

    def warmup(self, state: RobotState, velocity_command: np.ndarray) -> None:
        """Run a full step without advancing trajectory counters."""
        saved_ts = self._time_step
        saved_hc = self._hold_count
        self.step(state, velocity_command)
        self._time_step = saved_ts
        self._hold_count = saved_hc

    def step(self, state: RobotState, velocity_command: np.ndarray) -> RobotCommand:
        """Full policy tick: obs -> inference -> control law -> command.

        Builds observation from robot state and cached trajectory data,
        runs ONNX inference, applies control law, returns RobotCommand.
        """
        # 1. Get anchor body state for observation
        # Use SimRobot.get_body_state if available (ground-truth in sim),
        # otherwise use state estimator / IMU values
        if (
            self._robot is not None
            and hasattr(self._robot, 'get_body_state')
            and self._anchor_body_name
        ):
            anchor_pos, anchor_quat = self._robot.get_body_state(
                self._anchor_body_name
            )
        else:
            anchor_pos = state.base_position
            anchor_quat = state.imu_quaternion

        # 2. Build observation
        obs = self.build_observation(state, anchor_pos, anchor_quat)

        # 3. Inference with time_step (hold at start frame for hold_steps)
        raw_action = self.get_action(obs, time_step=self._time_step)
        if self._hold_count < self._hold_steps:
            self._hold_count += 1
        else:
            self._time_step += 1

        # 4. Smooth (EMA -> clip -> scale applied by _smooth_action)
        action = self._smooth_action(raw_action)

        # 5. Control law: target_q = default_pos + action (Ka already in _action_scale)
        target_q_policy = self._default_joint_pos + action
        target_q_robot = self._mapper.policy_to_robot(
            target_q_policy, template=self._default_pos_robot
        )

        # 6. Build gains in robot order
        kp_robot = self._mapper.fit_gains(self._kp, default=0.0)
        kd_robot = self._mapper.fit_gains(self._kd, default=self._kd_damp)

        # 7. Build command
        return self._build_command(state, target_q_robot, kp_robot, kd_robot)

    @property
    def time_step(self) -> int:
        """Current trajectory time step."""
        return self._time_step

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    # ------------------------------------------------------------------
    # BeyondMimic-specific properties
    # ------------------------------------------------------------------

    @property
    def target_q(self) -> np.ndarray:
        """Most recent reference joint positions from ONNX output."""
        return self._target_q

    @property
    def target_dq(self) -> np.ndarray:
        """Most recent reference joint velocities from ONNX output."""
        return self._target_dq

    @property
    def metadata(self) -> Dict[str, Any]:
        """Extracted ONNX metadata dictionary."""
        return self._metadata

    @property
    def action_scale(self) -> Optional[np.ndarray]:
        """Per-joint action scale (Ka) from metadata, or None."""
        return self._action_scale

    @property
    def obs_terms(self) -> List[str]:
        """Observation term names from metadata."""
        return self._obs_terms

    @property
    def anchor_body_name(self) -> str:
        return self._anchor_body_name

    @property
    def body_names(self) -> List[str]:
        return self._body_names

    @property
    def default_joint_pos(self) -> np.ndarray:
        return self._default_joint_pos

    @property
    def trajectory_length(self) -> int:
        """Number of unique reference trajectory frames (lazy-detected).

        The ONNX model's constant-table lookup clamps at the last valid
        index, producing identical outputs for all ``time_step`` values
        beyond the trajectory.  This property detects that boundary via
        binary search (≈14 ONNX calls, runs once and is cached).
        """
        if self._trajectory_length is None:
            self._trajectory_length = self._detect_trajectory_length()
        return self._trajectory_length

    def _detect_trajectory_length(self) -> int:
        """Detect trajectory length by binary search for where reference freezes."""
        if self._session is None:
            return 0

        obs = np.zeros((1, self._obs_dim), dtype=np.float32)
        tq_name = self._output_names[self._target_q_idx]

        def get_ref(ts: int) -> np.ndarray:
            return self._session.run(
                [tq_name],
                {"obs": obs, "time_step": np.array([[ts]], dtype=np.float32)},
            )[0].flatten()

        # Find upper bound where reference is frozen
        upper = 64
        while upper < 100000:
            if np.allclose(get_ref(upper), get_ref(upper + 1), atol=1e-7):
                break
            upper *= 2
        else:
            return upper  # Never froze — treat as very long trajectory

        # Binary search: find first index where ref[i] == ref[i+1]
        lo, hi = 0, upper
        while lo < hi:
            mid = (lo + hi) // 2
            if np.allclose(get_ref(mid), get_ref(mid + 1), atol=1e-7):
                hi = mid
            else:
                lo = mid + 1

        return lo + 1  # Number of unique frames (indices 0 through lo)

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_observation(
        self,
        robot_state: RobotState,
        anchor_body_pos_w: np.ndarray,
        anchor_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        """Build observation vector from current robot state + cached outputs.

        Args:
            robot_state: Current robot state.
            anchor_body_pos_w: Anchor body world position ``(3,)``.
            anchor_body_quat_w: Anchor body world quaternion ``(4,)`` wxyz.

        Returns:
            Observation vector of shape ``(obs_dim,)``.
        """
        parts = []
        for term in self._obs_terms:
            parts.append(self._build_obs_term(
                term, robot_state, anchor_body_pos_w, anchor_body_quat_w,
            ))
        obs = np.concatenate(parts)
        return obs

    def _build_obs_term(
        self,
        term: str,
        robot_state: RobotState,
        anchor_pos_w: np.ndarray,
        anchor_quat_w: np.ndarray,
    ) -> np.ndarray:
        """Build a single observation term."""
        if term == "command":
            return np.concatenate([self._prev_target_q, self._prev_target_dq])
        elif term == "goal_joint_pos":
            return self._prev_target_q.copy()
        elif term == "goal_joint_vel":
            return self._prev_target_dq.copy()
        elif term == "projected_gravity":
            R = quat_to_rotation_matrix(robot_state.imu_quaternion)
            return R.T @ np.array([0.0, 0.0, -1.0])
        elif term == "motion_anchor_pos_b":
            return self._compute_motion_anchor_pos_b(anchor_pos_w, anchor_quat_w)
        elif term == "motion_anchor_ori_b":
            return self._compute_motion_anchor_ori_b(anchor_quat_w)
        elif term == "base_lin_vel":
            R = quat_to_rotation_matrix(robot_state.imu_quaternion)
            return R.T @ robot_state.base_velocity
        elif term == "base_ang_vel":
            return robot_state.imu_angular_velocity.copy()
        elif term == "joint_pos":
            jp = self._mapper.robot_to_policy(robot_state.joint_positions)
            return jp - self._default_joint_pos
        elif term == "joint_vel":
            return self._mapper.robot_to_policy(robot_state.joint_velocities)
        elif term == "actions":
            return self._prev_action.copy()
        else:
            raise ValueError(f"Unknown observation term: {term!r}")

    def _compute_motion_anchor_pos_b(
        self, anchor_pos_w: np.ndarray, anchor_quat_w: np.ndarray
    ) -> np.ndarray:
        """Compute motion reference anchor position in body-relative frame."""
        if self._prev_body_pos_w is None:
            return np.zeros(3)
        # Extract anchor body's reference position from previous output
        ref_pos = self._prev_body_pos_w.reshape(-1, 3)[self._anchor_body_idx]
        return compute_body_relative_position(anchor_pos_w, anchor_quat_w, ref_pos)

    def _compute_motion_anchor_ori_b(
        self, anchor_quat_w: np.ndarray
    ) -> np.ndarray:
        """Compute motion reference anchor orientation in body-relative 6D."""
        if self._prev_body_quat_w is None:
            return quat_to_6d(np.array([1.0, 0.0, 0.0, 0.0]))
        ref_quat = self._prev_body_quat_w.reshape(-1, 4)[self._anchor_body_idx]
        rel_quat = quat_multiply(quat_inverse(anchor_quat_w), ref_quat)
        return quat_to_6d(rel_quat)

    # ------------------------------------------------------------------
    # Metadata handling
    # ------------------------------------------------------------------

    @staticmethod
    def load_metadata(onnx_path: str) -> Dict[str, Any]:
        """Extract embedded metadata from an ONNX model file.

        Uses ``ast.literal_eval`` for safety (never ``eval``).

        Returns:
            Raw metadata dict with string keys and string values.

        Raises:
            ValueError: If the file cannot be loaded.
        """
        try:
            import onnx
            model = onnx.load(onnx_path)
        except Exception as exc:
            raise ValueError(f"Failed to load ONNX metadata: {exc}") from exc

        metadata: Dict[str, Any] = {}
        for prop in model.metadata_props:
            metadata[prop.key] = prop.value
        return metadata

    def _apply_metadata(self) -> None:
        """Parse and apply loaded metadata."""
        md = self._metadata

        # Required field: joint_names
        if "joint_names" not in md:
            raise ValueError("ONNX metadata missing required field: 'joint_names'")

        # Parse comma-separated lists
        self._obs_terms = self._parse_csv(md.get("observation_names", ""))
        self._body_names = self._parse_csv(md.get("body_names", ""))
        self._anchor_body_name = md.get("anchor_body_name", "")

        if self._anchor_body_name and self._body_names:
            if self._anchor_body_name in self._body_names:
                self._anchor_body_idx = self._body_names.index(self._anchor_body_name)

        # Parse numeric arrays
        if "default_joint_pos" in md:
            self._default_joint_pos = self._parse_float_csv(md["default_joint_pos"])

        # Start/end timestep (skip unstable frames at trajectory edges)
        if "start_timestep" in md:
            self._start_timestep = int(md["start_timestep"])
        if "end_timestep" in md:
            self._trajectory_length = int(md["end_timestep"])

        if self._use_onnx_metadata:
            if "joint_stiffness" in md:
                self._kp = self._parse_float_csv(md["joint_stiffness"])
            if "joint_damping" in md:
                self._kd = self._parse_float_csv(md["joint_damping"])
            if "action_scale" in md:
                self._action_scale = self._parse_float_csv(md["action_scale"])

    def _find_output_index(
        self, candidates: List[str], required: bool = True
    ) -> Optional[int]:
        """Find output index by trying multiple candidate names."""
        for name in candidates:
            if name in self._output_names:
                return self._output_names.index(name)
        if required:
            raise ValueError(
                f"ONNX model missing required output. "
                f"Tried {candidates}, found {self._output_names}"
            )
        return None

    @staticmethod
    def _parse_csv(s: str) -> List[str]:
        """Parse comma-separated string to list."""
        if not s:
            return []
        return [x.strip() for x in s.split(",") if x.strip()]

    @staticmethod
    def _parse_float_csv(s: str) -> np.ndarray:
        """Parse comma-separated floats to numpy array."""
        return np.array([float(x) for x in s.split(",")], dtype=np.float64)


# ======================================================================
# Geometry helpers (module-level for testability)
# ======================================================================


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_to_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 6D rotation (first 2 columns of rotation matrix).

    Convention matches Isaac Lab's ``matrix_from_quat(...)[..., :2].reshape(-1)``
    and MTC's row-major extraction: ``[R00, R01, R10, R11, R20, R21]``.
    Returns shape ``(6,)``.
    """
    R = quat_to_rotation_matrix(quat_wxyz)
    return R[:, :2].flatten()


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Compute the inverse (conjugate) of a unit quaternion (wxyz format).

    Returns shape ``(4,)``.
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions.

    Returns shape ``(4,)``.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def compute_body_relative_position(
    anchor_pos: np.ndarray,
    anchor_quat_wxyz: np.ndarray,
    body_pos: np.ndarray,
) -> np.ndarray:
    """Transform world-frame body position to anchor-relative body frame.

    Returns shape ``(3,)``.
    """
    R = quat_to_rotation_matrix(anchor_quat_wxyz)
    return R.T @ (body_pos - anchor_pos)


def compute_body_relative_orientation(
    anchor_quat_wxyz: np.ndarray,
    body_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Compute body orientation relative to anchor, as 6D rotation.

    Returns shape ``(6,)``.
    """
    rel_quat = quat_multiply(quat_inverse(anchor_quat_wxyz), body_quat_wxyz)
    return quat_to_6d(rel_quat)
