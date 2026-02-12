"""BeyondMimic policy backend.

Loads a BeyondMimic-exported ONNX model (with embedded metadata) and runs
inference to produce joint position offsets.  Unlike IsaacLab, BeyondMimic:

- Receives a ``time_step`` input indexing into the reference trajectory.
- Outputs ``target_q`` / ``target_dq`` reference joint states.
- Uses a velocity-error control law:
      tau = Kp * (target_q + Ka * action - q) - Kd * (qdot - target_dq)
- Builds its own observation vector (metadata-driven structure).
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from src.policy.base import PolicyInterface
from src.policy.joint_mapper import JointMapper
from src.robot.base import RobotState


class BeyondMimicPolicy(PolicyInterface):
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
    ) -> None:
        self._mapper = joint_mapper
        self._obs_dim = obs_dim
        self._use_onnx_metadata = use_onnx_metadata
        self._session: ort.InferenceSession | None = None

        n = joint_mapper.n_controlled
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
        self._stiffness: Optional[np.ndarray] = None
        self._damping: Optional[np.ndarray] = None
        self._action_scale: Optional[np.ndarray] = None
        self._anchor_body_name: str = ""
        self._body_names: List[str] = []
        self._anchor_body_idx: int = 0

        # ONNX output name mapping (handles naming differences)
        self._output_names: List[str] = []
        self._action_idx: int = 0
        self._target_q_idx: int = 1
        self._target_dq_idx: int = 2
        self._body_pos_w_idx: Optional[int] = None
        self._body_quat_w_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # PolicyInterface implementation
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
        if model_action_dim != self._mapper.n_controlled:
            raise ValueError(
                f"ONNX action dim {model_action_dim} != "
                f"n_controlled {self._mapper.n_controlled}"
            )

        self._session = session

        # Apply metadata
        self._apply_metadata()
        self.reset()

    def reset(self) -> None:
        """Clear cached actions, targets, and body reference outputs."""
        n = self._mapper.n_controlled
        self._target_q = np.zeros(n)
        self._target_dq = np.zeros(n)
        self._prev_target_q = np.zeros(n)
        self._prev_target_dq = np.zeros(n)
        self._prev_action = np.zeros(n)
        self._prev_body_pos_w = None
        self._prev_body_quat_w = None

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

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._mapper.n_controlled

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
    def stiffness(self) -> Optional[np.ndarray]:
        """Per-joint stiffness (Kp) from metadata, or None."""
        return self._stiffness

    @property
    def damping(self) -> Optional[np.ndarray]:
        """Per-joint damping (Kd) from metadata, or None."""
        return self._damping

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
        elif term == "motion_anchor_pos_b":
            return self._compute_motion_anchor_pos_b(anchor_pos_w, anchor_quat_w)
        elif term == "motion_anchor_ori_b":
            return self._compute_motion_anchor_ori_b(anchor_quat_w)
        elif term == "base_lin_vel":
            return robot_state.base_velocity.copy()
        elif term == "base_ang_vel":
            return robot_state.imu_angular_velocity.copy()
        elif term == "joint_pos":
            jp = self._mapper.robot_to_observation(robot_state.joint_positions)
            return jp - self._default_joint_pos
        elif term == "joint_vel":
            return self._mapper.robot_to_observation(robot_state.joint_velocities)
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

        if self._use_onnx_metadata:
            if "joint_stiffness" in md:
                self._stiffness = self._parse_float_csv(md["joint_stiffness"])
            if "joint_damping" in md:
                self._damping = self._parse_float_csv(md["joint_damping"])
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

    @staticmethod
    def _safe_parse(s: str) -> Any:
        """Safely parse a metadata string value using ast.literal_eval.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Failed to parse metadata value: {s!r}") from exc


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

    Convention: ``[col0[0], col0[1], col0[2], col1[0], col1[1], col1[2]]``.
    Returns shape ``(6,)``.
    """
    R = quat_to_rotation_matrix(quat_wxyz)
    return np.concatenate([R[:, 0], R[:, 1]])


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
