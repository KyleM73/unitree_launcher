"""Main control loop for the Unitree G1 humanoid.

Handles policy inference, command building (PD control law), safety integration,
velocity commands, key handling, policy reloading, and telemetry.

Supports dual-policy mode: a default policy (IsaacLab velocity tracking) for
stance/idle and an active policy (BM, IL, etc.) for execution.

Shared core between Metal and Docker plans -- input dispatch differs.
"""
from __future__ import annotations

import glob
import os
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from unitree_launcher.config import (
    BM_ACTION_SCALE_29DOF,
    Config,
    ISAACLAB_KD_29DOF,
    ISAACLAB_KP_29DOF,
    Q_HOME_29DOF,
    Q_HOME_23DOF,
    STANDBY_KP_29DOF,
    STANDBY_KD_29DOF,
)
from unitree_launcher.control.safety import ControlMode, SafetyController, SystemState
from unitree_launcher.policy.base import PolicyInterface
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.policy.observations import ObservationBuilder
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

if TYPE_CHECKING:
    from unitree_launcher.datalog.logger import DataLogger


class Controller:
    """Main control loop: reads state, runs policy, sends commands.

    Supports dual-policy mode:

    - **Default policy** (IsaacLab velocity tracking): runs in IDLE/STOPPED
      states with zero velocity command for balanced stance.
    - **Active policy** (``--policy``): runs in RUNNING state (BM, IL, etc.).

    If no default policy is provided, IDLE/STOPPED uses static PD hold
    at home pose (standby gains).

    Supports mode sequences (e.g. gantry): an ordered list of
    ``(ControlMode, duration)`` pairs that auto-advance through
    DAMPING, INTERPOLATE, HOLD phases without requiring a policy.

    Args:
        robot: Robot backend (sim or real).
        safety: Safety state machine.
        config: Full configuration.
        policy: Active neural network policy (the ``--policy``).
            ``None`` when running a mode sequence without a policy.
        joint_mapper: Joint ordering mapper for the active policy.
            ``None`` when running a mode sequence without a policy.
        obs_builder: Observation builder for the active policy
            (``None`` for BeyondMimic, which builds its own).
        logger: Optional data logger.
        policy_dir: Optional directory of ONNX files for ``-/=`` key cycling.
        max_steps: Auto-terminate after this many RUNNING steps (0 = disabled).
        max_duration: Auto-terminate after this many seconds (0 = disabled).
        default_policy: Optional IsaacLab policy for stance/velocity tracking.
        default_obs_builder: Observation builder for the default policy.
        default_joint_mapper: Joint mapper for the default policy.
        mode_sequence: Ordered list of ``(ControlMode, duration_seconds)``
            pairs.  When set, the control loop auto-advances through these
            modes and stops after the last one completes.
        interp_target_q: Target joint positions for INTERPOLATE mode
            (robot-native ordering, all DOFs).
        interp_kp_end: Target Kp gains for INTERPOLATE mode end state.
        interp_kd_end: Target Kd gains for INTERPOLATE mode end state.
        interp_waypoints: Optional pre-computed waypoints for INTERPOLATE
            mode, shape ``(N, n_dof)``.  When provided, waypoints replace
            the naive linear blend for position targets.
    """

    def __init__(
        self,
        robot: RobotInterface,
        safety: SafetyController,
        config: Config,
        policy: Optional[PolicyInterface] = None,
        joint_mapper: Optional[JointMapper] = None,
        obs_builder: Optional[ObservationBuilder] = None,
        logger: Optional['DataLogger'] = None,
        policy_dir: Optional[str] = None,
        max_steps: int = 0,
        max_duration: float = 0.0,
        default_policy: Optional[PolicyInterface] = None,
        default_obs_builder: Optional[ObservationBuilder] = None,
        default_joint_mapper: Optional[JointMapper] = None,
        mode_sequence: Optional[List[Tuple[ControlMode, float]]] = None,
        interp_target_q: Optional[np.ndarray] = None,
        interp_kp_end: Optional[np.ndarray] = None,
        interp_kd_end: Optional[np.ndarray] = None,
        interp_waypoints: Optional[np.ndarray] = None,
    ):
        self.robot = robot
        self.policy = policy  # Active policy (None for mode-sequence-only)
        self.safety = safety
        self.joint_mapper = joint_mapper  # Active policy mapper
        self.obs_builder = obs_builder  # Active policy obs builder
        self.config = config
        self._logger = logger

        # Default policy for stance (optional)
        self._default_policy = default_policy
        self._default_obs_builder = default_obs_builder
        self._default_mapper = default_joint_mapper

        # Control parameters
        self._dt = 1.0 / config.control.policy_frequency
        self._kd_damp = config.control.kd_damp
        self._time_step = 0  # BeyondMimic trajectory index

        # Active policy gains (only when policy + mapper provided)
        if joint_mapper is not None:
            n_ctrl = joint_mapper.n_controlled
            self._kp = self._expand_gain(config.control.kp, n_ctrl)
            self._kd = self._expand_gain(config.control.kd, n_ctrl)
            self._ka = self._expand_gain(config.control.ka, n_ctrl)

            # IsaacLab per-joint gains and action scale for active IsaacLab policies.
            # Action scale = 0.25 * effort_limit / stiffness (same formula as BM).
            ctrl_joints = joint_mapper.controlled_joints
            self._isaaclab_kp = np.array(
                [ISAACLAB_KP_29DOF.get(j, 40.0) for j in ctrl_joints]
            )
            self._isaaclab_kd = np.array(
                [ISAACLAB_KD_29DOF.get(j, 2.0) for j in ctrl_joints]
            )
            self._isaaclab_ka = np.array(
                [BM_ACTION_SCALE_29DOF.get(j, 0.439) for j in ctrl_joints]
            )

            # BM per-joint action scale fallback (used when ONNX metadata absent)
            self._bm_ka = np.array(
                [BM_ACTION_SCALE_29DOF.get(j, 0.439) for j in ctrl_joints]
            )

            # Active policy home positions in controlled-joint order
            variant = config.robot.variant
            q_home_dict = config.control.q_home
            if q_home_dict is None:
                q_home_dict = Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
            self._q_home = np.array(
                [q_home_dict[j] for j in joint_mapper.controlled_joints]
            )
        else:
            self._kp = np.array([])
            self._kd = np.array([])
            self._ka = np.array([])
            self._bm_ka = np.array([])
            self._q_home = np.array([])

        # Default policy gains and home (if available)
        if default_joint_mapper is not None:
            n_def = default_joint_mapper.n_controlled
            def_joints = default_joint_mapper.controlled_joints
            q_home_dict = config.control.q_home
            if q_home_dict is None:
                variant = config.robot.variant
                q_home_dict = Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
            self._default_kp = np.array(
                [ISAACLAB_KP_29DOF.get(j, 40.0) for j in def_joints]
            )
            self._default_kd = np.array(
                [ISAACLAB_KD_29DOF.get(j, 2.5) for j in def_joints]
            )
            self._default_ka = self._expand_gain(config.policy.default_ka, n_def)
            self._default_q_home = np.array(
                [q_home_dict[j] for j in def_joints]
            )
            self._default_last_action = np.zeros(n_def)
        else:
            self._default_last_action = np.zeros(0)

        # Control mode: depends on mode_sequence or default policy availability
        if mode_sequence is not None:
            self._control_mode = mode_sequence[0][0]
        elif default_policy:
            self._control_mode = ControlMode.DEFAULT
        else:
            self._control_mode = ControlMode.HOLD

        # Velocity command (thread-safe via copy-on-write)
        self._velocity_command = np.zeros(3)
        self._vel_lock = threading.Lock()

        # Telemetry (thread-safe via dict replacement)
        self._telemetry: Dict = {
            "policy_hz": 0.0,
            "sim_hz": 0.0,
            "inference_ms": 0.0,
            "loop_ms": 0.0,
            "base_height": 0.0,
            "base_vel": np.zeros(3),
            "system_state": "idle",
            "step_count": 0,
        }
        self._telemetry_lock = threading.Lock()

        # Control thread
        self._running = False
        self._policy_active = False
        self._thread: Optional[threading.Thread] = None

        # Policy directory cycling
        self._policy_dir = policy_dir
        self._policy_files: List[str] = []
        self._policy_index: int = 0
        if policy_dir and os.path.isdir(policy_dir):
            self._policy_files = sorted(
                glob.glob(os.path.join(policy_dir, "*.onnx"))
            )

        # Pre-loaded policy objects: {path: (policy, mapper, obs_builder)}
        # Populated by main() at startup for instant switching.
        self._preloaded_policies: Dict[str, Tuple] = {}

        # Auto-termination
        self._max_steps = max_steps
        self._max_duration = max_duration

        # Mode sequence (gantry, etc.)
        self._mode_sequence = mode_sequence
        self._seq_index = 0
        self._seq_step = 0
        self._last_seq_cmd: Optional[RobotCommand] = None  # for slew-rate limiter

        # Interpolation state for INTERPOLATE mode
        self._interp_start_q: Optional[np.ndarray] = None
        self._interp_target_q = interp_target_q
        self._interp_kp_end = interp_kp_end
        self._interp_kd_end = interp_kd_end
        self._interp_waypoints = interp_waypoints

        # State estimator (optional, for real robot or estimator-in-the-loop)
        self._estimator = None

        # Monotonic sim-step counter (incremented after every robot.step()).
        # Read by the main thread for video recording -- no lock needed
        # since int reads are atomic in CPython.
        self.sim_step_count: int = 0

        # Stdout status
        self._last_print_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the control loop thread (idempotent).

        The loop runs continuously: in IDLE/STOPPED it runs the default
        policy (stance) or static hold; in RUNNING it executes the active
        policy.  Policy state is reset on the IDLE->RUNNING transition.

        When a mode_sequence is set, the loop auto-advances through the
        sequence modes and stops after the last one completes.
        """
        if self._running:
            return
        self._running = True
        self._time_step = 0
        self._policy_active = False

        if self._mode_sequence is not None:
            self._control_mode = self._mode_sequence[0][0]
            self._seq_index = 0
            self._seq_step = 0
            self._interp_start_q = None
            self._last_seq_cmd = None
        elif self._default_policy:
            self._control_mode = ControlMode.DEFAULT
        else:
            self._control_mode = ControlMode.HOLD

        if self.policy is not None:
            self.policy.reset()
        if self._default_policy is not None:
            self._default_policy.reset()
            self._default_last_action = np.zeros(
                self._default_mapper.n_controlled
            )
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the control loop and wait for the thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_estimator(self, estimator) -> None:
        """Attach a state estimator for base position/velocity.

        Must be called before ``start()``.  The estimator's ``update()``
        and ``populate_robot_state()`` are called each control tick so
        that ``base_position`` / ``base_velocity`` are filled before
        observation building.

        Args:
            estimator: A ``StateEstimator`` instance (or any object with
                ``update(robot_state)`` and
                ``populate_robot_state(robot_state) -> RobotState``).
        """
        self._estimator = estimator

    def set_velocity_command(
        self, vx: float, vy: float, yaw_rate: float
    ) -> None:
        """Set velocity command (thread-safe)."""
        with self._vel_lock:
            self._velocity_command = np.array([vx, vy, yaw_rate])

    def get_velocity_command(self) -> np.ndarray:
        """Get current velocity command (thread-safe).

        Returns:
            ``[vx, vy, yaw_rate]``
        """
        with self._vel_lock:
            return self._velocity_command.copy()

    def get_telemetry(self) -> dict:
        """Get latest telemetry (thread-safe, non-blocking)."""
        with self._telemetry_lock:
            return dict(self._telemetry)

    def reload_policy(self, policy_path: str) -> None:
        """Switch to a pre-loaded active policy by path.

        Looks up the policy in ``_preloaded_policies`` (populated at startup)
        and swaps in the pre-built policy, joint mapper, obs builder, and
        gain arrays.  Falls back to loading from disk if not pre-loaded.

        Temporarily stops the control thread to swap objects, then restores
        the previous running/safety state so the mode carries over.

        Raises:
            ValueError: If the path is invalid or model cannot be loaded.
        """
        prev_safety_state = self.safety.state
        was_running = self._running
        if was_running:
            self.safety.stop()
            self.stop()

        entry = self._preloaded_policies.get(policy_path)
        if entry is not None:
            policy, mapper, obs_builder = entry
            self.policy = policy
            self.joint_mapper = mapper
            self.obs_builder = obs_builder
        else:
            # Not pre-loaded — try loading into the current policy object.
            self.policy.load(policy_path)

        # Rebuild active-policy gains from (possibly new) mapper.
        n_ctrl = self.joint_mapper.n_controlled
        ctrl_joints = self.joint_mapper.controlled_joints
        self._kp = self._expand_gain(self.config.control.kp, n_ctrl)
        self._kd = self._expand_gain(self.config.control.kd, n_ctrl)
        self._ka = self._expand_gain(self.config.control.ka, n_ctrl)
        self._isaaclab_kp = np.array(
            [ISAACLAB_KP_29DOF.get(j, 40.0) for j in ctrl_joints]
        )
        self._isaaclab_kd = np.array(
            [ISAACLAB_KD_29DOF.get(j, 2.0) for j in ctrl_joints]
        )
        self._isaaclab_ka = np.array(
            [BM_ACTION_SCALE_29DOF.get(j, 0.439) for j in ctrl_joints]
        )
        self._bm_ka = np.array(
            [BM_ACTION_SCALE_29DOF.get(j, 0.439) for j in ctrl_joints]
        )
        variant = self.config.robot.variant
        q_home_dict = self.config.control.q_home
        if q_home_dict is None:
            q_home_dict = Q_HOME_29DOF if variant == "g1_29dof" else Q_HOME_23DOF
        self._q_home = np.array(
            [q_home_dict[j] for j in ctrl_joints]
        )

        self.policy.reset()
        self._time_step = 0

        # Restore previous state so mode carries over the switch.
        if was_running:
            self.start()
            if prev_safety_state == SystemState.RUNNING:
                self.safety.start()

    def handle_key(self, key: str) -> None:
        """Handle keyboard input (from MuJoCo viewer or CLI).

        Key map (avoids MuJoCo viewer letter-key conflicts):
            Space      -- start / stop
            Backspace  -- e-stop
            Enter      -- clear e-stop
            Delete     -- reset (Fn+Backspace on Mac)
            Up/Down    -- vx +/-
            Left/Right -- vy +/-
            ,/.        -- yaw +/-
            /          -- zero velocity
            -/=        -- prev/next policy

        Called from the main thread. Thread-safe.
        """
        if key == "space":
            if self.safety.state in (SystemState.IDLE, SystemState.STOPPED):
                self.safety.start()
                # Control loop is already running (default/hold mode);
                # it will detect RUNNING and activate the active policy.
                if not self._running:
                    self.start()
            elif self.safety.state == SystemState.RUNNING:
                self.safety.stop()
        elif key == "backspace":
            self.safety.estop()
        elif key == "enter":
            self.safety.clear_estop()
        elif key == "delete":
            self.robot.reset()
            self.safety.clear_estop()
        elif key == "up":
            self._adjust_velocity(0, 0.1, -1.0, 1.0)
        elif key == "down":
            self._adjust_velocity(0, -0.1, -1.0, 1.0)
        elif key == "left":
            self._adjust_velocity(1, 0.1, -0.5, 0.5)
        elif key == "right":
            self._adjust_velocity(1, -0.1, -0.5, 0.5)
        elif key == "comma":
            self._adjust_velocity(2, 0.1, -1.0, 1.0)
        elif key == "period":
            self._adjust_velocity(2, -0.1, -1.0, 1.0)
        elif key == "slash":
            with self._vel_lock:
                self._velocity_command = np.zeros(3)
        elif key == "equal":
            self._cycle_policy(1)
        elif key == "minus":
            self._cycle_policy(-1)

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(
        self, state: RobotState, action: np.ndarray
    ) -> RobotCommand:
        """Build a RobotCommand from active policy action output.

        Both IsaacLab and BeyondMimic use the same control law from training::

            tau = Kp * (default_q + Ka * action - q) - Kd * dq

        The ONNX ``joint_pos`` / ``joint_vel`` outputs are reference trajectory
        data from constant-table lookups (observation-independent).  They feed
        the ``command`` and ``motion_anchor_*`` observation terms but are NOT
        used as PD targets -- the learned actor network output (``action``) is
        what drives the robot through the training control law.
        """
        n_total = self.joint_mapper.n_total
        n_ctrl = self.joint_mapper.n_controlled

        target_pos = np.zeros(n_total)
        target_vel = np.zeros(n_total)
        target_tau = np.zeros(n_total)
        kp = np.zeros(n_total)
        kd = np.zeros(n_total)

        ctrl_idx = self.joint_mapper.controlled_indices
        non_ctrl_idx = self.joint_mapper.non_controlled_indices

        # Determine if BeyondMimic
        from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy

        is_beyondmimic = isinstance(self.policy, BeyondMimicPolicy)

        if is_beyondmimic:
            bm: BeyondMimicPolicy = self.policy
            # Use metadata gains if available, else MTC-derived per-joint values
            ctrl_kp = bm.stiffness if bm.stiffness is not None else self._kp
            ctrl_kd = bm.damping if bm.damping is not None else self._kd
            ctrl_ka = (
                bm.action_scale if bm.action_scale is not None else self._bm_ka
            )

            # Training control law: target = default_q + Ka * action
            ctrl_target_pos = bm.default_joint_pos + ctrl_ka * action
            ctrl_target_vel = np.zeros(n_ctrl)
        else:
            ctrl_kp = self._isaaclab_kp
            ctrl_kd = self._isaaclab_kd
            ctrl_ka = self._isaaclab_ka

            ctrl_target_pos = self._q_home + ctrl_ka * action
            ctrl_target_vel = np.zeros(n_ctrl)

        target_pos[ctrl_idx] = ctrl_target_pos
        target_vel[ctrl_idx] = ctrl_target_vel
        kp[ctrl_idx] = ctrl_kp
        kd[ctrl_idx] = ctrl_kd

        # Non-controlled joints: damping mode
        if len(non_ctrl_idx) > 0:
            target_pos[non_ctrl_idx] = state.joint_positions[non_ctrl_idx]
            kd[non_ctrl_idx] = self._kd_damp

        return RobotCommand(
            joint_positions=target_pos,
            joint_velocities=target_vel,
            joint_torques=target_tau,
            kp=kp,
            kd=kd,
        )

    def _build_default_command(
        self, state: RobotState, action: np.ndarray
    ) -> RobotCommand:
        """Build a RobotCommand from default policy action output.

        Uses IsaacLab training gains (ISAACLAB_KP/KD_29DOF) and
        config.policy.default_ka, mapped through the default policy's
        joint mapper.
        """
        mapper = self._default_mapper
        n_total = mapper.n_total

        target_pos = np.zeros(n_total)
        target_vel = np.zeros(n_total)
        target_tau = np.zeros(n_total)
        kp = np.zeros(n_total)
        kd = np.zeros(n_total)

        ctrl_idx = mapper.controlled_indices
        non_ctrl_idx = mapper.non_controlled_indices

        ctrl_target_pos = self._default_q_home + self._default_ka * action
        target_pos[ctrl_idx] = ctrl_target_pos
        kp[ctrl_idx] = self._default_kp
        kd[ctrl_idx] = self._default_kd

        if len(non_ctrl_idx) > 0:
            target_pos[non_ctrl_idx] = state.joint_positions[non_ctrl_idx]
            kd[non_ctrl_idx] = self._kd_damp

        return RobotCommand(
            joint_positions=target_pos,
            joint_velocities=target_vel,
            joint_torques=target_tau,
            kp=kp,
            kd=kd,
        )

    def _build_hold_command(self, state: RobotState) -> RobotCommand:
        """Build command to hold the robot at home pose (static PD).

        Uses MTC StandbyController gains (Kp=150-350 legs, 40 arms;
        Kd=5 legs, 10 knee, 3 arms) from config/g1/controllers.yaml.
        Fallback when no default policy is loaded.
        """
        n_total = self.joint_mapper.n_total
        ctrl_idx = self.joint_mapper.controlled_indices
        non_ctrl_idx = self.joint_mapper.non_controlled_indices

        target_pos = np.zeros(n_total)
        target_vel = np.zeros(n_total)
        target_tau = np.zeros(n_total)
        kp = np.zeros(n_total)
        kd = np.zeros(n_total)

        target_pos[ctrl_idx] = self._q_home

        ctrl_joints = self.joint_mapper.controlled_joints
        standby_kp = np.array(
            [STANDBY_KP_29DOF.get(j, 100.0) for j in ctrl_joints],
            dtype=np.float64,
        )
        standby_kd = np.array(
            [STANDBY_KD_29DOF.get(j, 5.0) for j in ctrl_joints],
            dtype=np.float64,
        )
        kp[ctrl_idx] = standby_kp
        kd[ctrl_idx] = standby_kd

        # Non-controlled joints: damping mode
        if len(non_ctrl_idx) > 0:
            target_pos[non_ctrl_idx] = state.joint_positions[non_ctrl_idx]
            kd[non_ctrl_idx] = self._kd_damp

        return RobotCommand(
            joint_positions=target_pos,
            joint_velocities=target_vel,
            joint_torques=target_tau,
            kp=kp,
            kd=kd,
        )

    def _build_damping_command(self, state: RobotState) -> RobotCommand:
        """Build pure velocity-damping command (kp=0, per-joint kd, pos=current).

        Uses per-joint kd from ``interp_kd_end`` (IsaacLab training gains)
        when available, otherwise falls back to the flat ``kd_damp`` config
        value.  Per-joint gains keep damping forces within each actuator's
        ``forcerange`` — a flat kd=5 exceeds the wrist actuators' ±5 Nm
        limit at modest velocities, causing oscillation.
        """
        n = self.robot.n_dof
        if self._interp_kd_end is not None:
            kd = self._interp_kd_end.copy()
        else:
            kd = np.full(n, self._kd_damp)
        return RobotCommand(
            joint_positions=state.joint_positions.copy(),
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=np.zeros(n),
            kd=kd,
        )

    def _build_interpolate_command(self, state: RobotState) -> RobotCommand:
        """Build cosine-interpolated command from start state to target.

        Ramps both position (start_q -> target_q) and gains
        (0 -> kp_end / kd_end) using ``smooth_alpha()`` from gantry.py.
        """
        from unitree_launcher.gantry import smooth_alpha

        _, duration = self._mode_sequence[self._seq_index]
        t = (self._seq_step + 1) * self._dt
        alpha = smooth_alpha(t, duration)

        n = self.robot.n_dof
        if (
            self._interp_waypoints is not None
            and self._seq_step < len(self._interp_waypoints)
        ):
            target_q = self._interp_waypoints[self._seq_step].copy()
        else:
            target_q = (1.0 - alpha) * self._interp_start_q + alpha * self._interp_target_q
        target_kp = alpha * self._interp_kp_end
        target_kd = alpha * self._interp_kd_end

        return RobotCommand(
            joint_positions=target_q,
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=target_kp,
            kd=target_kd,
        )

    def _build_seq_hold_command(self, state: RobotState) -> RobotCommand:
        """Build hold command using interpolation target position and gains.

        Used in mode sequences — holds at ``interp_target_q`` with the
        full ``interp_kp_end`` / ``interp_kd_end`` gains.
        """
        n = self.robot.n_dof
        return RobotCommand(
            joint_positions=self._interp_target_q.copy(),
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=self._interp_kp_end.copy(),
            kd=self._interp_kd_end.copy(),
        )

    # ------------------------------------------------------------------
    # EXPERIMENTAL: slew-rate limiter for mode-sequence transitions.
    # Clamps per-tick change in position, kp, and kd to avoid
    # discontinuous jumps at mode boundaries.  ESTOP bypasses this
    # entirely (separate branch in the control loop).
    #
    # To disable: replace the _smooth_command() call in the mode-
    # sequence block with a no-op (just return cmd unchanged).
    # ------------------------------------------------------------------

    # Max per-tick deltas (at 50 Hz → per 20 ms tick)
    _SMOOTH_MAX_POS_DELTA = 0.1    # rad/tick
    _SMOOTH_MAX_KP_DELTA = 5.0     # Nm·rad⁻¹/tick
    _SMOOTH_MAX_KD_DELTA = 1.0     # Nm·s·rad⁻¹/tick

    def _smooth_command(self, desired: RobotCommand) -> RobotCommand:
        """Slew-rate-limit a command against the previous tick's command.

        On the first call (no previous command), passes through unchanged.
        """
        prev = self._last_seq_cmd
        if prev is None:
            self._last_seq_cmd = desired
            return desired

        pos = prev.joint_positions + np.clip(
            desired.joint_positions - prev.joint_positions,
            -self._SMOOTH_MAX_POS_DELTA, self._SMOOTH_MAX_POS_DELTA,
        )
        kp = prev.kp + np.clip(
            desired.kp - prev.kp,
            -self._SMOOTH_MAX_KP_DELTA, self._SMOOTH_MAX_KP_DELTA,
        )
        kd = prev.kd + np.clip(
            desired.kd - prev.kd,
            -self._SMOOTH_MAX_KD_DELTA, self._SMOOTH_MAX_KD_DELTA,
        )

        smoothed = RobotCommand(
            joint_positions=pos,
            joint_velocities=desired.joint_velocities,
            joint_torques=desired.joint_torques,
            kp=kp,
            kd=kd,
        )
        self._last_seq_cmd = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """Main control loop -- runs in a background thread.

        State machine:

        - **ESTOP**: damping mode (kp=0, kd=kd_damp)
        - **Mode sequence** (if set): auto-advance through DAMPING,
          INTERPOLATE, HOLD, etc. with timed transitions
        - **IDLE/STOPPED + DEFAULT**: default policy with zero velocity (stance)
        - **IDLE/STOPPED + HOLD**: static PD at home pose (standby gains)
        - **RUNNING + ACTIVE_POLICY**: active policy executes
            - BM: when trajectory ends, auto-return to DEFAULT + STOPPED
            - IL: runs until Space / max_steps / max_duration
        """
        n_ctrl = self.joint_mapper.n_controlled if self.joint_mapper else 0
        last_action = np.zeros(n_ctrl)
        step_count = 0
        loop_start_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            try:
                # 1. E-stop: send damping and keep sim alive
                if self.safety.state == SystemState.ESTOP:
                    state = self.robot.get_state()
                    cmd = self.safety.get_damping_command(state)
                    self.robot.send_command(cmd)
                    self.robot.step()
                    self.sim_step_count += 1
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 2. Mode sequence (gantry, etc.): auto-advance through modes
                if self._mode_sequence is not None:
                    state = self.robot.get_state()
                    if self.safety.check_state_limits(state):
                        continue

                    mode, duration = self._mode_sequence[self._seq_index]
                    total_steps = int(duration * self.config.control.policy_frequency)

                    # Build command for current mode
                    if mode == ControlMode.DAMPING:
                        cmd = self._build_damping_command(state)
                    elif mode == ControlMode.INTERPOLATE:
                        if self._interp_start_q is None:
                            self._interp_start_q = state.joint_positions.copy()
                        cmd = self._build_interpolate_command(state)
                    elif mode == ControlMode.HOLD:
                        cmd = self._build_seq_hold_command(state)
                    elif mode == ControlMode.DEFAULT and self._default_policy:
                        vel_cmd = np.zeros(3)
                        obs = self._default_obs_builder.build(
                            state, self._default_last_action, vel_cmd
                        )
                        action = self._default_policy.get_action(obs)
                        self._default_last_action = action.copy()
                        cmd = self._build_default_command(state, action)
                    else:
                        cmd = self._build_damping_command(state)

                    cmd = self._smooth_command(cmd)  # EXPERIMENTAL slew-rate limiter
                    cmd = self.safety.clamp_command(cmd)
                    self.robot.send_command(cmd)
                    self.robot.step()
                    self.sim_step_count += 1

                    self._seq_step += 1

                    # Print progress every 25 steps (0.5s at 50 Hz)
                    if self._seq_step % 25 == 0:
                        print(
                            f"[controller] mode={mode.value} "
                            f"step={self._seq_step}/{total_steps}"
                        )

                    # Check for mode transition
                    if self._seq_step >= total_steps:
                        self._seq_index += 1
                        self._seq_step = 0

                        if self._seq_index >= len(self._mode_sequence):
                            print("[controller] Mode sequence completed.")
                            self._running = False
                            break

                        next_mode, next_dur = self._mode_sequence[self._seq_index]
                        self._control_mode = next_mode
                        print(
                            f"[controller] -> {next_mode.value} "
                            f"({next_dur:.1f}s)"
                        )

                        if next_mode == ControlMode.INTERPOLATE:
                            self._interp_start_q = state.joint_positions.copy()

                    self._sleep_until_next_tick(loop_start)
                    continue

                # 3. IDLE/STOPPED (no mode_sequence): run default policy or hold
                if self.safety.state != SystemState.RUNNING:
                    self._policy_active = False
                    self._control_mode = (
                        ControlMode.DEFAULT
                        if self._default_policy
                        else ControlMode.HOLD
                    )
                    state = self.robot.get_state()
                    if self._estimator is not None:
                        self._estimator.update(state)
                        state = self._estimator.populate_robot_state(state)
                    if self.safety.check_state_limits(state):
                        continue

                    if self._control_mode == ControlMode.DEFAULT:
                        # Default policy: IL velocity tracking with zero vel
                        vel_cmd = np.zeros(3)
                        obs = self._default_obs_builder.build(
                            state, self._default_last_action, vel_cmd
                        )
                        action = self._default_policy.get_action(obs)
                        self._default_last_action = action.copy()
                        cmd = self._build_default_command(state, action)
                    else:
                        # Static PD hold at home pose
                        cmd = self._build_hold_command(state)

                    self.robot.send_command(cmd)
                    self.robot.step()
                    self.sim_step_count += 1
                    self._sleep_until_next_tick(loop_start)
                    continue

                # 4. Detect IDLE/STOPPED -> RUNNING transition
                if not self._policy_active:
                    self._policy_active = True
                    self._control_mode = ControlMode.ACTIVE_POLICY
                    self._time_step = 0
                    self.policy.reset()
                    last_action = np.zeros(self.joint_mapper.n_controlled)
                    step_count = 0
                    loop_start_time = time.perf_counter()
                    self._last_print_time = 0.0

                    from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy

                    if isinstance(self.policy, BeyondMimicPolicy):
                        bm: BeyondMimicPolicy = self.policy
                        # Prefetch reference data for time_step=0 so the
                        # observation builder has _prev_target_q/dq.
                        # No teleport -- default policy already maintained
                        # stance near the starting pose.
                        bm.prefetch_reference(0)
                        print(
                            "[controller] Active policy started (BeyondMimic)."
                        )
                    else:
                        print("[controller] Active policy started.")

                # 4. Get robot state
                state = self.robot.get_state()
                if self._estimator is not None:
                    self._estimator.update(state)
                    state = self._estimator.populate_robot_state(state)
                if self.safety.check_state_limits(state):
                    continue

                # 5. Build observation and run policy inference
                inference_start = time.perf_counter()
                if self.obs_builder is not None:
                    # IsaacLab active policy
                    vel_cmd = self.get_velocity_command()
                    obs = self.obs_builder.build(state, last_action, vel_cmd)
                    action = self.policy.get_action(obs)
                else:
                    # BeyondMimic active policy: single ONNX call per step
                    # matching motion_tracking_controller C++ sim2sim.
                    #
                    # 1. Build obs using PREVIOUS step's cached trajectory
                    # 2. Run ONNX with current time_step -> action + ref data
                    # 3. Post-increment time_step (matching C++ timeStep_++)
                    from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy

                    bm = self.policy
                    if (
                        self._estimator is None
                        and hasattr(self.robot, "get_body_state")
                        and bm.anchor_body_name
                    ):
                        # No estimator: use MuJoCo ground-truth body state.
                        anchor_pos, anchor_quat = self.robot.get_body_state(
                            bm.anchor_body_name
                        )
                    else:
                        # Estimator active (or no get_body_state): use the
                        # estimator-populated state so sim matches real.
                        anchor_pos = state.base_position
                        anchor_quat = state.imu_quaternion
                    obs = bm.build_observation(
                        state, anchor_pos, anchor_quat
                    )
                    action = bm.get_action(
                        obs, time_step=self._time_step
                    )
                    self._time_step += 1

                    # Detect end of reference trajectory
                    if self._time_step >= bm.trajectory_length:
                        print(
                            f"[controller] BeyondMimic trajectory ended at "
                            f"step {self._time_step}. Returning to "
                            f"{'default policy' if self._default_policy else 'hold'}."
                        )
                        self._control_mode = (
                            ControlMode.DEFAULT
                            if self._default_policy
                            else ControlMode.HOLD
                        )
                        self.safety.stop()
                        # Still apply this last action below

                inference_time = time.perf_counter() - inference_start

                # Safety: replace NaN/Inf in policy output with zeros
                if not np.all(np.isfinite(action)):
                    print(
                        "[controller] WARNING: NaN/Inf in policy action, "
                        "using zeros"
                    )
                    action = np.zeros_like(action)

                # 6. Build and clamp command
                cmd = self._build_command(state, action)
                cmd = self.safety.clamp_command(cmd)

                # 7. Send command
                self.robot.send_command(cmd)

                # 8. Step simulation
                self.robot.step()
                self.sim_step_count += 1

                # 9. Store for next iteration
                last_action = action.copy()
                step_count += 1

                # 10. Log (expand to fixed widths for consistent array shape)
                if self._logger is not None:
                    loop_time = time.perf_counter() - loop_start
                    log_action = self.joint_mapper.action_to_robot(action)
                    # Pad obs to fixed width so policy switches don't break logging.
                    _LOG_OBS_DIM = 160  # max(IsaacLab=99, BeyondMimic=160)
                    if obs.shape[0] < _LOG_OBS_DIM:
                        log_obs = np.zeros(_LOG_OBS_DIM, dtype=obs.dtype)
                        log_obs[:obs.shape[0]] = obs
                    else:
                        log_obs = obs
                    self._logger.log_step(
                        timestamp=time.time(),
                        robot_state=state,
                        observation=log_obs,
                        action=log_action,
                        command=cmd,
                        system_state=self.safety.state,
                        velocity_command=self.get_velocity_command(),
                        timing={
                            "inference_ms": inference_time * 1000.0,
                            "loop_ms": loop_time * 1000.0,
                        },
                    )

                # 11. Update telemetry
                self._update_telemetry(
                    state, inference_time, loop_start, step_count
                )

                # 12. Print status at 1 Hz
                now = time.perf_counter()
                if now - self._last_print_time >= 1.0:
                    vel_cmd = self.get_velocity_command()
                    mode_str = self._control_mode.value
                    print(
                        f"[controller] state={self.safety.state.value} "
                        f"mode={mode_str} step={step_count} "
                        f"policy_hz="
                        f"{1.0 / max(now - loop_start, 1e-6):.1f} "
                        f"vel_cmd=[{vel_cmd[0]:.1f}, {vel_cmd[1]:.1f}, "
                        f"{vel_cmd[2]:.1f}]"
                    )
                    self._last_print_time = now

                # 13. Check auto-termination
                if self._max_steps > 0 and step_count >= self._max_steps:
                    self.safety.stop()
                    self._running = False
                    break
                if self._max_duration > 0 and (
                    now - loop_start_time
                ) >= self._max_duration:
                    self.safety.stop()
                    self._running = False
                    break

            except Exception as exc:
                # Any exception in the control loop triggers E-stop
                print(f"[controller] EXCEPTION in control loop: {exc}")
                self.safety.estop()

            self._sleep_until_next_tick(loop_start)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sleep_until_next_tick(self, loop_start: float) -> None:
        """Sleep to maintain policy frequency."""
        elapsed = time.perf_counter() - loop_start
        remaining = self._dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _update_telemetry(
        self,
        state: RobotState,
        inference_time: float,
        loop_start: float,
        step_count: int,
    ) -> None:
        """Update telemetry dict (thread-safe via replacement)."""
        loop_time = time.perf_counter() - loop_start
        telem = {
            "policy_hz": 1.0 / max(loop_time, 1e-6),
            "sim_hz": self.config.control.sim_frequency,
            "inference_ms": inference_time * 1000.0,
            "loop_ms": loop_time * 1000.0,
            "base_height": state.base_position[2],
            "base_vel": state.base_velocity.copy(),
            "system_state": self.safety.state.value,
            "step_count": step_count,
        }
        with self._telemetry_lock:
            self._telemetry = telem

    def _adjust_velocity(
        self, index: int, delta: float, vmin: float, vmax: float
    ) -> None:
        """Adjust one component of velocity command, clamped to [vmin, vmax]."""
        with self._vel_lock:
            vc = self._velocity_command.copy()
            vc[index] = np.clip(vc[index] + delta, vmin, vmax)
            self._velocity_command = vc

    def _cycle_policy(self, direction: int) -> None:
        """Cycle to next/previous active policy in policy_dir."""
        if not self._policy_files:
            return
        self._policy_index = (self._policy_index + direction) % len(
            self._policy_files
        )
        path = self._policy_files[self._policy_index]
        try:
            self.reload_policy(path)
            print(f"[controller] Loaded policy: {os.path.basename(path)}")
        except Exception as exc:
            print(f"[controller] Failed to load policy: {exc}")

    @staticmethod
    def _expand_gain(value, n: int) -> np.ndarray:
        """Expand a scalar or list gain to an (n,) array."""
        if isinstance(value, (int, float)):
            return np.full(n, value, dtype=np.float64)
        return np.array(value, dtype=np.float64)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def control_mode(self) -> ControlMode:
        """Current control mode."""
        return self._control_mode
