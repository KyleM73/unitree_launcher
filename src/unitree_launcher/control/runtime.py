"""Control pipeline for the Unitree G1 humanoid.

``Runtime.step()`` is the atomic control unit — one tick of state read,
policy inference, command building, and command send. No sleep, no threading.
Callers manage timing:

- **headless/eval**: tight ``while runtime.step(): pass`` loop (max speed)
- **gui/viser/real**: ``runtime.start_threaded()`` (daemon thread + sleep)

Supports dual-policy mode: a default policy (IsaacLab velocity tracking) for
stance/idle and an active policy (BM, IL, etc.) for execution.

"""
from __future__ import annotations

import glob
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from unitree_launcher.config import Config
from unitree_launcher.control.safety import ControlMode, SafetyController, SystemState
from unitree_launcher.policy.base import Policy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.robot.base import RobotCommand, RobotInterface, RobotState

if TYPE_CHECKING:
    from unitree_launcher.datalog.logger import DataLogger


class Runtime:
    """Control pipeline: reads state, runs policy, sends commands.

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
        logger: Optional data logger.
        policy_dir: Optional directory of ONNX files for ``-/=`` key cycling.
        default_policy: Optional IsaacLab policy for stance/velocity tracking.
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
        policy: Optional[Policy] = None,
        joint_mapper: Optional[JointMapper] = None,
        logger: Optional['DataLogger'] = None,
        policy_dir: Optional[str] = None,
        default_policy: Optional[Policy] = None,
        default_joint_mapper: Optional[JointMapper] = None,
        input_manager: Optional['InputManager'] = None,
        mode_sequence: Optional[List[Tuple[ControlMode, float]]] = None,
        interp_target_q: Optional[np.ndarray] = None,
        interp_kp_end: Optional[np.ndarray] = None,
        interp_kd_end: Optional[np.ndarray] = None,
        interp_waypoints: Optional[np.ndarray] = None,
        idle_damping: bool = False,
    ):
        self.robot = robot
        self.policy = policy
        self.safety = safety
        self.joint_mapper = joint_mapper
        self.config = config
        self._logger = logger

        self._default_policy = default_policy
        self._default_mapper = default_joint_mapper

        # Control parameters
        self._dt = 1.0 / config.control.policy_frequency
        self._kd_damp = config.control.kd_damp

        # Policies own their gains via step() — no gain arrays needed in Runtime

        # Control mode: depends on mode_sequence or default policy availability
        if mode_sequence is not None:
            self._control_mode = mode_sequence[0][0]
        elif default_policy:
            self._control_mode = ControlMode.DEFAULT
        else:
            self._control_mode = ControlMode.HOLD

        # When stopped/idle, use damping instead of default policy (gantry mode)
        self._idle_damping = idle_damping

        # Input controller system
        from unitree_launcher.controller.input import InputManager
        self._input_manager = input_manager or InputManager()

        # Telemetry (thread-safe via dict replacement)
        self._telemetry: Dict = {
            "loop_hz": 0.0,
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

        # Pre-loaded policy objects: {path: (policy, mapper)}
        # Populated by main() at startup for instant switching.
        self._preloaded_policies: Dict[str, Tuple] = {}
        self._active_policy_path: Optional[str] = None

        # Mode sequence (gantry, etc.)
        self._mode_sequence = mode_sequence
        self._initial_mode_sequence = list(mode_sequence) if mode_sequence else None
        self._seq_index = 0
        self._seq_step = 0
        self._last_seq_cmd: Optional[RobotCommand] = None  # for slew-rate limiter

        # Interpolation state for INTERPOLATE mode
        self._interp_start_q: Optional[np.ndarray] = None
        self._interp_target_q = interp_target_q
        self._interp_kp_end = interp_kp_end
        self._interp_kd_end = interp_kd_end
        self._interp_waypoints = interp_waypoints

        # Prepare mode state
        self._prepare_reset_done: bool = False
        self._prepare_done: bool = False  # True after PREPARE completes; cleared on stop/reset

        # Transition interpolation state: cosine-ramp from current position
        # to the incoming policy's starting position before handing off.
        self._transition_active = False
        self._transition_start_pos: Optional[np.ndarray] = None
        self._transition_start_kp: Optional[np.ndarray] = None
        self._transition_start_kd: Optional[np.ndarray] = None
        self._transition_target_pos: Optional[np.ndarray] = None
        self._transition_target_kp: Optional[np.ndarray] = None
        self._transition_target_kd: Optional[np.ndarray] = None
        self._transition_step = 0
        self._transition_total = config.control.transition_steps
        # Pending start gains from outgoing policy (set by reload_policy)
        self._pending_start_kp: Optional[np.ndarray] = None
        self._pending_start_kd: Optional[np.ndarray] = None

        # State estimator (optional, for real robot or estimator-in-the-loop)
        self._estimator = None

        # Monotonic sim-step counter (incremented after every robot.step()).
        # Read by the main thread for video recording -- no lock needed
        # since int reads are atomic in CPython.
        self.sim_step_count: int = 0


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize pipeline state. Does NOT start a thread.

        Call ``step()`` in a loop after this, or use ``start_threaded()``
        for gui/viser/real modes that need a background control thread.

        The pipeline runs continuously: in IDLE/STOPPED it runs the default
        policy (stance) or static hold; in RUNNING it executes the active
        policy.  Policy state is reset on the IDLE->RUNNING transition.
        """
        if self._running:
            return
        self._running = True
        self._threaded = False
        self._policy_active = False
        self._step_count = 0

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

    def start_threaded(self) -> None:
        """Initialize pipeline and start a daemon thread calling step() + sleep.

        Use for modes that need a background control loop (gui, viser, real).
        The thread maintains policy_frequency timing via sleep.
        """
        self.start()
        self._threaded = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the pipeline and wait for any thread to finish."""
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

    @property
    def input_manager(self):
        """The input manager for this runtime."""
        return self._input_manager

    def _restart_with_prepare(self) -> None:
        """Re-enter PREPARE mode sequence before activating the policy.

        Called when transitioning from STOPPED/IDLE to RUNNING and a
        mode sequence was configured. Replays PREPARE so the robot
        blends smoothly to home before the policy takes over.
        """
        self._policy_active = False
        self._prepare_done = False
        self._clear_transition()
        self._mode_sequence = list(self._initial_mode_sequence)
        self._seq_index = 0
        self._seq_step = 0
        self._last_seq_cmd = None
        self._prepare_reset_done = False
        self._control_mode = self._mode_sequence[0][0]
        if self.policy is not None:
            self.policy.reset()
        self.safety.start()
        print("[runtime] Re-entering prepare sequence")

    def _handle_commands(self, commands: set) -> None:
        """Process discrete commands from input controllers."""
        for cmd in commands:
            if cmd == "[TOGGLE]":
                if self.safety.state in (SystemState.IDLE, SystemState.STOPPED):
                    self.safety.start()
                    if not self._running:
                        self.start()
                elif self.safety.state == SystemState.RUNNING:
                    self.safety.stop()
            elif cmd == "[START]":
                if self.safety.state in (SystemState.IDLE, SystemState.STOPPED):
                    self.safety.start()
                    if not self._running:
                        self.start()
            elif cmd == "[STOP]":
                if self.safety.state == SystemState.RUNNING:
                    self.safety.stop()
            elif cmd == "[SHUTDOWN]":
                self.safety.estop()
                print("[runtime] E-stop triggered")
            elif cmd == "[ESTOP_CLEAR]":
                self.safety.clear_estop()
                print("[runtime] E-stop cleared")
            elif cmd == "[RESET]":
                if self.safety.state == SystemState.RUNNING:
                    self.safety.stop()
                self.safety.clear_estop()
                self.robot.reset()
                if self._estimator is not None:
                    self._estimator.reset()
                self._policy_active = False
                self._step_count = 0
                self._clear_transition()
                # Replay mode sequence (e.g. PREPARE) if one was configured
                if self._initial_mode_sequence is not None:
                    self._mode_sequence = list(self._initial_mode_sequence)
                    self._seq_index = 0
                    self._seq_step = 0
                    self._last_seq_cmd = None
                    self._prepare_reset_done = False
                    self._control_mode = self._mode_sequence[0][0]
                    self.safety.start()
                if self.policy is not None:
                    self.policy.reset()
                if self._default_policy is not None:
                    self._default_policy.reset()
                with self._telemetry_lock:
                    self._telemetry = {
                        "loop_hz": 0.0,
                        "sim_hz": self.config.control.sim_frequency,
                        "inference_ms": 0.0,
                        "loop_ms": 0.0,
                        "base_height": 0.0,
                        "base_vel": np.zeros(3),
                        "system_state": "stopped",
                        "step_count": 0,
                    }
                print("[runtime] Reset")
            elif cmd == "[MOTION_FADE_OUT]":
                if self.safety.state == SystemState.RUNNING:
                    self.safety.stop()
                    print("[runtime] Motion fade out (stopped)")
            elif cmd == "[MOTION_FADE_IN]":
                if self.safety.state in (SystemState.IDLE, SystemState.STOPPED):
                    self.safety.start()
                    if not self._running:
                        self.start()
                    print("[runtime] Motion fade in (started)")
            elif cmd == "[MOTION_RESET]":
                if self.safety.state == SystemState.RUNNING:
                    self.safety.stop()
                self._policy_active = False
                self._clear_transition()
                if self.policy is not None:
                    self.policy.reset()
                self.safety.start()
                if not self._running:
                    self.start()
                print("[runtime] Motion reset")
            elif cmd == "[POLICY_NEXT]":
                self._cycle_policy(1)
            elif cmd == "[POLICY_PREV]":
                self._cycle_policy(-1)
            elif cmd.startswith("[POLICY_LOAD],"):
                path = cmd.split(",", 1)[1]
                try:
                    self.reload_policy(path)
                except Exception as exc:
                    print(f"[runtime] Failed to load policy: {exc}")

    def get_telemetry(self) -> dict:
        """Get latest telemetry (thread-safe, non-blocking)."""
        with self._telemetry_lock:
            return dict(self._telemetry)

    def reload_policy(self, policy_path: str) -> None:
        """Switch to a pre-loaded active policy by path.

        Looks up the policy in ``_preloaded_policies`` (populated at startup)
        and swaps in the pre-built policy and joint mapper. Falls back to
        loading from disk if not pre-loaded.

        Safe to call from the control thread (e.g. keyboard/viser policy
        switching) — does an in-place swap without stopping the thread.

        Raises:
            ValueError: If the path is invalid or model cannot be loaded.
        """
        # Skip if already running this policy (avoids resetting mid-trajectory).
        entry = self._preloaded_policies.get(policy_path)
        if entry is not None and entry[0] is self.policy:
            return

        on_control_thread = (
            self._thread is not None
            and threading.current_thread() is self._thread
        )

        # Capture outgoing policy gains so the transition starts from them
        if on_control_thread and self.safety.state == SystemState.RUNNING:
            try:
                self._pending_start_kp = self.policy.stiffness.copy()
                self._pending_start_kd = self.policy.damping.copy()
            except Exception:
                self._pending_start_kp = None
                self._pending_start_kd = None

        prev_safety_state = self.safety.state
        if not on_control_thread and self._running:
            self.safety.stop()
            self.stop()
        if entry is not None:
            policy, mapper, *_rest = entry
            self.policy = policy
            self.joint_mapper = mapper
        else:
            self.policy.load(policy_path)

        # Mark inactive so the next step() triggers proper activation
        # (reset, prefetch_reference, gain setup, etc.)
        self._policy_active = False
        self._active_policy_path = policy_path

        if not on_control_thread and not self._running and prev_safety_state != SystemState.IDLE:
            if self._threaded:
                self.start_threaded()
            else:
                self.start()
            if prev_safety_state == SystemState.RUNNING:
                self.safety.start()


    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------


    def _build_prepare_command(self, state: RobotState) -> RobotCommand:
        """Build prepare command: linear blend from current pos to default pose.

        Re-reads actual motor position each step (tracks settling on real
        hardware). Gains are constant (default policy stiffness/damping).
        Matches RoboJuDo's prepare() exactly.
        """
        _, duration = self._mode_sequence[self._seq_index]
        total_steps = int(duration * self.config.control.policy_frequency)
        ramp_steps = int(total_steps * 0.5)

        current = state.joint_positions
        desired = self._default_policy.default_pos
        blend = min(self._seq_step / max(ramp_steps, 1), 1.0)
        target = (1.0 - blend) * current + blend * desired

        n = self.robot.n_dof
        return RobotCommand(
            joint_positions=target,
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=self._default_policy.stiffness,
            kd=self._default_policy.damping,
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
    # Policy transition: position interpolation to starting pose
    # ------------------------------------------------------------------

    def _start_transition(
        self,
        start_pos: np.ndarray,
        start_kp: np.ndarray,
        start_kd: np.ndarray,
        target_pos: np.ndarray,
        target_kp: np.ndarray,
        target_kd: np.ndarray,
    ) -> None:
        """Begin interpolating from current state to the incoming policy's starting pose.

        During transition, no policy inference runs. The robot is moved to the
        policy's ``default_pos`` with cosine-ramped gains so the policy starts
        from its expected configuration.

        If ``transition_total <= 0``, the transition is skipped (instant switch).
        """
        if self._transition_total <= 0:
            self._control_mode = ControlMode.ACTIVE_POLICY
            return
        self._transition_active = True
        self._transition_start_pos = start_pos.copy()
        self._transition_start_kp = start_kp.copy()
        self._transition_start_kd = start_kd.copy()
        self._transition_target_pos = target_pos.copy()
        self._transition_target_kp = target_kp.copy()
        self._transition_target_kd = target_kd.copy()
        self._transition_step = 0
        self._control_mode = ControlMode.TRANSITION

    def _transition_step_command(self) -> RobotCommand:
        """Generate one interpolated command toward the policy's starting pose.

        Cosine ease-in-out: ``alpha = 0.5 * (1 - cos(pi * t/N))``.
        When complete, clears transition state and sets mode to ACTIVE_POLICY.
        """
        self._transition_step += 1
        t = self._transition_step / self._transition_total
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))

        n = len(self._transition_start_pos)
        cmd = RobotCommand(
            joint_positions=(1.0 - alpha) * self._transition_start_pos + alpha * self._transition_target_pos,
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
            kp=(1.0 - alpha) * self._transition_start_kp + alpha * self._transition_target_kp,
            kd=(1.0 - alpha) * self._transition_start_kd + alpha * self._transition_target_kd,
        )

        if self._transition_step >= self._transition_total:
            self._transition_active = False
            self._transition_start_pos = None
            self._transition_target_pos = None
            self._control_mode = ControlMode.ACTIVE_POLICY

        return cmd

    def _clear_transition(self) -> None:
        """Clear all transition state."""
        self._transition_active = False
        self._transition_start_pos = None
        self._transition_target_pos = None
        self._transition_start_kp = None
        self._transition_start_kd = None
        self._transition_target_kp = None
        self._transition_target_kd = None

    # Slew-rate limiter for mode-sequence transitions.
    # Clamps per-tick change in position, kp, and kd to avoid
    # discontinuous jumps at mode boundaries.

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

    def step(self) -> bool:
        """Execute one control tick. Returns False if pipeline should stop.

        Does NOT sleep or manage timing — caller is responsible for pacing.
        Call ``start()`` before the first ``step()`` to initialize state.

        State machine branches:
        - **ESTOP**: damping mode
        - **Mode sequence**: auto-advance through DAMPING/INTERPOLATE/HOLD
        - **IDLE/STOPPED**: default policy (stance) or static hold
        - **RUNNING**: active policy
        """
        if not self._running:
            return False

        from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy

        loop_start = time.perf_counter()

        try:
            # 0. Process input from all controllers
            self._input_manager.update()
            self._handle_commands(self._input_manager.get_commands())

            # 1. E-stop: send damping and keep sim alive
            if self.safety.state == SystemState.ESTOP:
                self._clear_transition()
                state = self.robot.get_state()
                cmd = self.safety.get_damping_command(state)
                self.robot.send_command(cmd)
                self.robot.step()
                self.sim_step_count += 1
                return True

            # 2. Mode sequence (gantry, etc.): auto-advance through modes
            if self._mode_sequence is not None:
                state = self.robot.get_state()
                if self.safety.check_state_limits(state):
                    return True

                mode, duration = self._mode_sequence[self._seq_index]
                total_steps = int(duration * self.config.control.policy_frequency)

                if mode == ControlMode.DAMPING:
                    cmd = self._build_damping_command(state)
                elif mode == ControlMode.PREPARE:
                    cmd = self._build_prepare_command(state)
                    # Warmup policies during prepare
                    if self.policy is not None:
                        self.policy.warmup(state, np.zeros(3))
                    if self._default_policy is not None:
                        self._default_policy.warmup(state, np.zeros(3))
                    # At 90%: reset policies for zero-position calibration
                    reset_step = int(total_steps * 0.9)
                    if self._seq_step == reset_step and not self._prepare_reset_done:
                        self._prepare_reset_done = True
                        print(f"[runtime] Prepare: resetting zero position at step {self._seq_step}")
                        if self.policy is not None:
                            self.policy.reset()
                        if self._default_policy is not None:
                            self._default_policy.reset()
                elif mode == ControlMode.INTERPOLATE:
                    if self._interp_start_q is None:
                        self._interp_start_q = state.joint_positions.copy()
                    cmd = self._build_interpolate_command(state)
                elif mode == ControlMode.HOLD:
                    cmd = self._build_seq_hold_command(state)
                elif mode == ControlMode.DEFAULT and self._default_policy:
                    cmd = self._default_policy.step(state, np.zeros(3))
                else:
                    cmd = self._build_damping_command(state)

                cmd = self._smooth_command(cmd)
                cmd = self.safety.clamp_command(cmd, state)
                self.robot.send_command(cmd)
                self.robot.step()
                self.sim_step_count += 1

                self._seq_step += 1
                if self._seq_step % 100 == 0:
                    pct = int(100 * self._seq_step / total_steps) if total_steps > 0 else 100
                    print(
                        f"[runtime] mode={mode.value} "
                        f"step={self._seq_step}/{total_steps} ({pct}%)"
                    )

                if self._seq_step >= total_steps:
                    self._seq_index += 1
                    self._seq_step = 0
                    if self._seq_index >= len(self._mode_sequence):
                        print("[runtime] Mode sequence completed")
                        # Clear mode sequence so step() falls through
                        # to normal operation. Mark prepare done so
                        # branch 4 activates the policy instead of
                        # replaying PREPARE.
                        self._mode_sequence = None
                        self._prepare_done = True
                        return True
                    next_mode, next_dur = self._mode_sequence[self._seq_index]
                    self._control_mode = next_mode
                    print(f"[runtime] -> {next_mode.value} ({next_dur:.1f}s)")
                    if next_mode == ControlMode.INTERPOLATE:
                        self._interp_start_q = state.joint_positions.copy()
                    if next_mode == ControlMode.PREPARE:
                        self._prepare_reset_done = False

                return True

            # 3. IDLE/STOPPED: run default policy, hold, or damping
            if self.safety.state != SystemState.RUNNING:
                self._policy_active = False
                self._prepare_done = False
                self._clear_transition()
                state = self.robot.get_state()
                if self._estimator is not None:
                    self._estimator.update(state)
                    state = self._estimator.populate_robot_state(state)
                if self.safety.check_state_limits(state):
                    return True
                if not self.safety.check_tilt(state.imu_quaternion):
                    return True

                if self._idle_damping:
                    self._control_mode = ControlMode.DAMPING
                    cmd = self._build_damping_command(state)
                elif self._default_policy:
                    self._control_mode = ControlMode.DEFAULT
                    cmd = self._default_policy.step(state, np.zeros(3))
                else:
                    self._control_mode = ControlMode.HOLD
                    cmd = self._build_damping_command(state)
                cmd = self.safety.clamp_command(cmd, state)

                self.robot.send_command(cmd)
                self.robot.step()
                self.sim_step_count += 1
                return True

            # 4. Detect IDLE/STOPPED -> RUNNING transition
            if not self._policy_active:
                # If a mode sequence (PREPARE) is configured, replay it
                # before activating the policy. This ensures smooth
                # transitions on every start (not just the first).
                if self._initial_mode_sequence and self._mode_sequence is None and not self._prepare_done:
                    self._restart_with_prepare()
                    return True
                # Capture current state as interpolation start
                state_now = self.robot.get_state()
                start_pos = state_now.joint_positions.copy()

                # Determine starting gains (what was commanding the robot)
                if self._pending_start_kp is not None:
                    start_kp = self._pending_start_kp
                    start_kd = self._pending_start_kd
                    self._pending_start_kp = None
                    self._pending_start_kd = None
                elif self._default_policy is not None:
                    start_kp = self._default_policy.stiffness.copy()
                    start_kd = self._default_policy.damping.copy()
                else:
                    n = self.robot.n_dof
                    start_kp = np.zeros(n)
                    start_kd = np.full(n, self._kd_damp)

                self._policy_active = True
                self.policy.reset()
                self._step_count = 0

                if isinstance(self.policy, BeyondMimicPolicy):
                    self.policy.prefetch_reference(self.policy._start_timestep)
                    print("[runtime] Active policy started (BeyondMimic)")
                else:
                    print("[runtime] Active policy started")

                # Interpolation target: the policy's starting pose and gains.
                # starting_pos uses the first reference frame for BM,
                # default_pos for other policies.
                target_pos = self.policy.starting_pos.copy()
                target_kp = self.policy.stiffness.copy()
                target_kd = self.policy.damping.copy()
                self._start_transition(
                    start_pos, start_kp, start_kd,
                    target_pos, target_kp, target_kd,
                )

            # 5. TRANSITION: interpolate to policy starting position.
            #    Warm up the policy (ONNX session + obs history) each tick.
            if self._transition_active:
                state = self.robot.get_state()
                if self._estimator is not None:
                    self._estimator.update(state)
                    state = self._estimator.populate_robot_state(state)
                if self.safety.check_state_limits(state):
                    return True
                if not self.safety.check_tilt(state.imu_quaternion):
                    return True

                self.policy.warmup(state, np.zeros(3))
                cmd = self._transition_step_command()
                cmd = self.safety.clamp_command(cmd, state)
                self.robot.send_command(cmd)
                self.robot.step()
                self.sim_step_count += 1
                self._step_count += 1
                return True

            # 6. Get robot state
            state = self.robot.get_state()
            if self._estimator is not None:
                self._estimator.update(state)
                state = self._estimator.populate_robot_state(state)
            if self.safety.check_state_limits(state):
                return True
            if not self.safety.check_tilt(state.imu_quaternion):
                return True

            # 7. Policy step: obs -> inference -> control law -> command
            inference_start = time.perf_counter()
            vel_cmd = self._input_manager.get_velocity()
            cmd = self.policy.step(state, vel_cmd)
            inference_time = time.perf_counter() - inference_start

            # Check for BeyondMimic trajectory end — instant return to default
            if isinstance(self.policy, BeyondMimicPolicy):
                bm = self.policy
                if bm.time_step >= bm.trajectory_length:
                    print(
                        f"[runtime] BeyondMimic trajectory ended at "
                        f"step {bm.time_step}, returning to "
                        f"{'default policy' if self._default_policy else 'hold'}"
                    )
                    self._control_mode = (
                        ControlMode.DEFAULT
                        if self._default_policy
                        else ControlMode.HOLD
                    )
                    self.safety.stop()

            # 8. Clamp and send
            cmd = self.safety.clamp_command(cmd, state)
            self.robot.send_command(cmd)

            # 9. Step simulation
            self.robot.step()
            self.sim_step_count += 1

            # 10. Step count
            self._step_count += 1

            # 11. Log
            if self._logger is not None:
                loop_time = time.perf_counter() - loop_start
                # Get last action from policy (in robot order for logging)
                policy_action = self.policy.last_action
                log_action = self.joint_mapper.policy_to_robot(policy_action)
                # Observation not available post-step(); log zeros
                log_obs = np.zeros(160)
                self._logger.log_step(
                    timestamp=time.time(),
                    robot_state=state,
                    observation=log_obs,
                    action=log_action,
                    command=cmd,
                    system_state=self.safety.state,
                    velocity_command=vel_cmd,
                    timing={
                        "inference_ms": inference_time * 1000.0,
                        "loop_ms": loop_time * 1000.0,
                    },
                )

            # 12. Update telemetry
            self._update_telemetry(
                state, inference_time, loop_start, self._step_count
            )

        except Exception as exc:
            print(f"[runtime] EXCEPTION in control loop: {exc}")
            self.safety.estop()

        return True

    def _control_loop(self) -> None:
        """Threaded control loop: calls step() + sleep to maintain frequency.

        Used by ``start_threaded()`` for gui/viser/real modes.
        Includes frame-drop detection for timed execution.
        """
        while self._running:
            loop_start = time.perf_counter()
            self.step()
            # Frame-drop check (only meaningful for timed execution)
            elapsed = time.perf_counter() - loop_start
            self.safety.check_frame_drop(elapsed)
            # Sleep to maintain policy frequency
            remaining = self._dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            "loop_hz": 1.0 / max(loop_time, 1e-6),
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
            print(f"[runtime] Loaded policy: {os.path.basename(path)}")
        except Exception as exc:
            print(f"[runtime] Failed to load policy: {exc}")

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
