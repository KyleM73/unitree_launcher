"""Unitree G1 Deployment Stack -- Main Entry Point.

Provides:
    run_with_viewer() -- interactive MuJoCo passive viewer with GLFW key callbacks
    run_headless()    -- headless simulation for server/batch evaluation
    main()            -- CLI entry point with argparse
"""
from __future__ import annotations

# Patch SDK FIRST (before any unitree_sdk2py imports).
from unitree_launcher.compat import patch_unitree_b2_import, patch_unitree_threading
patch_unitree_b2_import()
patch_unitree_threading()

import argparse
import queue
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    G1_23DOF_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    apply_cli_overrides,
    load_config,
)
from unitree_launcher.control.controller import Controller
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.datalog.logger import DataLogger
from unitree_launcher.policy.base import PolicyInterface, detect_policy_format
from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy
from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.policy.observations import ObservationBuilder
from unitree_launcher.robot.real_robot import RealRobot
from unitree_launcher.robot.sim_robot import SimRobot

# GLFW key constants.
# Avoid letters bound by MuJoCo viewer (w=wireframe, s=shadow, a=analytic,
# d=geom, e=frame, r=reflect, c=contact, n=overlay, x=connect, z=perturb).
# See: https://www.glfw.org/docs/latest/group__keys.html
GLFW_KEY_MAP = {
    32: "space",        # GLFW_KEY_SPACE  -- start / stop
    265: "up",          # GLFW_KEY_UP     -- vx +
    264: "down",        # GLFW_KEY_DOWN   -- vx -
    263: "left",        # GLFW_KEY_LEFT   -- vy +
    262: "right",       # GLFW_KEY_RIGHT  -- vy -
    44: "comma",        # GLFW_KEY_COMMA  -- yaw +
    46: "period",       # GLFW_KEY_PERIOD -- yaw -
    47: "slash",        # GLFW_KEY_SLASH  -- zero velocity
    259: "backspace",   # GLFW_KEY_BACKSPACE -- e-stop
    257: "enter",       # GLFW_KEY_ENTER  -- clear e-stop
    45: "minus",        # GLFW_KEY_MINUS  -- prev policy
    61: "equal",        # GLFW_KEY_EQUAL  -- next policy
    261: "delete",       # GLFW_KEY_DELETE (Fn+Backspace on Mac) -- reset
}


# ============================================================================
# Viewer / Headless Runners
# ============================================================================

def run_with_viewer(
    sim_robot: SimRobot,
    controller: Controller,
    recorder=None,
    viser_port: Optional[int] = None,
) -> None:
    """Run simulation with interactive MuJoCo viewer.

    Args:
        viser_port: If set, also launch a viser web viewer on this port
            alongside the GLFW viewer (both stay in sync).

    Threading model:
        Main thread   -- viewer.sync() + drain key queue (this function)
        Control thread -- get_state / send_command / mj_step (controller)
        Viewer thread  -- GLFW rendering + event loop (MuJoCo internal)

    The key callback fires on MuJoCo's viewer thread.  To avoid
    cross-thread deadlocks with sim_robot.lock, the callback only
    enqueues key names; the main loop drains the queue *outside*
    the lock before calling viewer.sync().
    """
    import mujoco.viewer  # Local import — requires display (GLFW)

    # Optional viser sidecar.
    viser_viewer = None
    if viser_port is not None:
        from unitree_launcher.viz.viser_viewer import ViserViewer
        viser_viewer = ViserViewer(sim_robot.mj_model, port=viser_port)
        viser_viewer.setup()

    key_queue: queue.SimpleQueue = queue.SimpleQueue()

    def key_callback(keycode: int) -> None:
        key = GLFW_KEY_MAP.get(keycode)
        if key:
            key_queue.put(key)

    with mujoco.viewer.launch_passive(
        sim_robot.mj_model,
        sim_robot.mj_data,
        key_callback=key_callback,
    ) as viewer:
        controller.start()
        try:
            while viewer.is_running():
                # Drain key events (no lock held -- safe to call handle_key).
                while not key_queue.empty():
                    controller.handle_key(key_queue.get_nowait())

                with sim_robot.lock:
                    viewer.sync()

                # Update viser sidecar (outside lock).
                if viser_viewer is not None:
                    viser_viewer.update(sim_robot.mj_data, sim_robot.lock)

                # Capture OUTSIDE the lock — offscreen render can take
                # several ms and must never block the control thread.
                if recorder:
                    recorder.capture(sim_robot.mj_data)
                time.sleep(1.0 / 100.0)  # 100 Hz viewer refresh
        except KeyboardInterrupt:
            pass
        finally:
            controller.stop()
            if viser_viewer is not None:
                viser_viewer.close()


def run_headless(
    sim_robot: SimRobot,
    controller: Controller,
    duration: Optional[float] = None,
    max_steps: Optional[int] = None,
    auto_start_policy: bool = True,
    recorder=None,
) -> None:
    """Run simulation without viewer (for server evals).

    Termination conditions (first one wins):
        1. Ctrl+C (KeyboardInterrupt)
        2. *duration* seconds elapsed
        3. *max_steps* policy steps completed
        4. BeyondMimic trajectory ends (controller auto-stops -> STOPPED)
        5. Mode sequence completes (controller auto-stops)

    Args:
        duration: Auto-terminate after this many seconds (None = no limit).
        max_steps: Auto-terminate after this many policy steps (None = no limit).
        auto_start_policy: If True (default), auto-start the active policy
            via ``safety.start()``.  Set to False for mode-sequence-only
            runs (e.g. gantry) that don't need RUNNING state.
    """
    if max_steps is not None:
        controller._max_steps = max_steps

    controller.start()
    if auto_start_policy:
        # Auto-start the policy (no viewer to press Space).
        controller.safety.start()

    start_time = time.time()
    try:
        while True:
            # Capture OUTSIDE any lock — never block the control thread.
            if recorder:
                recorder.capture(sim_robot.mj_data)

            # Yield CPU but don't throttle — termination checks + capture
            # run as fast as the OS scheduler allows.
            time.sleep(0)

            if duration is not None and (time.time() - start_time) >= duration:
                print(f"[headless] Duration limit reached ({duration}s). Stopping.")
                break

            if not controller.is_running:
                print("[headless] Controller stopped (max_steps/duration or error).")
                break

            # BM trajectory auto-completes: safety transitions to STOPPED
            if controller.safety.state == SystemState.STOPPED:
                print("[headless] Active policy completed. Returning to stance.")
                break

    except KeyboardInterrupt:
        print("\n[headless] Ctrl+C received. Stopping.")
    finally:
        controller.stop()


def run_with_viser(
    sim_robot: SimRobot,
    controller: Controller,
    port: int = 8080,
    duration: Optional[float] = None,
    max_steps: Optional[int] = None,
    play: bool = False,
    policy_paths: Optional[list[str]] = None,
    active_policy: Optional[str] = None,
    recorder=None,
) -> None:
    """Run simulation with the viser web viewer.

    Combines viser visualization with headless termination logic so
    ``--viser`` works with ``--duration``/``--steps`` flags.

    Args:
        play: If True, don't exit when trajectory ends or controller stops.
            The simulation loops until Ctrl+C or duration limit.
        policy_paths: List of ONNX policy files for the dropdown selector.
        active_policy: Display name (stem) of the initially active policy.

    Threading model:
        Main thread    -- drain key queue, velocity sliders, viewer.update()
        Control thread -- get_state / send_command / mj_step (controller)
        Viser thread   -- asyncio WebSocket server (never touches mj_data)
    """
    from unitree_launcher.viz.viser_viewer import ViserViewer

    viewer = ViserViewer(
        sim_robot.mj_model,
        port=port,
        policy_paths=policy_paths,
        active_policy=active_policy,
    )
    viewer.setup()

    if max_steps is not None:
        controller._max_steps = max_steps

    controller.start()
    current_policy_name = active_policy or "unknown"

    start_time = time.time()
    try:
        while viewer.is_running():
            # Drain key events from GUI buttons.
            for key in viewer.drain_key_queue():
                controller.handle_key(key)
                # Re-capture follow origin on start or reset.
                if key in ("space", "delete"):
                    viewer._follow_origin_xy = None

            # Handle policy button selection.
            selected_policy = viewer.drain_policy_selection()
            if selected_policy is not None:
                try:
                    controller.reload_policy(selected_policy)
                    current_policy_name = Path(selected_policy).stem
                    print(f"[viser] Loaded policy: {current_policy_name}")
                except Exception as exc:
                    print(f"[viser] Failed to load policy: {exc}")

            # Read velocity sliders and apply to controller.
            vel = viewer.get_velocity_commands()
            if vel is not None:
                vx, vy, yaw = vel
                controller.set_velocity_command(vx, vy, yaw)

            # Update scene transforms.
            viewer.update(sim_robot.mj_data, sim_robot.lock)

            # Update status panel.
            state_name = controller.safety.state.name
            viewer.set_status(state_name, current_policy_name)

            # Capture recording outside lock.
            if recorder:
                recorder.capture(sim_robot.mj_data)

            # Termination checks (play mode skips trajectory-end exits).
            if duration is not None and (time.time() - start_time) >= duration:
                print(f"[viser] Duration limit reached ({duration}s). Stopping.")
                break

            if not play:
                if not controller.is_running:
                    print("[viser] Controller stopped.")
                    break

                if controller.safety.state == SystemState.STOPPED:
                    print("[viser] Active policy completed. Returning to stance.")
                    break

            time.sleep(1.0 / 50.0)  # 50 Hz update

    except KeyboardInterrupt:
        print("\n[viser] Ctrl+C received. Stopping.")
    finally:
        controller.stop()
        viewer.close()


# ============================================================================
# CLI
# ============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between sim and real sub-commands."""
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--policy", default=None,
                        help="Path to ONNX policy file")
    parser.add_argument("--default-policy", default=None,
                        help="Override default stance/velocity-tracking policy")
    _default_policy_dir = str(
        Path(__file__).resolve().parent.parent.parent / "assets" / "policies"
    )
    parser.add_argument("--policy-dir", default=_default_policy_dir,
                        help="Directory of ONNX files for policy switching")
    parser.add_argument("--robot", default=None,
                        help="Robot variant override (g1_29dof or g1_23dof)")
    parser.add_argument("--domain-id", type=int, default=None,
                        help="DDS domain ID (default: 1 for sim, 0 for real)")
    parser.add_argument("--log-dir", default="logs/",
                        help="Log output directory")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable logging")
    parser.add_argument("--no-est", action="store_true",
                        help="Override policy.use_estimator to false")
    parser.add_argument("--estimator", action="store_true",
                        help="Enable InEKF state estimator (auto-enabled for real)")
    parser.add_argument("--record", nargs="?", const="sim.mp4", default=None,
                        metavar="PATH",
                        help="Record video to MP4 (default: recording.mp4)")
    parser.add_argument("--gamepad", action="store_true",
                        help="Enable gamepad e-stop (Logitech F310)")
    parser.add_argument("--gamepad-debug", action="store_true",
                        help="Log raw HID reports on change (for button mapping)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch MuJoCo GLFW viewer (use mjpython on macOS)")
    parser.add_argument("--viser", action="store_true",
                        help="Launch viser web viewer")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser web viewer port (default: 8080)")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser. Exposed for testing."""
    parser = argparse.ArgumentParser(
        description="Unitree G1 Deployment Stack"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- sim sub-command --
    sim_parser = subparsers.add_parser("sim", help="Simulation mode")
    _add_common_args(sim_parser)
    sim_parser.add_argument("--gantry", action="store_true",
                            help="Gantry mode: damping -> interpolate -> hold (no policy needed)")
    sim_parser.add_argument("--duration", type=float, default=None,
                            help="Auto-stop after N seconds")
    sim_parser.add_argument("--steps", type=int, default=None,
                            help="Auto-stop after N policy steps")
    sim_parser.add_argument("--play", action="store_true",
                            help="Loop forever: don't exit when trajectory ends (viser only)")
    # Real-only args with defaults (so args.interface etc. always exist).
    sim_parser.set_defaults(interface=None, backend="python")

    # -- real sub-command --
    real_parser = subparsers.add_parser("real", help="Real robot mode")
    _add_common_args(real_parser)
    real_parser.add_argument("--interface", required=True,
                             help="Network interface (e.g. eth0)")
    real_parser.add_argument("--backend", choices=["python", "cpp"], default="python",
                             help="Real robot SDK: python (unitree_sdk2py) or cpp (pybind11)")
    # Sim-only args with defaults (so args.gantry etc. always exist).
    real_parser.set_defaults(gantry=False, duration=None, steps=None, play=False)

    return parser


def _start_gamepad(args, safety):
    """Create and start a GamepadMonitor, or return None on failure."""
    from unitree_launcher.control.gamepad import start_gamepad
    return start_gamepad(safety, debug=args.gamepad_debug)


def _is_docker() -> bool:
    """Return True if running inside a Docker container."""
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def main(argv: Optional[list] = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list for testing (default: sys.argv[1:]).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- Validate display flags ----
    if args.gui and _is_docker():
        parser.error("--gui is not supported in Docker (no GLFW display). Use --viser instead.")

    # ---- Validate ----
    is_gantry = args.mode == "sim" and args.gantry
    # When --policy-dir is given without --policy, pick the first ONNX file.
    if not args.policy and args.policy_dir:
        import glob as _glob
        _candidates = sorted(_glob.glob(str(Path(args.policy_dir) / "*.onnx")))
        if _candidates:
            args.policy = _candidates[0]
    if not is_gantry and not args.policy:
        parser.error("--policy is required (or use --gantry for gantry simulation)")

    # ---- Config ----
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    # Domain ID defaults: sim=1, real=0
    if args.domain_id is not None:
        config.network.domain_id = args.domain_id
    elif args.mode == "real":
        config.network.domain_id = 0

    # Interface for real mode
    if args.mode == "real":
        config.network.interface = args.interface

    # ---- Robot ----
    variant = config.robot.variant
    if args.mode == "sim":
        robot = SimRobot(config)
    elif args.backend == "cpp":
        from unitree_launcher.robot.cpp_real_robot import CppRealRobot
        robot = CppRealRobot(config)
    else:
        robot = RealRobot(config)

    robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS

    # ---- Gantry mode (sim only, no policy needed) ----
    if is_gantry:
        from unitree_launcher.control.safety import ControlMode
        from unitree_launcher.gantry import (
            ElasticBand,
            build_gain_arrays,
            build_home_positions,
            enable_gantry,
            get_torso_body_id,
            setup_gantry_band,
        )

        safety = SafetyController(config, n_dof=robot.n_dof)

        enable_gantry(robot)
        band = ElasticBand()
        torso_id = get_torso_body_id(robot.mj_model)
        setup_gantry_band(robot, band, torso_id)

        kp_end, kd_end = build_gain_arrays("isaaclab")
        target_q = build_home_positions()

        mode_sequence = [
            (ControlMode.DAMPING, 5.0),
            (ControlMode.INTERPOLATE, 5.0),
            (ControlMode.HOLD, 5.0),
            (ControlMode.DAMPING, 5.0),
        ]

        controller = Controller(
            robot=robot,
            safety=safety,
            config=config,
            mode_sequence=mode_sequence,
            interp_target_q=target_q,
            interp_kp_end=kp_end,
            interp_kd_end=kd_end,
        )

        robot.connect()

        # ---- Video recorder (gantry) ----
        recorder = None
        if args.record:
            from unitree_launcher.recording import VideoRecorder, normalize_record_path
            record_path = normalize_record_path(args.record)
            recorder = VideoRecorder(
                record_path, robot.mj_model, robot.mj_data,
                step_fn=lambda: controller.sim_step_count,
            )

        print(f"[main] Gantry mode: DAMPING(5s) -> INTERPOLATE(5s) -> HOLD(5s) -> DAMPING(5s)")

        # ---- Gamepad E-Stop (gantry) ----
        gamepad_monitor = None
        if args.gamepad:
            gamepad_monitor = _start_gamepad(args, safety)

        try:
            _gui = args.gui
            _vis = args.viser
            if _gui:
                run_with_viewer(
                    robot, controller, recorder=recorder,
                    viser_port=args.port if _vis else None,
                )
            elif _vis:
                run_with_viser(
                    robot, controller, port=args.port,
                    duration=args.duration,
                    play=args.play,
                    recorder=recorder,
                )
            else:
                run_headless(
                    robot, controller,
                    duration=args.duration,
                    auto_start_policy=False,
                    recorder=recorder,
                )
        finally:
            if gamepad_monitor is not None:
                gamepad_monitor.stop()
            if recorder:
                recorder.close()
            controller.stop()
            robot.disconnect()

        return

    # ---- Default policy (IL velocity tracking for stance) ----
    default_policy_path = args.default_policy or config.policy.default_policy
    default_policy = None
    default_obs_builder = None
    default_joint_mapper = None

    if default_policy_path and Path(default_policy_path).exists():
        try:
            import onnxruntime as _ort
            _sess = _ort.InferenceSession(
                default_policy_path, providers=["CPUExecutionProvider"]
            )
            model_obs_dim = _sess.get_inputs()[0].shape[1]
            n_actions = _sess.get_outputs()[0].shape[1]
            del _sess

            use_estimator = config.policy.use_estimator
            if args.no_est:
                use_estimator = False

            # Derive the joint subset from ONNX output size.  IsaacLab
            # locomotion policies use the first N joints of the canonical
            # IsaacLab ordering (typically 23 = 29 minus 6 wrist joints).
            il_joints = [
                j.replace("_joint", "")
                for j in ISAACLAB_G1_29DOF_JOINTS[:n_actions]
            ]
            default_joint_mapper = JointMapper(
                robot_joints=robot_joints,
                observed_joints=il_joints,
                controlled_joints=il_joints,
            )

            # Pick the estimator setting that matches the model's obs_dim.
            matched = False
            for use_est in [use_estimator, not use_estimator]:
                default_obs_builder = ObservationBuilder(
                    default_joint_mapper, config, use_estimator=use_est
                )
                if default_obs_builder.observation_dim == model_obs_dim:
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Cannot match default policy obs_dim={model_obs_dim} "
                    f"(n_actions={n_actions}) with ObservationBuilder"
                )

            default_policy = IsaacLabPolicy(
                default_joint_mapper, model_obs_dim
            )
            default_policy.load(default_policy_path)
            print(
                f"[main] Default policy loaded: {default_policy_path} "
                f"(obs_dim={model_obs_dim})"
            )
        except Exception as exc:
            print(
                f"[main] WARNING: Could not load default policy: {exc}. "
                f"Using static hold mode."
            )
            default_policy = None
            default_obs_builder = None
            default_joint_mapper = None
    elif default_policy_path:
        print(
            f"[main] WARNING: Default policy not found: {default_policy_path}. "
            f"Using static hold mode."
        )

    # ---- Active policy format & joint ordering ----
    policy_format = config.policy.format or detect_policy_format(args.policy)

    observed_joints = config.policy.observed_joints
    controlled_joints = config.policy.controlled_joints

    # BeyondMimic: extract joint ordering from ONNX metadata so that the
    # JointMapper correctly reorders between robot-native and policy order.
    if policy_format == "beyondmimic" and controlled_joints is None:
        metadata = BeyondMimicPolicy.load_metadata(args.policy)
        if "joint_names" in metadata:
            policy_joints = [
                j.strip().replace("_joint", "")
                for j in metadata["joint_names"].split(",")
            ]
            controlled_joints = policy_joints
            observed_joints = policy_joints
    elif policy_format == "isaaclab" and controlled_joints is None:
        # IsaacLab policies use canonical joint ordering.  Derive subset
        # from ONNX output size (same approach as default policy loading).
        import onnxruntime as _ort
        _sess = _ort.InferenceSession(
            args.policy, providers=["CPUExecutionProvider"]
        )
        n_actions = _sess.get_outputs()[0].shape[1]
        del _sess
        il_joints = [
            j.replace("_joint", "")
            for j in ISAACLAB_G1_29DOF_JOINTS[:n_actions]
        ]
        controlled_joints = il_joints
        observed_joints = il_joints

    active_joint_mapper = JointMapper(
        robot_joints=robot_joints,
        observed_joints=observed_joints,
        controlled_joints=controlled_joints,
    )

    # ---- Active policy ----
    if policy_format == "isaaclab":
        use_estimator = config.policy.use_estimator
        if args.no_est:
            use_estimator = False
        active_obs_builder = ObservationBuilder(
            active_joint_mapper, config, use_estimator=use_estimator
        )
        active_policy: PolicyInterface = IsaacLabPolicy(
            active_joint_mapper, active_obs_builder.observation_dim
        )
    else:
        # BeyondMimic builds its own observations internally.
        active_obs_builder = None
        active_policy = BeyondMimicPolicy(
            active_joint_mapper,
            obs_dim=160,  # standard BeyondMimic observation size
            use_onnx_metadata=config.policy.use_onnx_metadata,
        )

    active_policy.load(args.policy)

    # Match simulation parameters to the training environment.
    if isinstance(active_policy, BeyondMimicPolicy) and hasattr(robot, 'set_home_positions'):
        # 1. Set initial robot pose to policy's default joint positions.
        default_native = active_joint_mapper.action_to_robot(
            active_policy.default_joint_pos
        )
        robot.set_home_positions(default_native)

        # 2. Set per-joint armature to match training sim.
        if active_policy.stiffness is not None:
            import numpy as _np
            _natural_freq = 10.0 * 2.0 * _np.pi
            armature_policy = active_policy.stiffness / (_natural_freq ** 2)
            armature_native = active_joint_mapper.action_to_robot(armature_policy)
            robot.set_armature(armature_native)

        # 3. Override actuator kp/kv to match BM training gains.
        #    The XML defaults are IsaacLab gains (from mjlab armature formula).
        #    BM uses its own stiffness/damping from ONNX metadata.
        if hasattr(robot, 'set_actuator_gains') and active_policy.stiffness is not None:
            import numpy as _np
            bm_kp_policy = active_policy.stiffness
            bm_kv_policy = active_policy.damping if active_policy.damping is not None else _np.zeros_like(bm_kp_policy)
            bm_kp_native = active_joint_mapper.action_to_robot(bm_kp_policy)
            bm_kv_native = active_joint_mapper.action_to_robot(bm_kv_policy)
            robot.set_actuator_gains(bm_kp_native, bm_kv_native)

    # Validate obs_dim match (IsaacLab only).
    if active_obs_builder is not None:
        assert active_policy.observation_dim == active_obs_builder.observation_dim, (
            f"Policy obs_dim={active_policy.observation_dim} != "
            f"builder obs_dim={active_obs_builder.observation_dim}"
        )

    # ---- Safety ----
    safety = SafetyController(config, n_dof=robot.n_dof)

    # Wire safety reference for real robot watchdog
    if hasattr(robot, 'set_safety'):
        robot.set_safety(safety)

    # ---- Logger ----
    logger = None
    if not args.no_log:
        policy_name = Path(args.policy).stem
        run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{args.mode}_{policy_name}"
        logger = DataLogger(config.logging, run_name, args.log_dir)

    # ---- State Estimator (for real robot or estimator-in-the-loop) ----
    estimator = None
    if args.mode == "real" or args.estimator:
        from unitree_launcher.estimation import StateEstimator
        estimator = StateEstimator(config)
        print("[main] State estimator enabled (InEKF).")

    # ---- Controller ----
    controller = Controller(
        robot=robot,
        policy=active_policy,
        safety=safety,
        joint_mapper=active_joint_mapper,
        obs_builder=active_obs_builder,
        config=config,
        logger=logger,
        policy_dir=args.policy_dir,
        default_policy=default_policy,
        default_obs_builder=default_obs_builder,
        default_joint_mapper=default_joint_mapper,
    )
    if estimator is not None:
        controller.set_estimator(estimator)

    # ---- Pre-load all policies in policy_dir for instant switching ----
    for policy_file in controller._policy_files:
        if policy_file == args.policy:
            # Already loaded as the active policy — register it directly.
            controller._preloaded_policies[policy_file] = (
                active_policy, active_joint_mapper, active_obs_builder,
            )
            continue
        try:
            fmt = detect_policy_format(policy_file)
            if fmt == "isaaclab":
                import onnxruntime as _ort
                _sess = _ort.InferenceSession(
                    policy_file, providers=["CPUExecutionProvider"]
                )
                _n_act = _sess.get_outputs()[0].shape[1]
                del _sess
                _il_joints = [
                    j.replace("_joint", "")
                    for j in ISAACLAB_G1_29DOF_JOINTS[:_n_act]
                ]
                _mapper = JointMapper(
                    robot_joints=robot_joints,
                    observed_joints=_il_joints,
                    controlled_joints=_il_joints,
                )
                _use_est = config.policy.use_estimator
                if args.no_est:
                    _use_est = False
                _obs = ObservationBuilder(_mapper, config, use_estimator=_use_est)
                _pol: PolicyInterface = IsaacLabPolicy(_mapper, _obs.observation_dim)
                _pol.load(policy_file)
                controller._preloaded_policies[policy_file] = (_pol, _mapper, _obs)
            else:
                _meta = BeyondMimicPolicy.load_metadata(policy_file)
                _pj = None
                if "joint_names" in _meta:
                    _pj = [
                        j.strip().replace("_joint", "")
                        for j in _meta["joint_names"].split(",")
                    ]
                _mapper = JointMapper(
                    robot_joints=robot_joints,
                    observed_joints=_pj,
                    controlled_joints=_pj,
                )
                _pol = BeyondMimicPolicy(
                    _mapper, obs_dim=160,
                    use_onnx_metadata=config.policy.use_onnx_metadata,
                )
                _pol.load(policy_file)
                controller._preloaded_policies[policy_file] = (_pol, _mapper, None)
            print(f"[main] Pre-loaded policy: {Path(policy_file).stem}")
        except Exception as exc:
            print(f"[main] WARNING: Could not pre-load {Path(policy_file).name}: {exc}")

    # Also register the active policy if not already in the dir listing.
    if args.policy and args.policy not in controller._preloaded_policies:
        controller._preloaded_policies[args.policy] = (
            active_policy, active_joint_mapper, active_obs_builder,
        )

    # ---- Connect & run ----
    robot.connect()
    if logger is not None:
        logger.start()

    # ---- Gamepad E-Stop ----
    gamepad_monitor = None
    if args.gamepad:
        gamepad_monitor = _start_gamepad(args, safety)

    # ---- Video recorder (created after logger.start so log dir exists) ----
    recorder = None
    if args.record and args.mode == "sim":
        from unitree_launcher.recording import VideoRecorder, normalize_record_path
        log_dir = logger.log_path if logger is not None else None
        record_path = normalize_record_path(args.record, directory=log_dir)
        recorder = VideoRecorder(
            record_path, robot.mj_model, robot.mj_data,
            step_fn=lambda: controller.sim_step_count,
        )

    # Install signal handlers for SIGTERM/SIGHUP so the robot gets a clean
    # shutdown even if the process is killed or the SSH session drops.
    shutdown_requested = False

    def _signal_shutdown(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n[main] Received {sig_name}. Shutting down...")
        if shutdown_requested:
            # Second signal — force exit
            raise SystemExit(1)
        shutdown_requested = True
        controller.stop()
        if hasattr(robot, 'graceful_shutdown'):
            robot.graceful_shutdown()
        else:
            robot.disconnect()
        if logger is not None:
            logger.stop()
        raise SystemExit(0)

    prev_sigterm = signal.signal(signal.SIGTERM, _signal_shutdown)
    prev_sighup = signal.signal(signal.SIGHUP, _signal_shutdown)

    use_gui = args.gui
    use_viser = args.viser
    try:
        if use_gui:
            # --gui (optionally with --viser sidecar)
            run_with_viewer(
                robot, controller, recorder=recorder,
                viser_port=args.port if use_viser else None,
            )
        elif use_viser:
            # --viser only (headless physics + web viewer)
            _viser_paths = list(controller._preloaded_policies.keys())
            run_with_viser(
                robot, controller, port=args.port,
                duration=args.duration,
                max_steps=args.steps,
                play=args.play,
                policy_paths=_viser_paths or None,
                active_policy=Path(args.policy).stem if args.policy else None,
                recorder=recorder,
            )
        else:
            # headless (default)
            run_headless(
                robot, controller,
                duration=args.duration,
                max_steps=args.steps,
                recorder=recorder,
            )
    finally:
        # Restore original signal handlers to avoid double-shutdown
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGHUP, prev_sighup)

        if gamepad_monitor is not None:
            gamepad_monitor.stop()
        if recorder:
            recorder.close()
        controller.stop()
        if hasattr(robot, 'graceful_shutdown'):
            robot.graceful_shutdown()
        else:
            robot.disconnect()
        if logger is not None:
            logger.stop()


if __name__ == "__main__":
    main()
