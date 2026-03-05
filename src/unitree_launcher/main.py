"""Unitree G1 Deployment Stack -- Main Entry Point.

Modes:
    sim    -- development simulation (500Hz physics, gui/viser/headless)
    eval   -- accurate evaluation (1000Hz physics, headless)
    real   -- onboard robot deployment (C++ unitree_cpp)
    mirror -- read-only DDS visualization of real robot (gui/viser)

Entry points (via ``uv run``):
    uv run sim --gui --policy stance.onnx
    uv run eval --steps 500 --policy stance.onnx
    uv run real --policy assets/policies/beyondmimic_29dof.onnx
    uv run mirror --gui --interface en8
"""
from __future__ import annotations


import argparse
import platform
import queue
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    G1_23DOF_JOINTS,
    apply_cli_overrides,
    load_config,
    merge_configs,
)
from unitree_launcher.control.runtime import Runtime
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.datalog.logger import DataLogger
from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy
from unitree_launcher.policy.factory import load_policy, load_default_policy, preload_policy_dir

# Robot backends imported lazily in main():
#   SimRobot (mujoco) for sim/eval modes
#   RealRobot (C++ unitree_cpp) for real mode
#   MirrorRobot (Python DDS subscriber) for mirror mode — imported in mirror.py

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
    runtime: Runtime,
    recorder=None,
    viser_port: Optional[int] = None,
) -> None:
    """Run with interactive MuJoCo viewer. Control loop in daemon thread.

    Threading model:
        Main thread    -- viewer.sync() + feed keyboard input
        Control thread -- runtime.step() at policy_frequency (start_threaded)
        Viewer thread  -- GLFW rendering + event loop (MuJoCo internal)
    """
    import mujoco.viewer

    sim_robot = runtime.robot

    # Get the keyboard input controller from the input manager
    from unitree_launcher.controller.keyboard import KeyboardInput
    keyboard = None
    for ctrl in runtime.input_manager.controllers:
        if isinstance(ctrl, KeyboardInput):
            keyboard = ctrl
            break

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
        runtime.start_threaded()
        try:
            while viewer.is_running():
                # Feed GLFW keys to keyboard input controller
                while not key_queue.empty():
                    key = key_queue.get_nowait()
                    if keyboard:
                        keyboard.push_key(key)

                with sim_robot.lock:
                    viewer.sync()

                if viser_viewer is not None:
                    viser_viewer.update(sim_robot.mj_data, sim_robot.lock)

                if recorder:
                    recorder.capture(sim_robot.mj_data)
                time.sleep(1.0 / 100.0)
        except KeyboardInterrupt:
            pass
        finally:
            runtime.stop()
            if viser_viewer is not None:
                viser_viewer.close()


def run_headless(
    runtime: Runtime,
    duration: Optional[float] = None,
    max_steps: Optional[int] = None,
    auto_start_policy: bool = True,
    recorder=None,
    rate_limit: bool = False,
) -> None:
    """Run runtime in a headless loop (no viewer).

    In sim/eval mode (rate_limit=False), runs as fast as compute allows —
    physics stepping is the rate limiter. On real hardware (rate_limit=True),
    sleeps to maintain policy_frequency.

    Termination conditions (first one wins):
        1. Ctrl+C (KeyboardInterrupt)
        2. *duration* seconds elapsed
        3. *max_steps* policy steps completed
        4. BeyondMimic trajectory ends (safety -> STOPPED)
        5. Mode sequence completes (runtime stops itself)
    """
    runtime.start()
    if auto_start_policy:
        runtime.safety.start()

    start_time = time.time()
    step_count = 0
    policy_freq = runtime.config.control.policy_frequency
    dt = 1.0 / policy_freq
    telemetry_interval = policy_freq  # Print every ~1 second
    try:
        while True:
            step_start = time.time()

            if not runtime.step():
                break
            step_count += 1

            if recorder and hasattr(runtime.robot, 'mj_data'):
                recorder.capture(runtime.robot.mj_data)

            # Periodic telemetry for SSH monitoring
            if step_count % telemetry_interval == 0:
                t = runtime.get_telemetry()
                print(
                    f"[{t['loop_hz']:.0f} Hz | {t['inference_ms']:.1f}ms inf | "
                    f"{t['base_height']:.3f}m | {t['system_state']} | step {t['step_count']}]"
                )

            if max_steps is not None and step_count >= max_steps:
                print(f"[headless] Reached {max_steps} steps. Stopping")
                break

            if duration is not None and (time.time() - start_time) >= duration:
                print(f"[headless] Duration limit reached ({duration}s). Stopping")
                break

            if runtime.safety.state == SystemState.STOPPED:
                print("[headless] Active policy completed")
                break

            # Per-step rate limiting (real hardware only)
            if rate_limit:
                remaining = dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[headless] Ctrl+C, triggering E-stop")
        runtime.safety.estop()
    finally:
        runtime.stop()


def run_with_viser(
    runtime: Runtime,
    port: int = 8080,
    duration: Optional[float] = None,
    max_steps: Optional[int] = None,
    play: bool = False,
    policy_paths: Optional[list[str]] = None,
    active_policy: Optional[str] = None,
    recorder=None,
    gantry: bool = False,
) -> None:
    """Run with viser web viewer. Control loop in daemon thread.

    Threading model:
        Main thread    -- drain key queue, velocity sliders, viewer.update()
        Control thread -- runtime.step() at policy_frequency (start_threaded)
        Viser thread   -- asyncio WebSocket server (never touches mj_data)
    """
    from unitree_launcher.viz.viser_viewer import ViserViewer

    sim_robot = runtime.robot

    viewer = ViserViewer(
        sim_robot.mj_model,
        port=port,
        policy_paths=policy_paths,
        active_policy=active_policy,
        gantry=gantry,
    )
    viewer.setup()

    # Get input controllers from the input manager
    from unitree_launcher.controller.keyboard import KeyboardInput
    from unitree_launcher.controller.viser_input import ViserInput
    keyboard = None
    viser_input = None
    for ctrl in runtime.input_manager.controllers:
        if isinstance(ctrl, KeyboardInput):
            keyboard = ctrl
        elif isinstance(ctrl, ViserInput):
            viser_input = ctrl

    runtime.start_threaded()

    start_time = time.time()
    try:
        while viewer.is_running():
            # Feed viser keys to keyboard controller
            for key in viewer.drain_key_queue():
                if keyboard:
                    keyboard.push_key(key)
                if key in ("space", "delete"):
                    viewer._follow_origin_xy = None

            # Feed policy selection to viser input controller
            selected_policy = viewer.drain_policy_selection()
            if selected_policy is not None and viser_input:
                viser_input.push_policy_selection(selected_policy)

            # Feed velocity sliders to viser input controller
            vel = viewer.get_velocity_commands()
            if vel is not None and viser_input:
                vx, vy, yaw = vel
                viser_input.push_velocity(vx, vy, yaw)

            viewer.update(sim_robot.mj_data, sim_robot.lock)

            state_name = runtime.safety.state.name
            vel_cmd = runtime.input_manager.get_velocity()
            policy_name = Path(runtime._active_policy_path).stem if runtime._active_policy_path else (active_policy or "unknown")
            telem = runtime.get_telemetry()
            viewer.set_status(state_name, policy_name, velocity_command=vel_cmd, telemetry=telem)

            if recorder:
                recorder.capture(sim_robot.mj_data)

            if duration is not None and (time.time() - start_time) >= duration:
                print(f"[viser] Duration limit reached ({duration}s). Stopping")
                break

            if not play:
                if not runtime.is_running:
                    print("[viser] Pipeline stopped")
                    break
                if runtime.safety.state == SystemState.STOPPED:
                    print("[viser] Active policy completed")
                    break

            time.sleep(1.0 / 50.0)

    except KeyboardInterrupt:
        print("\n[viser] Ctrl+C received. Stopping")
    finally:
        runtime.stop()
        viewer.close()


# ============================================================================
# CLI
# ============================================================================

def _add_base_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    """Add arguments shared by all modes."""
    parser.add_argument("-c", "--config", default=default_config,
                        help=f"Path to YAML config (default: {default_config})")
    parser.add_argument("--preset", default=None,
                        help="Named config preset (e.g., unsafe)")
    parser.add_argument("--policy", default=None,
                        help="Path to ONNX policy file")
    parser.add_argument("--default-policy", default=None,
                        help="Override default stance/velocity-tracking policy")
    _default_policy_dir = str(
        Path(__file__).resolve().parent.parent.parent / "assets" / "policies"
    )
    parser.add_argument("-d", "--policy-dir", default=_default_policy_dir,
                        help="Directory of ONNX files for policy switching")
    parser.add_argument("--robot", default=None,
                        help="Robot variant override (g1_29dof or g1_23dof)")
    parser.add_argument("--domain-id", type=int, default=None,
                        help="DDS domain ID override")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable logging")
    parser.add_argument("--log-dir", default="logs/",
                        help="Log output directory")
    parser.add_argument("--no-est", action="store_true",
                        help="Override policy.use_estimator to false")
    parser.add_argument("--estimator", action="store_true",
                        help="Enable InEKF state estimator")
    parser.add_argument("--estimator-verbose", action="store_true",
                        help="Enable estimator diagnostic logging")
    parser.add_argument("--estimate-imu", action="store_true",
                        help="Also estimate IMU orientation and angular velocity")


def _add_viewer_args(parser: argparse.ArgumentParser) -> None:
    """Add viewer flags (sim and mirror only)."""
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

    # -- sim: development simulation (500Hz, any viewer) --
    sim_parser = subparsers.add_parser("sim", help="Simulation (dev, 500Hz physics, gui/viser/headless)")
    _add_base_args(sim_parser, "configs/sim.yaml")
    _add_viewer_args(sim_parser)
    sim_parser.add_argument("--gantry", action="store_true",
                            help="Gantry mode: elastic band + sinusoid (no policy needed)")
    sim_parser.add_argument("--duration", type=float, default=None,
                            help="Auto-stop after N seconds")
    sim_parser.add_argument("--steps", type=int, default=None,
                            help="Auto-stop after N policy steps")
    sim_parser.add_argument("--play", action="store_true",
                            help="Loop forever: don't exit when trajectory ends (viser only)")
    sim_parser.add_argument("--record", nargs="?", const="sim.mp4", default=None,
                            metavar="PATH", help="Record video to MP4")
    sim_parser.add_argument("--gamepad", action="store_true",
                            help="Enable gamepad e-stop (Logitech F310)")
    sim_parser.add_argument("--gamepad-debug", action="store_true",
                            help="Log raw HID reports on change (for button mapping)")
    sim_parser.set_defaults(interface=None)

    # -- eval: accurate headless evaluation (1000Hz) --
    eval_parser = subparsers.add_parser("eval", help="Evaluation (1000Hz physics, headless)")
    _add_base_args(eval_parser, "configs/sim.yaml")
    eval_parser.add_argument("--gantry", action="store_true",
                             help="Gantry mode: elastic band + sinusoid (no policy needed)")
    eval_parser.add_argument("--duration", type=float, default=None,
                             help="Auto-stop after N seconds")
    eval_parser.add_argument("--steps", type=int, default=None,
                             help="Auto-stop after N policy steps")
    eval_parser.add_argument("--record", nargs="?", const="eval.mp4", default=None,
                             metavar="PATH", help="Record video to MP4")
    eval_parser.set_defaults(gui=False, viser=False, play=False, interface=None,
                             gamepad=False, gamepad_debug=False)

    # -- real: onboard robot deployment --
    real_parser = subparsers.add_parser("real", help="Onboard robot deployment (run on G1)")
    _add_base_args(real_parser, "configs/real.yaml")
    real_parser.add_argument("--interface", default="eth0",
                             help="Network interface (default: eth0 on G1 Orin)")
    real_parser.add_argument("--gantry", action="store_true",
                             help="Gantry arm test: prepare -> sinusoid on right shoulder")
    real_parser.add_argument("--duration", type=float, default=None,
                             help="Auto-stop after N seconds")
    real_parser.set_defaults(steps=None, play=False, gui=False, viser=False,
                             record=None, gamepad=False, gamepad_debug=False)

    # -- mirror: read-only DDS subscriber to visualize real robot --
    mirror_parser = subparsers.add_parser("mirror", help="Mirror real robot state into MuJoCo viewer")
    mirror_parser.add_argument("-c", "--config", default="configs/sim.yaml",
                                help="Path to YAML config (default: configs/sim.yaml)")
    mirror_parser.add_argument("--robot", default=None,
                                help="Robot variant override (g1_29dof or g1_23dof)")
    mirror_parser.add_argument("--interface", default="en8",
                                help="Network interface to real robot (default: en8)")
    mirror_parser.add_argument("--domain-id", type=int, default=None,
                                help="DDS domain ID override")
    _add_viewer_args(mirror_parser)
    mirror_parser.set_defaults(gantry=False, duration=None, steps=None, play=False,
                               preset=None, policy=None, default_policy=None,
                               policy_dir=None, no_log=True, log_dir="logs/",
                               no_est=False, estimator=False, estimator_verbose=False,
                               estimate_imu=False, record=None, gamepad=False,
                               gamepad_debug=False)

    return parser


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

    # ---- Mirror mode: read-only DDS visualization ----
    if args.mode == "mirror":
        if not args.gui and not args.viser:
            parser.error("mirror mode requires --gui or --viser (no headless mirror)")
        from unitree_launcher.mirror import run_mirror
        run_mirror(args)
        return

    # ---- Validate display flags ----
    if getattr(args, 'gui', False) and _is_docker():
        parser.error("--gui is not supported in Docker (no GLFW display). Use --viser instead.")

    # ---- Validate ----
    is_gantry = getattr(args, 'gantry', False)
    # When --policy-dir is given without --policy, pick the first ONNX file.
    if not args.policy and args.policy_dir:
        import glob as _glob
        _candidates = sorted(_glob.glob(str(Path(args.policy_dir) / "*.onnx")))
        if _candidates:
            args.policy = _candidates[0]
    if not is_gantry and not args.policy:
        parser.error("--policy is required (or use --gantry for gantry mode)")

    # ---- Config ----
    if args.preset:
        preset_path = Path(__file__).resolve().parent.parent.parent / "configs" / f"{args.preset}.yaml"
        if not preset_path.exists():
            parser.error(f"Unknown preset: {args.preset}")
        config = load_config(str(preset_path))
        if args.config not in ("configs/sim.yaml", "configs/real.yaml"):
            config = merge_configs(config, load_config(args.config))
    else:
        config = load_config(args.config)
    apply_cli_overrides(config, args)

    # Eval mode: force 1000Hz for maximum accuracy
    if args.mode == "eval":
        config.control.sim_frequency = 1000

    # Domain ID defaults: sim/eval=1, real/mirror=0
    if args.domain_id is not None:
        config.network.domain_id = args.domain_id
    elif args.mode in ("real", "mirror"):
        config.network.domain_id = 0

    # ---- Robot ----
    variant = config.robot.variant
    if args.mode in ("sim", "eval"):
        from unitree_launcher.robot.sim_robot import SimRobot
        robot = SimRobot(config)
    else:
        # real mode: onboard via C++ unitree_cpp
        from unitree_launcher.robot.real_robot import RealRobot
        robot = RealRobot(config)

    robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS

    # ---- Gantry mode: arm sinusoid test (sim + real) ----
    if is_gantry:
        from unitree_launcher.policy.sinusoid_policy import SinusoidPolicy
        from unitree_launcher.policy.hold_policy import HoldPolicy
        from unitree_launcher.policy.joint_mapper import JointMapper

        # Gantry: disable state-limit E-stops (robot hangs on harness,
        # gravity pulls joints past limits and causes torque/velocity
        # readings that exceed normal thresholds).
        config.safety.joint_position_limits = False
        config.safety.joint_velocity_limits = False
        config.safety.torque_limits = False
        safety = SafetyController(config, n_dof=robot.n_dof)
        mapper = JointMapper(robot_joints)
        sinusoid = SinusoidPolicy(mapper, config)
        default_policy = HoldPolicy(mapper, config)

        # Sim: enable elastic band gantry
        if args.mode in ("sim", "eval"):
            from unitree_launcher.gantry import (
                ElasticBand,
                enable_gantry,
                get_torso_body_id,
                setup_gantry_band,
            )
            enable_gantry(robot)
            band = ElasticBand()
            torso_id = get_torso_body_id(robot.mj_model)
            setup_gantry_band(robot, band, torso_id)

        # Input controllers (wireless for real, gamepad + keyboard for sim)
        from unitree_launcher.controller.input import InputManager
        from unitree_launcher.controller.keyboard import KeyboardInput
        input_controllers = []
        wireless = None
        if args.mode == "real":
            from unitree_launcher.controller.wireless import WirelessInput
            wireless = WirelessInput()
            input_controllers.append(wireless)
        from unitree_launcher.controller.gamepad_input import start_gamepad_input
        gamepad = start_gamepad_input()
        if gamepad:
            input_controllers.append(gamepad)
        input_controllers.append(KeyboardInput())
        input_mgr = InputManager(input_controllers)

        runtime = Runtime(
            robot=robot,
            policy=sinusoid,
            safety=safety,
            joint_mapper=mapper,
            config=config,
            default_policy=default_policy,
            default_joint_mapper=mapper,
            input_manager=input_mgr,
            idle_damping=True,
        )

        robot.connect()
        if hasattr(robot, 'set_safety'):
            robot.set_safety(safety)
        if wireless is not None and hasattr(robot, 'set_wireless_handler'):
            robot.set_wireless_handler(wireless.parse)

        # Mode sequence: settle → prepare → normal operation.
        # Sim: DAMPING settle (5s) lets the robot hang under gravity + elastic band
        # before blending to home. Real: no settle needed (robot is on gantry).
        from unitree_launcher.control.safety import ControlMode
        seq = []
        if args.mode in ("sim", "eval"):
            seq.append((ControlMode.DAMPING, 10.0))
        seq.append((ControlMode.PREPARE, 20.0))
        runtime._mode_sequence = seq
        runtime._initial_mode_sequence = list(seq)

        recorder = None
        if args.record and hasattr(robot, 'mj_model'):
            from unitree_launcher.recording import VideoRecorder, normalize_record_path
            record_path = normalize_record_path(args.record)
            recorder = VideoRecorder(
                record_path, robot.mj_model, robot.mj_data,
                step_fn=lambda: runtime.sim_step_count,
            )

        # start() + safety.start(): sinusoid auto-activates after PREPARE
        runtime.start()
        safety.start()
        print(
            f"[main] Gantry: PREPARE(20s) → {sinusoid.joint_name} "
            f"sinusoid ({sinusoid.freq_hz}Hz, {sinusoid.amplitude:.3f}rad)"
        )

        try:
            _gui = getattr(args, 'gui', False)
            _vis = getattr(args, 'viser', False)
            if _gui:
                run_with_viewer(runtime, recorder=recorder)
            elif _vis:
                run_with_viser(
                    runtime, port=getattr(args, 'port', 8080),
                    duration=args.duration, play=True, recorder=recorder,
                    gantry=True,
                )
            else:
                run_headless(
                    runtime, duration=args.duration,
                    max_steps=getattr(args, 'steps', None),
                    recorder=recorder,
                    rate_limit=(args.mode == "real"),
                )
        finally:
            if gamepad is not None:
                gamepad.stop()
            if recorder:
                recorder.close()
            runtime.stop()
            if hasattr(robot, 'graceful_shutdown'):
                robot.graceful_shutdown()
            else:
                robot.disconnect()
        return

    # ---- Default policy ----
    if args.default_policy:
        config.policy.default_policy = args.default_policy
    default_policy, default_joint_mapper = load_default_policy(config, robot_joints)

    # ---- Active policy ----
    active_policy, active_joint_mapper = load_policy(
        args.policy, config, robot_joints, robot
    )

    # Match simulation parameters to the training environment.
    if isinstance(active_policy, BeyondMimicPolicy) and hasattr(robot, 'set_home_positions'):
        # 1. Set initial robot pose to policy's default joint positions.
        default_native = active_joint_mapper.policy_to_robot(
            active_policy.default_joint_pos
        )
        robot.set_home_positions(default_native)

        # 2. Set per-joint armature to match training sim.
        import numpy as _np
        _kp = active_policy.stiffness
        if _kp is not None:
            _natural_freq = 10.0 * 2.0 * _np.pi
            armature = _kp / (_natural_freq ** 2)
            robot.set_armature(armature)

        # 3. Override actuator kp/kv to match BM training gains.
        #    The XML defaults are IsaacLab gains (from mjlab armature formula).
        #    BM uses its own stiffness/damping from ONNX metadata.
        if hasattr(robot, 'set_actuator_gains') and _kp is not None:
            _kd = active_policy.damping
            if _kd is None:
                _kd = _np.zeros_like(_kp)
            robot.set_actuator_gains(_kp, _kd)

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
        est_verbose = getattr(args, "estimator_verbose", False)
        est_imu = getattr(args, "estimate_imu", False)
        estimator = StateEstimator(
            config, verbose=est_verbose, estimate_imu=est_imu,
        )
        mode_str = "pos+vel+imu" if est_imu else "pos+vel"
        print(f"[main] State estimator enabled (InEKF, {mode_str})"
              + (" [verbose]" if est_verbose else ""))

    # ---- Input controllers ----
    from unitree_launcher.controller.input import InputManager
    from unitree_launcher.controller.keyboard import KeyboardInput

    input_controllers = []

    wireless = None
    if args.mode == "real":
        # Wireless controller has highest priority on real robot
        from unitree_launcher.controller.wireless import WirelessInput
        wireless = WirelessInput()
        input_controllers.append(wireless)

    # Gamepad: try to connect on any mode (sim or real)
    from unitree_launcher.controller.gamepad_input import start_gamepad_input
    gamepad = start_gamepad_input()
    if gamepad:
        input_controllers.append(gamepad)

    # Keyboard (always available for sim; SSH terminal for real)
    keyboard_input = KeyboardInput()
    input_controllers.append(keyboard_input)

    if args.viser:
        from unitree_launcher.controller.viser_input import ViserInput
        input_controllers.append(ViserInput())

    input_mgr = InputManager(input_controllers)

    # ---- Runtime ----
    runtime = Runtime(
        robot=robot,
        policy=active_policy,
        safety=safety,
        joint_mapper=active_joint_mapper,
        config=config,
        logger=logger,
        policy_dir=args.policy_dir,
        default_policy=default_policy,
        default_joint_mapper=default_joint_mapper,
        input_manager=input_mgr,
    )
    if estimator is not None:
        runtime.set_estimator(estimator)

    # ---- Pre-load all policies in policy_dir for instant switching ----
    preloaded = preload_policy_dir(
        config, args.policy_dir, robot_joints, robot,
        exclude={args.policy} if args.policy else None,
    )
    runtime._preloaded_policies.update(preloaded)
    if args.policy:
        runtime._preloaded_policies[args.policy] = (
            active_policy, active_joint_mapper,
        )
        runtime._active_policy_path = args.policy

    # ---- Connect & run ----
    robot.connect()
    if wireless is not None and hasattr(robot, 'set_wireless_handler'):
        robot.set_wireless_handler(wireless.parse)
    if logger is not None:
        logger.start()

    # ---- Video recorder (created after logger.start so log dir exists) ----
    recorder = None
    if args.record and args.mode in ("sim", "eval"):
        from unitree_launcher.recording import VideoRecorder, normalize_record_path
        log_dir = logger.log_path if logger is not None else None
        record_path = normalize_record_path(args.record, directory=log_dir)
        recorder = VideoRecorder(
            record_path, robot.mj_model, robot.mj_data,
            step_fn=lambda: runtime.sim_step_count,
        )

    # Install signal handlers for SIGTERM/SIGHUP so the robot gets a clean
    # shutdown even if the process is killed or the SSH session drops.
    shutdown_requested = False

    def _signal_shutdown(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n[main] Received {sig_name}, triggering E-stop")
        if shutdown_requested:
            raise SystemExit(1)
        shutdown_requested = True
        safety.estop()
        runtime.stop()
        if hasattr(robot, 'graceful_shutdown'):
            robot.graceful_shutdown()
        else:
            robot.disconnect()
        if logger is not None:
            logger.stop()
        raise SystemExit(0)

    prev_sigterm = signal.signal(signal.SIGTERM, _signal_shutdown)
    prev_sighup = signal.signal(signal.SIGHUP, _signal_shutdown)

    # ---- Prepare phase (real robot only: smooth blend to default pose) ----
    # Runs as a PREPARE mode sequence inside the Runtime step loop,
    # so viewers, wireless E-stop, and telemetry all work during prepare.
    if args.mode == "real":
        from unitree_launcher.control.safety import ControlMode as _CM
        runtime._mode_sequence = [(_CM.PREPARE, 20.0)]
        runtime._initial_mode_sequence = [(_CM.PREPARE, 20.0)]

    use_gui = args.gui
    use_viser = args.viser
    try:
        if use_gui:
            run_with_viewer(
                runtime, recorder=recorder,
                viser_port=args.port if use_viser else None,
            )
        elif use_viser:
            _viser_paths = list(runtime._preloaded_policies.keys())
            run_with_viser(
                runtime, port=args.port,
                duration=args.duration,
                max_steps=args.steps,
                play=args.play,
                policy_paths=_viser_paths or None,
                active_policy=Path(args.policy).stem if args.policy else None,
                recorder=recorder,
            )
        else:
            run_headless(
                runtime,
                duration=args.duration,
                max_steps=args.steps,
                recorder=recorder,
                rate_limit=(args.mode == "real"),
            )
    finally:
        # Restore original signal handlers to avoid double-shutdown
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGHUP, prev_sighup)

        if gamepad is not None:
            gamepad.stop()
        if recorder:
            recorder.close()
        runtime.stop()
        if hasattr(robot, 'graceful_shutdown'):
            robot.graceful_shutdown()
        else:
            robot.disconnect()
        if logger is not None:
            logger.stop()


def _reexec_with_mjpython_args(main_args: list):
    """On macOS with --gui, re-exec using mjpython for GLFW support.

    MuJoCo's ``launch_passive`` requires the process be started via
    ``mjpython`` on macOS. This transparently re-launches the current
    command under mjpython so ``uv run sim --gui`` just works.

    Args:
        main_args: Arguments to pass to main() (e.g. ["sim", "--gui", "--policy", ...])
    """
    import os

    # Find mjpython in the venv (installed by mujoco package)
    venv_bin = Path(sys.executable).parent
    mjpython = venv_bin / "mjpython"
    if not mjpython.exists():
        print(
            "Error: --gui on macOS requires mjpython (from the mujoco package).\n"
            "It should be in your venv after 'uv sync'.\n"
            f"Expected at: {mjpython}\n"
            "Or use --viser for the web viewer instead.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # Re-exec: replace current process with mjpython running this module
    mjpython_str = str(mjpython)
    exec_args = [mjpython_str, "-m", "unitree_launcher.main"] + main_args
    os.execv(mjpython_str, exec_args)


def cli_sim():
    """Entry point for ``uv run sim`` command.

    On macOS, transparently re-execs with mjpython when --gui is used.
    """
    user_args = sys.argv[1:]  # Everything after the script name
    if platform.system() == "Darwin" and "--gui" in user_args:
        # Re-exec: mjpython -m unitree_launcher.main sim <user_args>
        _reexec_with_mjpython_args(["sim"] + user_args)
    main(["sim"] + user_args)


def cli_eval():
    """Entry point for ``unitree-eval`` command (uv script alias).

    Headless evaluation at 1000Hz physics for maximum accuracy.
    """
    main(["eval"] + sys.argv[1:])


def cli_real():
    """Entry point for ``uv run real`` command.

    Onboard robot deployment. Sets OMP_NUM_THREADS=1 on aarch64 to
    prevent thread contention on Jetson Orin.
    """
    import os
    if platform.machine().startswith("aarch64"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(["real"] + sys.argv[1:])


def cli_mirror():
    """Entry point for ``uv run mirror`` command.

    Mirrors real robot state into MuJoCo viewer or viser over DDS.
    On macOS with --gui, re-execs with mjpython for GLFW support.
    """
    user_args = sys.argv[1:]
    if platform.system() == "Darwin" and ("--gui" in user_args or "--viser" not in user_args):
        # Default to --gui if neither specified; --gui needs mjpython on macOS
        if "--gui" not in user_args and "--viser" not in user_args:
            user_args.append("--gui")
        if "--gui" in user_args:
            _reexec_with_mjpython_args(["mirror"] + user_args)
    main(["mirror"] + user_args)


if __name__ == "__main__":
    main()
