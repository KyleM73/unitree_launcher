"""Unitree G1 Deployment Stack — Main Entry Point (Metal / native macOS).

Provides:
    run_with_viewer() — interactive MuJoCo passive viewer with GLFW key callbacks
    run_headless()    — headless simulation for server/batch evaluation
    main()            — CLI entry point with argparse
"""
from __future__ import annotations

# Patch SDK threading FIRST (before any unitree_sdk2py imports).
from src.compat import patch_unitree_threading
patch_unitree_threading()

import argparse
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import mujoco.viewer

from src.config import (
    Config,
    G1_29DOF_JOINTS,
    G1_23DOF_JOINTS,
    apply_cli_overrides,
    load_config,
)
from src.control.controller import Controller
from src.control.safety import SafetyController
from src.logging.logger import DataLogger
from src.policy.base import PolicyInterface, detect_policy_format
from src.policy.beyondmimic_policy import BeyondMimicPolicy
from src.policy.isaaclab_policy import IsaacLabPolicy
from src.policy.joint_mapper import JointMapper
from src.policy.observations import ObservationBuilder
from src.robot.real_robot import RealRobot
from src.robot.sim_robot import SimRobot

# GLFW key constants.
# Avoid letters bound by MuJoCo viewer (w=wireframe, s=shadow, a=analytic,
# d=geom, e=frame, r=reflect, c=contact, n=overlay, x=connect, z=perturb).
# See: https://www.glfw.org/docs/latest/group__keys.html
GLFW_KEY_MAP = {
    32: "space",        # GLFW_KEY_SPACE  — start / stop
    265: "up",          # GLFW_KEY_UP     — vx +
    264: "down",        # GLFW_KEY_DOWN   — vx -
    263: "left",        # GLFW_KEY_LEFT   — vy +
    262: "right",       # GLFW_KEY_RIGHT  — vy -
    44: "comma",        # GLFW_KEY_COMMA  — yaw +
    46: "period",       # GLFW_KEY_PERIOD — yaw -
    47: "slash",        # GLFW_KEY_SLASH  — zero velocity
    259: "backspace",   # GLFW_KEY_BACKSPACE — e-stop
    257: "enter",       # GLFW_KEY_ENTER  — clear e-stop
    45: "minus",        # GLFW_KEY_MINUS  — prev policy
    61: "equal",        # GLFW_KEY_EQUAL  — next policy
    261: "delete",       # GLFW_KEY_DELETE (Fn+Backspace on Mac) — reset
}


# ============================================================================
# Viewer / Headless Runners (from Phase 9)
# ============================================================================

def run_with_viewer(sim_robot: SimRobot, controller: Controller) -> None:
    """Run simulation with interactive MuJoCo viewer.

    Threading model:
        Main thread   — viewer.sync() + drain key queue (this function)
        Control thread — get_state / send_command / mj_step (controller)
        Viewer thread  — GLFW rendering + event loop (MuJoCo internal)

    The key callback fires on MuJoCo's viewer thread.  To avoid
    cross-thread deadlocks with sim_robot.lock, the callback only
    enqueues key names; the main loop drains the queue *outside*
    the lock before calling viewer.sync().
    """
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
                # Drain key events (no lock held — safe to call handle_key).
                while not key_queue.empty():
                    controller.handle_key(key_queue.get_nowait())

                with sim_robot.lock:
                    viewer.sync()
                time.sleep(1.0 / 60.0)  # ~60 FPS viewer update
        except KeyboardInterrupt:
            pass
        finally:
            controller.stop()


def run_headless(
    sim_robot: SimRobot,
    controller: Controller,
    duration: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> None:
    """Run simulation without viewer (for server evals).

    Termination conditions (first one wins):
        1. Ctrl+C (KeyboardInterrupt)
        2. *duration* seconds elapsed
        3. *max_steps* policy steps completed
        4. BeyondMimic trajectory ends (controller auto-stops)

    Args:
        duration: Auto-terminate after this many seconds (None = no limit).
        max_steps: Auto-terminate after this many policy steps (None = no limit).
    """
    if max_steps is not None:
        controller._max_steps = max_steps

    controller.start()
    # Auto-start the policy (no viewer to press Space).
    controller.safety.start()

    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)

            if duration is not None and (time.time() - start_time) >= duration:
                print(f"[headless] Duration limit reached ({duration}s). Stopping.")
                break

            if not controller.is_running:
                print("[headless] Controller stopped (trajectory end or error).")
                break

    except KeyboardInterrupt:
        print("\n[headless] Ctrl+C received. Stopping.")
    finally:
        controller.stop()


# ============================================================================
# CLI (Phase 12)
# ============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between sim and real sub-commands."""
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--policy", required=True,
                        help="Path to ONNX policy file")
    parser.add_argument("--policy-dir", default=None,
                        help="Directory of ONNX files for N/P key cycling")
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser. Exposed for testing."""
    parser = argparse.ArgumentParser(
        description="Unitree G1 Deployment Stack (Metal)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- sim sub-command --
    sim_parser = subparsers.add_parser("sim", help="Simulation mode")
    _add_common_args(sim_parser)
    sim_parser.add_argument("--headless", action="store_true",
                            help="Run without viewer (for server evals)")
    sim_parser.add_argument("--duration", type=float, default=None,
                            help="Auto-stop after N seconds (headless only)")
    sim_parser.add_argument("--steps", type=int, default=None,
                            help="Auto-stop after N policy steps (headless only)")

    # -- real sub-command --
    real_parser = subparsers.add_parser("real", help="Real robot mode")
    _add_common_args(real_parser)
    real_parser.add_argument("--interface", required=True,
                             help="Network interface (e.g. eth0)")

    return parser


def main(argv: Optional[list] = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list for testing (default: sys.argv[1:]).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

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
    else:
        robot = RealRobot(config)

    # ---- Policy format & joint ordering ----
    robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS
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

    joint_mapper = JointMapper(
        robot_joints=robot_joints,
        observed_joints=observed_joints,
        controlled_joints=controlled_joints,
    )

    # ---- Policy ----
    if policy_format == "isaaclab":
        use_estimator = config.policy.use_estimator
        if args.no_est:
            use_estimator = False
        obs_builder = ObservationBuilder(
            joint_mapper, config, use_estimator=use_estimator
        )
        policy: PolicyInterface = IsaacLabPolicy(
            joint_mapper, obs_builder.observation_dim
        )
    else:
        # BeyondMimic builds its own observations internally.
        obs_builder = None
        policy = BeyondMimicPolicy(
            joint_mapper,
            obs_dim=160,  # standard BeyondMimic observation size
            use_onnx_metadata=config.policy.use_onnx_metadata,
        )

    policy.load(args.policy)

    # Match simulation parameters to the training environment.
    if isinstance(policy, BeyondMimicPolicy) and hasattr(robot, 'set_home_positions'):
        # 1. Set initial robot pose to policy's default joint positions.
        default_native = joint_mapper.action_to_robot(policy.default_joint_pos)
        robot.set_home_positions(default_native)

        # 2. Set per-joint armature to match training sim.
        #    Training uses: armature = stiffness / (natural_freq)^2
        #    where natural_freq = 10 * 2π rad/s (from g1.py reference).
        if policy.stiffness is not None:
            import numpy as _np
            _natural_freq = 10.0 * 2.0 * _np.pi
            armature_policy = policy.stiffness / (_natural_freq ** 2)
            armature_native = joint_mapper.action_to_robot(armature_policy)
            robot.set_armature(armature_native)

    # Validate obs_dim match (IsaacLab only).
    if obs_builder is not None:
        assert policy.observation_dim == obs_builder.observation_dim, (
            f"Policy obs_dim={policy.observation_dim} != "
            f"builder obs_dim={obs_builder.observation_dim}"
        )

    # ---- Safety ----
    safety = SafetyController(config, n_dof=robot.n_dof)

    # ---- Logger ----
    logger = None
    if not args.no_log:
        policy_name = Path(args.policy).stem
        run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{args.mode}_{policy_name}"
        logger = DataLogger(config.logging, run_name, args.log_dir)

    # ---- Controller ----
    controller = Controller(
        robot=robot,
        policy=policy,
        safety=safety,
        joint_mapper=joint_mapper,
        obs_builder=obs_builder,
        config=config,
        logger=logger,
        policy_dir=args.policy_dir,
    )

    # ---- Connect & run ----
    robot.connect()
    if logger is not None:
        logger.start()

    try:
        if args.mode == "sim" and not args.headless:
            run_with_viewer(robot, controller)
        else:
            run_headless(
                robot, controller,
                duration=getattr(args, "duration", None),
                max_steps=getattr(args, "steps", None),
            )
    finally:
        robot.disconnect()
        if logger is not None:
            logger.stop()


if __name__ == "__main__":
    main()
