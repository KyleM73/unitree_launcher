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

# GLFW key constants — printable ASCII chars match their ASCII values.
# See: https://www.glfw.org/docs/latest/group__keys.html
GLFW_KEY_MAP = {
    32: "space",   # GLFW_KEY_SPACE
    65: "a",       # GLFW_KEY_A
    67: "c",       # GLFW_KEY_C
    68: "d",       # GLFW_KEY_D
    69: "e",       # GLFW_KEY_E
    78: "n",       # GLFW_KEY_N
    80: "p",       # GLFW_KEY_P
    81: "q",       # GLFW_KEY_Q
    82: "r",       # GLFW_KEY_R
    83: "s",       # GLFW_KEY_S
    87: "w",       # GLFW_KEY_W
    88: "x",       # GLFW_KEY_X
    90: "z",       # GLFW_KEY_Z
}


# ============================================================================
# Viewer / Headless Runners (from Phase 9)
# ============================================================================

def run_with_viewer(sim_robot: SimRobot, controller: Controller) -> None:
    """Run simulation with interactive MuJoCo viewer.

    The viewer runs in the main thread.  The control loop runs in a
    background thread (started by controller.start()).  MuJoCo's
    launch_passive handles its own thread-safety for rendering.

    Key callback fires on key PRESS only (not repeat or release).
    """

    def key_callback(keycode: int) -> None:
        key = GLFW_KEY_MAP.get(keycode)
        if key:
            controller.handle_key(key)

    with mujoco.viewer.launch_passive(
        sim_robot.mj_model,
        sim_robot.mj_data,
        key_callback=key_callback,
    ) as viewer:
        controller.start()
        try:
            while viewer.is_running():
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

    # ---- Joint mapper ----
    robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS
    joint_mapper = JointMapper(
        robot_joints=robot_joints,
        observed_joints=config.policy.observed_joints,
        controlled_joints=config.policy.controlled_joints,
    )

    # ---- Policy ----
    policy_format = config.policy.format or detect_policy_format(args.policy)

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
