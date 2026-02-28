"""Validate the InEKF state estimator against MuJoCo ground truth.

Runs the full control loop with the MuJoCo viewer (same as the normal
``mjpython -m unitree_launcher.main sim --policy ... --estimator`` flow).
Records ground-truth vs estimated states at each control tick — inside
the estimator itself so there is no threading race — and produces a
multi-panel matplotlib plot.

Press Space to start the BM policy (just like the normal viewer).
Close the viewer window to stop and generate the plot.

Usage:
    mjpython scripts/validate_estimator.py [--duration 10] [--output estimator_validation.png]
"""
from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Patch SDK threading before any unitree imports.
from unitree_launcher.compat import patch_unitree_threading
patch_unitree_threading()

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    load_config,
)
from unitree_launcher.control.controller import Controller
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.estimation import StateEstimator
from unitree_launcher.estimation.state_estimator import _WARMUP_TICKS, _BLEND_TICKS
from unitree_launcher.policy.base import detect_policy_format
from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy
from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
from unitree_launcher.policy.joint_mapper import JointMapper
from unitree_launcher.policy.observations import ObservationBuilder
from unitree_launcher.robot.base import RobotState
from unitree_launcher.robot.sim_robot import SimRobot

# GLFW key constants (same as main.py).
GLFW_KEY_MAP = {
    32: "space", 265: "up", 264: "down", 263: "left", 262: "right",
    44: "comma", 46: "period", 47: "slash", 259: "backspace",
    257: "enter", 45: "minus", 61: "equal", 261: "delete",
}


# ---------------------------------------------------------------------------
# Recording wrapper — captures data at the exact moment the estimator runs
# ---------------------------------------------------------------------------

class RecordingEstimator(StateEstimator):
    """StateEstimator subclass that records GT vs estimate at each tick."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.log_time: List[float] = []
        self.log_gt_pos: List[np.ndarray] = []
        self.log_gt_vel: List[np.ndarray] = []
        self.log_ekf_pos: List[np.ndarray] = []
        self.log_ekf_vel: List[np.ndarray] = []
        self.log_leg_vel: List[np.ndarray] = []
        self.log_out_pos: List[np.ndarray] = []
        self.log_out_vel: List[np.ndarray] = []
        self.log_alpha: List[float] = []
        self.log_left_contact: List[bool] = []
        self.log_right_contact: List[bool] = []
        self._last_raw_state: Optional[RobotState] = None

    def update(self, robot_state: RobotState, dt=None):
        self._last_raw_state = robot_state
        super().update(robot_state, dt)

    def populate_robot_state(self, robot_state: RobotState) -> RobotState:
        result = super().populate_robot_state(robot_state)

        # Record at the exact moment data flows to the policy
        raw = robot_state
        self.log_time.append(raw.timestamp)
        self.log_gt_pos.append(raw.base_position.copy())
        self.log_gt_vel.append(raw.base_velocity.copy())

        if self._ekf.initialized:
            self.log_ekf_pos.append(self._ekf.position.copy())
            self.log_ekf_vel.append(self._ekf.velocity.copy())
        else:
            self.log_ekf_pos.append(np.full(3, np.nan))
            self.log_ekf_vel.append(np.full(3, np.nan))

        self.log_leg_vel.append(self._leg_velocity.copy())
        self.log_out_pos.append(result.base_position.copy())
        self.log_out_vel.append(result.base_velocity.copy())
        self.log_alpha.append(self._blend_alpha())
        self.log_left_contact.append(self.left_contact)
        self.log_right_contact.append(self.right_contact)

        return result

    def get_arrays(self):
        return {
            "time": np.array(self.log_time),
            "gt_pos": np.array(self.log_gt_pos),
            "gt_vel": np.array(self.log_gt_vel),
            "ekf_pos": np.array(self.log_ekf_pos),
            "ekf_vel": np.array(self.log_ekf_vel),
            "leg_vel": np.array(self.log_leg_vel),
            "out_pos": np.array(self.log_out_pos),
            "out_vel": np.array(self.log_out_vel),
            "alpha": np.array(self.log_alpha),
            "left_contact": np.array(self.log_left_contact, dtype=float),
            "right_contact": np.array(self.log_right_contact, dtype=float),
        }


# ---------------------------------------------------------------------------
# Setup (mirrors main.py)
# ---------------------------------------------------------------------------

def build_sim(policy_path: str):
    """Build all components, same as main.py."""
    config = load_config("configs/default.yaml")
    robot = SimRobot(config)
    robot_joints = G1_29DOF_JOINTS

    # Default stance policy
    default_policy = None
    default_obs_builder = None
    default_joint_mapper = None
    default_policy_path = config.policy.default_policy

    if default_policy_path and Path(default_policy_path).exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(default_policy_path, providers=["CPUExecutionProvider"])
        model_obs_dim = sess.get_inputs()[0].shape[1]
        n_actions = sess.get_outputs()[0].shape[1]
        del sess

        il_joints = [j.replace("_joint", "") for j in ISAACLAB_G1_29DOF_JOINTS[:n_actions]]
        default_joint_mapper = JointMapper(
            robot_joints=robot_joints, observed_joints=il_joints, controlled_joints=il_joints,
        )
        for use_est in [True, False]:
            default_obs_builder = ObservationBuilder(default_joint_mapper, config, use_estimator=use_est)
            if default_obs_builder.observation_dim == model_obs_dim:
                break
        default_policy = IsaacLabPolicy(default_joint_mapper, model_obs_dim)
        default_policy.load(default_policy_path)
        print(f"[val] Default policy: {default_policy_path} (obs_dim={model_obs_dim})")

    # Active policy (BeyondMimic)
    metadata = BeyondMimicPolicy.load_metadata(policy_path)
    if "joint_names" in metadata:
        policy_joints = [j.strip().replace("_joint", "") for j in metadata["joint_names"].split(",")]
    else:
        policy_joints = None

    active_mapper = JointMapper(
        robot_joints=robot_joints, observed_joints=policy_joints, controlled_joints=policy_joints,
    )
    active_policy = BeyondMimicPolicy(active_mapper, obs_dim=160, use_onnx_metadata=True)
    active_policy.load(policy_path)

    # Match sim to training dynamics
    if hasattr(robot, "set_home_positions"):
        robot.set_home_positions(active_mapper.action_to_robot(active_policy.default_joint_pos))
    if active_policy.stiffness is not None:
        if hasattr(robot, "set_armature"):
            robot.set_armature(active_mapper.action_to_robot(
                active_policy.stiffness / (10.0 * 2.0 * np.pi) ** 2))
        if hasattr(robot, "set_actuator_gains"):
            kd = active_policy.damping if active_policy.damping is not None else np.zeros_like(active_policy.stiffness)
            robot.set_actuator_gains(
                active_mapper.action_to_robot(active_policy.stiffness),
                active_mapper.action_to_robot(kd))

    safety = SafetyController(config, n_dof=robot.n_dof)
    estimator = RecordingEstimator(config)

    controller = Controller(
        robot=robot, policy=active_policy, safety=safety,
        joint_mapper=active_mapper, obs_builder=None, config=config,
        default_policy=default_policy,
        default_obs_builder=default_obs_builder,
        default_joint_mapper=default_joint_mapper,
    )
    controller.set_estimator(estimator)

    robot.connect()
    return robot, controller, safety, estimator


# ---------------------------------------------------------------------------
# Viewer loop (mirrors main.py run_with_viewer)
# ---------------------------------------------------------------------------

def run_with_viewer(robot, controller, safety, stance_before=5.0, stance_after=5.0, recorder=None):
    """Run with MuJoCo viewer.

    Automated sequence:
        1. Stance (default policy) for ``stance_before`` seconds
        2. Auto-start BM policy (simulates pressing Space)
        3. BM walks until trajectory ends (controller auto-returns to stance)
        4. Stance for ``stance_after`` seconds
        5. Stop and close viewer

    Manual keys still work (Space, Backspace, etc.) if you want to
    override the automation.
    """
    import mujoco.viewer

    key_queue: queue.SimpleQueue = queue.SimpleQueue()

    def key_callback(keycode):
        key = GLFW_KEY_MAP.get(keycode)
        if key:
            key_queue.put(key)

    start = time.time()
    bm_started = False
    bm_ended = False
    bm_end_time = None

    with mujoco.viewer.launch_passive(
        robot.mj_model, robot.mj_data, key_callback=key_callback,
    ) as viewer:
        controller.start()
        try:
            while viewer.is_running():
                # Drain manual key events
                while not key_queue.empty():
                    controller.handle_key(key_queue.get_nowait())

                with robot.lock:
                    viewer.sync()
                # Capture OUTSIDE the lock — never block the control thread.
                if recorder:
                    recorder.capture(robot.mj_data)

                elapsed = time.time() - start

                # Phase 1 → 2: auto-start BM after stance_before seconds
                if not bm_started and elapsed >= stance_before:
                    bm_started = True
                    safety.start()
                    print(f"[val] t={elapsed:.1f}s: Auto-starting BM policy")

                # Phase 2 → 3: detect BM trajectory end
                if bm_started and not bm_ended:
                    if safety.state == SystemState.STOPPED:
                        bm_ended = True
                        bm_end_time = time.time()
                        print(f"[val] t={elapsed:.1f}s: BM trajectory ended, "
                              f"stance for {stance_after:.0f}s more")

                # Phase 3 → done: stop after stance_after seconds post-BM
                if bm_ended and (time.time() - bm_end_time) >= stance_after:
                    print(f"[val] t={elapsed:.1f}s: Post-BM stance complete. Stopping.")
                    break

                time.sleep(1.0 / 100.0)
        except KeyboardInterrupt:
            pass
        finally:
            controller.stop()


def run_headless(robot, controller, safety, stance_before=5.0, stance_after=5.0, recorder=None):
    """Run without viewer (same automated sequence as run_with_viewer)."""
    start = time.time()
    bm_started = False
    bm_ended = False
    bm_end_time = None

    controller.start()
    try:
        while True:
            # Capture outside any lock — never block the control thread.
            if recorder:
                recorder.capture(robot.mj_data)

            elapsed = time.time() - start

            if not bm_started and elapsed >= stance_before:
                bm_started = True
                safety.start()
                print(f"[val] t={elapsed:.1f}s: Auto-starting BM policy")

            if bm_started and not bm_ended:
                if safety.state == SystemState.STOPPED:
                    bm_ended = True
                    bm_end_time = time.time()
                    print(f"[val] t={elapsed:.1f}s: BM trajectory ended, "
                          f"stance for {stance_after:.0f}s more")

            if bm_ended and (time.time() - bm_end_time) >= stance_after:
                print(f"[val] t={elapsed:.1f}s: Post-BM stance complete. Stopping.")
                break

            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(data: dict, output_path: str):
    """Create a multi-panel comparison plot."""
    ts = data["time"]
    if len(ts) == 0:
        print("[val] No data recorded — did you run long enough?")
        return

    # Make time relative to first sample
    t0 = ts[0]
    ts = ts - t0

    gt_pos = data["gt_pos"]; gt_vel = data["gt_vel"]
    ekf_pos = data["ekf_pos"]; ekf_vel = data["ekf_vel"]
    leg_vel = data["leg_vel"]
    out_pos = data["out_pos"]; out_vel = data["out_vel"]
    alpha = data["alpha"]
    left_c = data["left_contact"]; right_c = data["right_contact"]

    warmup_t = _WARMUP_TICKS * 0.02
    blend_end_t = warmup_t + _BLEND_TICKS * 0.02

    # Detect BM phase from velocity magnitude (walking has vel > 0.3)
    vel_mag = np.linalg.norm(gt_vel, axis=1)
    bm_mask = vel_mag > 0.15
    bm_start_t = ts[np.argmax(bm_mask)] if np.any(bm_mask) else None
    bm_end_idx = len(ts) - 1 - np.argmax(bm_mask[::-1]) if np.any(bm_mask) else None
    bm_end_t = ts[bm_end_idx] if bm_end_idx is not None else None

    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True,
                              gridspec_kw={"hspace": 0.12})
    fig.suptitle("InEKF State Estimator vs MuJoCo Ground Truth", fontsize=14, y=0.99)

    gt_c = "#1976D2"; out_c = "#E65100"; ekf_c = "#C62828"; leg_c = "#2E7D32"

    def shade(ax):
        ax.axvspan(0, warmup_t, alpha=0.07, color="red")
        ax.axvspan(warmup_t, blend_end_t, alpha=0.07, color="orange")
        ax.axvline(warmup_t, color="gray", ls="--", lw=0.6, alpha=0.4)
        ax.axvline(blend_end_t, color="gray", ls="--", lw=0.6, alpha=0.4)
        if bm_start_t is not None:
            ax.axvline(bm_start_t, color="purple", ls="-.", lw=1.0, alpha=0.5)
        if bm_end_t is not None:
            ax.axvline(bm_end_t, color="purple", ls="-.", lw=1.0, alpha=0.5)
        if bm_start_t is not None and bm_end_t is not None:
            ax.axvspan(bm_start_t, bm_end_t, alpha=0.04, color="purple")

    labels = ["x", "y", "z"]

    # --- Panel 1: Position ---
    ax = axes[0]
    shade(ax)
    for i in range(3):
        ax.plot(ts, gt_pos[:, i], color=gt_c, lw=1.5 if i==2 else 0.8, alpha=0.9)
        ax.plot(ts, out_pos[:, i], color=out_c, lw=1.5 if i==2 else 0.8, ls="--", alpha=0.9)
    # Dummy entries for legend
    ax.plot([], [], color=gt_c, lw=1.5, label="Ground truth")
    ax.plot([], [], color=out_c, lw=1.5, ls="--", label="Estimator output")
    ax.plot([], [], color=ekf_c, lw=0.8, alpha=0.4, label="EKF raw (internal)")
    for i in range(3):
        ax.plot(ts, ekf_pos[:, i], color=ekf_c, lw=0.5, alpha=0.3)
    ax.set_ylabel("Position (m)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_title("Base Position (x, y, z)", fontsize=10, loc="left")

    # --- Panel 2: Velocity ---
    ax = axes[1]
    shade(ax)
    for i in range(3):
        ax.plot(ts, gt_vel[:, i], color=gt_c, lw=1.5 if i==2 else 0.8, alpha=0.9)
        ax.plot(ts, out_vel[:, i], color=out_c, lw=1.5 if i==2 else 0.8, ls="--", alpha=0.9)
        ax.plot(ts, leg_vel[:, i], color=leg_c, lw=0.5, alpha=0.3)
    ax.plot([], [], color=gt_c, lw=1.5, label="Ground truth")
    ax.plot([], [], color=out_c, lw=1.5, ls="--", label="Estimator output")
    ax.plot([], [], color=leg_c, lw=0.8, alpha=0.5, label="Leg measurement")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_title("Base Velocity (x, y, z)", fontsize=10, loc="left")

    # --- Panel 3: Errors ---
    ax = axes[2]
    shade(ax)
    pos_err = np.linalg.norm(out_pos - gt_pos, axis=1)
    vel_err = np.linalg.norm(out_vel - gt_vel, axis=1)
    ax.plot(ts, pos_err, color=ekf_c, lw=1.2, label="Position error (m)")
    ax.plot(ts, vel_err, color=out_c, lw=1.2, label="Velocity error (m/s)")
    ax.set_ylabel("Error magnitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Estimation Errors (norm)", fontsize=10, loc="left")
    ax.set_ylim(bottom=0, top=max(0.5, np.nanpercentile(pos_err, 99) * 1.2))

    # --- Panel 4: Blend alpha + contacts ---
    ax = axes[3]
    shade(ax)
    ax.fill_between(ts, 0, alpha, alpha=0.25, color="orange")
    ax.plot(ts, alpha, color="orange", lw=1.2, label="Blend alpha")
    ax.step(ts, left_c * 0.4 + 0.55, color=gt_c, lw=0.8, where="post", label="Left contact")
    ax.step(ts, right_c * 0.4 + 0.05, color=ekf_c, lw=0.8, where="post", label="Right contact")
    ax.set_ylabel("Alpha / Contact")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_title("Blend Alpha & Contact Detection", fontsize=10, loc="left")

    # --- Panel 5: Height comparison ---
    ax = axes[4]
    shade(ax)
    ax.plot(ts, gt_pos[:, 2], color=gt_c, lw=2.0, label="GT height (z)")
    ax.plot(ts, out_pos[:, 2], color=out_c, lw=2.0, ls="--", label="Estimated height (z)")
    ax.fill_between(ts, gt_pos[:, 2], out_pos[:, 2], alpha=0.15, color=ekf_c)
    ax.set_ylabel("Height (m)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Pelvis Height", fontsize=10, loc="left")

    # Phase labels on top panel
    axes[0].text(warmup_t * 0.5, 0.97, "warmup", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontsize=8, color="red", alpha=0.7)
    axes[0].text((warmup_t + blend_end_t) / 2, 0.97, "blend",
                 transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontsize=8, color="#E65100", alpha=0.7)
    if bm_start_t is not None and bm_end_t is not None:
        axes[0].text((bm_start_t + bm_end_t) / 2, 0.97, "BM walking",
                     transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=9, color="purple", alpha=0.8,
                     fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[val] Plot saved to {output_path}")
    plt.close()

    # Print summary statistics per phase
    active = alpha >= 1.0
    print(f"\n{'='*60}")
    print(f"  Estimator Validation Summary")
    print(f"{'='*60}")

    def print_phase(name, mask):
        if not np.any(mask):
            return
        pe = pos_err[mask]; ve = vel_err[mask]
        h = gt_pos[mask, 2]
        print(f"\n  {name}:")
        print(f"    Position error: mean={pe.mean():.4f} m, max={pe.max():.4f} m")
        print(f"    Velocity error: mean={ve.mean():.4f} m/s, max={ve.max():.4f} m/s")
        print(f"    Height range:   {h.min():.3f} – {h.max():.3f} m "
              f"({'upright' if h.min() > 0.5 else 'FELL'})")

    if bm_start_t is not None and bm_end_t is not None:
        pre_bm = active & (ts < bm_start_t)
        during_bm = active & (ts >= bm_start_t) & (ts <= bm_end_t)
        post_bm = active & (ts > bm_end_t)
        print_phase("Pre-BM stance", pre_bm)
        print_phase("BM walking", during_bm)
        print_phase("Post-BM stance", post_bm)
    elif np.any(active):
        print_phase("Full estimator (alpha=1)", active)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate InEKF estimator vs MuJoCo GT")
    parser.add_argument("--stance-before", type=float, default=5.0,
                        help="Seconds of stance before BM starts (default: 5)")
    parser.add_argument("--stance-after", type=float, default=5.0,
                        help="Seconds of stance after BM ends (default: 5)")
    parser.add_argument("--policy", default="assets/policies/beyondmimic_29dof.onnx",
                        help="Path to BM policy ONNX file")
    parser.add_argument("--headless", action="store_true",
                        help="Run without MuJoCo viewer")
    parser.add_argument("--output", "-o", default="estimator_validation.png",
                        help="Output PNG path (default: estimator_validation.png)")
    parser.add_argument("--record", nargs="?", const="sim.mp4", default=None,
                        metavar="PATH",
                        help="Record video to MP4 (default: recording.mp4)")
    args = parser.parse_args()

    robot, controller, safety, estimator = build_sim(args.policy)

    print(f"[val] Sequence: {args.stance_before}s stance -> BM trajectory -> "
          f"{args.stance_after}s stance -> plot")
    if not args.headless:
        print(f"[val] Viewer will open. Close window early to stop & generate plot.")

    # ---- Video recorder ----
    recorder = None
    if args.record:
        from unitree_launcher.recording import VideoRecorder, normalize_record_path
        record_path = normalize_record_path(args.record)
        recorder = VideoRecorder(
            record_path, robot.mj_model, robot.mj_data,
            step_fn=lambda: controller.sim_step_count,
        )

    run_fn = run_headless if args.headless else run_with_viewer
    try:
        run_fn(robot, controller, safety,
               stance_before=args.stance_before,
               stance_after=args.stance_after,
               recorder=recorder)
    finally:
        if recorder:
            recorder.close()
        robot.disconnect()

    data = estimator.get_arrays()
    n = len(data["time"])
    print(f"\n[val] Recorded {n} estimator ticks ({n * 0.02:.1f}s of data).")

    if n > 10:
        make_plot(data, args.output)
    else:
        print("[val] Too few samples to plot. Run longer next time.")


if __name__ == "__main__":
    main()
