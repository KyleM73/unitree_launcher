"""Policy factory — centralised loading, format detection, and gain overrides.

Three public functions:

    load_policy()          — load a single ONNX policy (auto-detect format)
    load_default_policy()  — load the default stance/velocity policy (HoldPolicy fallback)
    preload_policy_dir()   — glob a directory and load all ONNX files
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import numpy as np

from unitree_launcher.config import (
    G1_29DOF_JOINTS,
    G1_23DOF_JOINTS,
    ISAACLAB_G1_29DOF_JOINTS,
    Config,
)
from unitree_launcher.policy.base import Policy, detect_policy_format
from unitree_launcher.policy.beyondmimic_policy import BeyondMimicPolicy
from unitree_launcher.policy.isaaclab_policy import IsaacLabPolicy
from unitree_launcher.policy.joint_mapper import JointMapper

if TYPE_CHECKING:
    from unitree_launcher.robot.base import Robot


def load_policy(
    onnx_path: str,
    config: Config,
    robot_joints: Optional[list[str]] = None,
    robot: Optional['Robot'] = None,
) -> Tuple[Policy, JointMapper]:
    """Load a single ONNX policy, auto-detecting format.

    Args:
        onnx_path: Path to the ONNX file.
        config: Full configuration.
        robot_joints: Robot joint names (defaults to variant joints).
        robot: Robot instance for BeyondMimic anchor body lookups.

    Returns:
        (policy, joint_mapper) tuple.
    """
    if robot_joints is None:
        variant = config.robot.variant
        robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS

    fmt = config.policy.format or detect_policy_format(onnx_path)

    use_estimator = config.policy.use_estimator

    if fmt == "isaaclab":
        policy, mapper = _load_isaaclab(onnx_path, config, robot_joints, use_estimator)
    else:
        policy, mapper = _load_beyondmimic(onnx_path, config, robot_joints, robot)

    _apply_gain_overrides(policy, mapper, config)
    return policy, mapper


def load_default_policy(
    config: Config,
    robot_joints: Optional[list[str]] = None,
) -> Tuple[Policy, JointMapper]:
    """Load the default stance/velocity-tracking policy.

    Falls back to HoldPolicy if the ONNX file is missing or fails to load.

    Returns:
        (policy, joint_mapper) tuple.
    """
    if robot_joints is None:
        variant = config.robot.variant
        robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS

    default_path = config.policy.default_policy
    if default_path and Path(default_path).exists():
        try:
            return _load_default_isaaclab(default_path, config, robot_joints)
        except Exception as exc:
            print(
                f"[factory] WARNING: Could not load default policy: {exc}. "
                f"Using static hold mode."
            )

    elif default_path:
        print(
            f"[factory] WARNING: Default policy not found: {default_path}. "
            f"Using static hold mode."
        )

    from unitree_launcher.policy.hold_policy import HoldPolicy
    mapper = JointMapper(robot_joints)
    policy = HoldPolicy(mapper, config)
    return policy, mapper


def preload_policy_dir(
    config: Config,
    policy_dir: str,
    robot_joints: Optional[list[str]] = None,
    robot: Optional['Robot'] = None,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, Tuple[Policy, JointMapper]]:
    """Load all ONNX files in a directory for instant switching.

    Args:
        config: Full configuration.
        policy_dir: Directory to glob for ``*.onnx`` files.
        robot_joints: Robot joint names.
        robot: Robot instance for BeyondMimic policies.
        exclude: Paths to skip (already loaded).

    Returns:
        ``{path: (policy, mapper)}`` dict.
    """
    if robot_joints is None:
        variant = config.robot.variant
        robot_joints = G1_29DOF_JOINTS if "29" in variant else G1_23DOF_JOINTS

    exclude = exclude or set()
    result: Dict[str, Tuple[Policy, JointMapper]] = {}

    policy_files = sorted(Path(policy_dir).glob("*.onnx"))
    for pf in policy_files:
        path_str = str(pf)
        if path_str in exclude:
            continue
        try:
            pol, mapper = load_policy(path_str, config, robot_joints, robot)
            result[path_str] = (pol, mapper)
            print(f"[factory] Pre-loaded policy: {pf.stem}")
        except Exception as exc:
            print(f"[factory] WARNING: Could not pre-load {pf.name}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_isaaclab(
    onnx_path: str,
    config: Config,
    robot_joints: list[str],
    use_estimator: bool,
) -> Tuple[IsaacLabPolicy, JointMapper]:
    """Load an IsaacLab policy from ONNX."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    n_actions = sess.get_outputs()[0].shape[1]
    del sess

    il_joints = [
        j.replace("_joint", "")
        for j in ISAACLAB_G1_29DOF_JOINTS[:n_actions]
    ]
    mapper = JointMapper(robot_joints=robot_joints, policy_joints=il_joints)

    # Try both estimator settings to match obs_dim
    for use_est in [use_estimator, not use_estimator]:
        policy = IsaacLabPolicy(mapper, config, use_estimator=use_est)
        try:
            policy.load(onnx_path)
            return policy, mapper
        except ValueError:
            continue

    raise ValueError(
        f"Cannot match IsaacLab policy obs_dim (n_actions={n_actions})"
    )


def _load_default_isaaclab(
    onnx_path: str,
    config: Config,
    robot_joints: list[str],
) -> Tuple[IsaacLabPolicy, JointMapper]:
    """Load the default policy as IsaacLab with estimator fallback."""
    policy, mapper = _load_isaaclab(
        onnx_path, config, robot_joints, config.policy.use_estimator
    )
    print(
        f"[factory] Default policy loaded: {onnx_path} "
        f"(obs_dim={policy.observation_dim})"
    )
    return policy, mapper


def _load_beyondmimic(
    onnx_path: str,
    config: Config,
    robot_joints: list[str],
    robot: Optional['Robot'] = None,
) -> Tuple[BeyondMimicPolicy, JointMapper]:
    """Load a BeyondMimic policy from ONNX with metadata."""
    import onnxruntime as ort

    metadata = BeyondMimicPolicy.load_metadata(onnx_path)

    policy_joints = None
    if "joint_names" in metadata:
        policy_joints = [
            j.strip().replace("_joint", "")
            for j in metadata["joint_names"].split(",")
        ]

    # Read obs_dim from the ONNX model (not hardcoded).
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    obs_input = next(i for i in sess.get_inputs() if i.name == "obs")
    obs_dim = obs_input.shape[1]
    del sess

    mapper = JointMapper(robot_joints=robot_joints, policy_joints=policy_joints)
    policy = BeyondMimicPolicy(
        mapper, obs_dim=obs_dim,
        use_onnx_metadata=config.policy.use_onnx_metadata,
        config=config,
    )
    policy.load(onnx_path)

    if robot is not None:
        policy.set_robot(robot)

    return policy, mapper


def _apply_gain_overrides(
    policy: Policy,
    mapper: JointMapper,
    config: Config,
) -> None:
    """Apply config.control.kp/kd/ka overrides when not None.

    Runs after policy.load() so it trumps ONNX metadata.
    """
    n = mapper.n_policy

    if config.control.kp is not None:
        policy._kp = _expand_gain(config.control.kp, n)

    if config.control.kd is not None:
        policy._kd = _expand_gain(config.control.kd, n)

    if config.control.ka is not None:
        policy._action_scale = _expand_gain(config.control.ka, n)


def _expand_gain(value, n: int) -> np.ndarray:
    """Expand a scalar or list gain to an array of length n."""
    if isinstance(value, (int, float)):
        return np.full(n, value, dtype=np.float64)
    return np.array(value, dtype=np.float64)
