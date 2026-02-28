"""Sim integration tests for check_state_limits() — drive joints to limits in MuJoCo.

Uses SimRobot + SafetyController (no full Controller needed) to verify that
measured state violations trigger ESTOP in actual simulation.
"""
from __future__ import annotations

import numpy as np
import pytest

from unitree_launcher.config import (
    Config,
    G1_23DOF_JOINTS,
    G1_29DOF_JOINTS,
    JOINT_LIMITS_23DOF,
    JOINT_LIMITS_29DOF,
    TORQUE_LIMITS_23DOF,
    TORQUE_LIMITS_29DOF,
    VELOCITY_LIMITS_23DOF,
    VELOCITY_LIMITS_29DOF,
    load_config,
)
from unitree_launcher.control.safety import SafetyController, SystemState
from unitree_launcher.robot.base import RobotCommand
from unitree_launcher.robot.sim_robot import SimRobot

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs" / "default.yaml")


@pytest.fixture
def sim_config():
    config = load_config(DEFAULT_CONFIG)
    config.robot.variant = "g1_29dof"
    return config


@pytest.fixture
def sim_robot(sim_config):
    return SimRobot(sim_config)


class TestSafetySimPosition:
    """Drive a joint to its position limit in MuJoCo and verify ESTOP."""

    def test_position_limit_triggers_estop(self, sim_config, sim_robot):
        safety = SafetyController(sim_config, n_dof=29)
        safety.start()

        # Command left_hip_pitch (index 0) to its max position (2.88 rad)
        # with high kp to drive it there quickly.
        pos_max = JOINT_LIMITS_29DOF["left_hip_pitch"][1]
        cmd = RobotCommand(
            joint_positions=np.zeros(29),
            joint_velocities=np.zeros(29),
            joint_torques=np.zeros(29),
            kp=np.full(29, 200.0),
            kd=np.full(29, 5.0),
        )
        cmd.joint_positions[0] = pos_max

        sim_robot.send_command(cmd)

        # Step physics multiple times to let the joint reach the limit
        for _ in range(200):
            sim_robot.step()

        state = sim_robot.get_state()
        fault = safety.check_state_limits(state)
        assert fault is True, (
            f"Expected position fault but got None. "
            f"left_hip_pitch pos={state.joint_positions[0]:.4f}, limit={pos_max}"
        )
        assert safety.state == SystemState.ESTOP


class TestSafetySimVelocity:
    """Generate high velocity in MuJoCo and verify ESTOP."""

    def test_velocity_limit_triggers_estop(self, sim_config, sim_robot):
        safety = SafetyController(sim_config, n_dof=29)
        safety.start()

        # Set very high kp and a large position step to generate high velocity.
        # Target far from current position with high gains.
        cmd = RobotCommand(
            joint_positions=np.zeros(29),
            joint_velocities=np.zeros(29),
            joint_torques=np.zeros(29),
            kp=np.full(29, 5000.0),
            kd=np.full(29, 0.1),
        )
        # Large step for left_hip_pitch
        cmd.joint_positions[0] = 2.5

        sim_robot.send_command(cmd)

        # Check after a few substeps — the high kp should create a velocity spike
        triggered = False
        for _ in range(50):
            sim_robot.step()
            state = sim_robot.get_state()
            if safety.check_state_limits(state):
                triggered = True
                break

        assert triggered, (
            f"Expected velocity fault. "
            f"left_hip_pitch vel={state.joint_velocities[0]:.4f}, "
            f"limit={VELOCITY_LIMITS_29DOF['left_hip_pitch']}"
        )
        assert safety.state == SystemState.ESTOP


class TestSafetySimTorque:
    """Generate high torque in MuJoCo and verify ESTOP."""

    def test_torque_limit_triggers_estop(self, sim_config, sim_robot):
        safety = SafetyController(sim_config, n_dof=29)
        safety.start()

        # Very high kp + large position error → torque sensor exceeds limit.
        cmd = RobotCommand(
            joint_positions=np.zeros(29),
            joint_velocities=np.zeros(29),
            joint_torques=np.zeros(29),
            kp=np.full(29, 5000.0),
            kd=np.full(29, 0.1),
        )
        cmd.joint_positions[0] = 2.5

        sim_robot.send_command(cmd)

        triggered = False
        for _ in range(50):
            sim_robot.step()
            state = sim_robot.get_state()
            if safety.check_state_limits(state):
                triggered = True
                break

        assert triggered, (
            f"Expected torque fault. "
            f"left_hip_pitch torque={state.joint_torques[0]:.4f}, "
            f"limit={TORQUE_LIMITS_29DOF['left_hip_pitch']}"
        )
        assert safety.state == SystemState.ESTOP


class TestOrientationEstop:
    """Directly manipulate the free joint quaternion to simulate inversion."""

    def test_orientation_estop_in_sim(self, sim_config, sim_robot):
        import mujoco
        safety = SafetyController(sim_config, n_dof=29)
        safety.start()

        # Rotate pelvis to be inverted (180° about X axis)
        # Free joint quat is at qpos[3:7] in wxyz format
        sim_robot.mj_data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # 180° X rotation
        mujoco.mj_forward(sim_robot.mj_model, sim_robot.mj_data)

        state = sim_robot.get_state()
        is_safe, msg = safety.check_orientation(state.imu_quaternion)
        assert is_safe is False, f"Inverted robot should trigger orientation fault: {msg}"

        # Trigger ESTOP via check_orientation path
        if not is_safe:
            safety.estop()
        assert safety.state == SystemState.ESTOP


class TestClamp23DOF:
    """Verify clamp_command works correctly with 23-DOF config."""

    def test_clamp_command_23dof(self):
        config_path = str(PROJECT_ROOT / "configs" / "g1_23dof.yaml")
        config = load_config(config_path)
        safety = SafetyController(config, n_dof=23)

        # Create command exceeding all limits
        cmd = RobotCommand(
            joint_positions=np.full(23, 100.0),
            joint_velocities=np.full(23, 100.0),
            joint_torques=np.full(23, 999.0),
            kp=np.full(23, 100.0),
            kd=np.full(23, 10.0),
        )
        result = safety.clamp_command(cmd)

        # Verify position clamped
        for i, name in enumerate(G1_23DOF_JOINTS):
            _, hi = JOINT_LIMITS_23DOF[name]
            assert result.joint_positions[i] == pytest.approx(hi), \
                f"Joint {name} not clamped to max position"

        # Verify velocity clamped
        for i, name in enumerate(G1_23DOF_JOINTS):
            assert result.joint_velocities[i] == pytest.approx(VELOCITY_LIMITS_23DOF[name]), \
                f"Joint {name} velocity not clamped"

        # Verify torque clamped
        for i, name in enumerate(G1_23DOF_JOINTS):
            assert result.joint_torques[i] == pytest.approx(TORQUE_LIMITS_23DOF[name]), \
                f"Joint {name} torque not clamped"
