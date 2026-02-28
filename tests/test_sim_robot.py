"""Tests for SimRobot (Phase 7).

Tests cover: init, n_dof, get_state shape, reset, step, gravity,
damping holds, send_command, IMU upright, connect/disconnect,
Metal-specific viewer properties, impedance control values,
sensor mapping correctness, 23-DOF variant, substep count,
base position, reset with custom state, and DDS publish mock.
"""
import threading
import time
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from unitree_launcher.config import (
    Config,
    G1_29DOF_JOINTS,
    G1_29DOF_MUJOCO_JOINTS,
    Q_HOME_29DOF,
    load_config,
)
from unitree_launcher.robot.base import RobotCommand, RobotState
from unitree_launcher.robot.sim_robot import SimRobot

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Default 29-DOF config."""
    return load_config("configs/default.yaml")


@pytest.fixture
def config_23dof():
    """23-DOF config."""
    return load_config("configs/g1_23dof.yaml")


@pytest.fixture
def robot(default_config):
    """SimRobot with 29-DOF default config (no DDS)."""
    return SimRobot(default_config)


@pytest.fixture
def robot_23dof(config_23dof):
    """SimRobot with 23-DOF config (no DDS)."""
    return SimRobot(config_23dof)


# ---------------------------------------------------------------------------
# Test: init
# ---------------------------------------------------------------------------

def test_sim_robot_init(robot):
    """SimRobot initializes without errors."""
    assert robot is not None
    assert robot.mj_model is not None
    assert robot.mj_data is not None


# ---------------------------------------------------------------------------
# Test: n_dof
# ---------------------------------------------------------------------------

def test_sim_robot_n_dof_29(robot):
    """29-DOF config yields n_dof=29."""
    assert robot.n_dof == 29


# ---------------------------------------------------------------------------
# Test: get_state shape
# ---------------------------------------------------------------------------

def test_sim_robot_get_state_shape(robot):
    """get_state() returns correctly shaped arrays."""
    state = robot.get_state()
    assert state.joint_positions.shape == (29,)
    assert state.joint_velocities.shape == (29,)
    assert state.joint_torques.shape == (29,)
    assert state.imu_quaternion.shape == (4,)
    assert state.imu_angular_velocity.shape == (3,)
    assert state.imu_linear_acceleration.shape == (3,)
    assert state.base_position.shape == (3,)
    assert state.base_velocity.shape == (3,)


# ---------------------------------------------------------------------------
# Test: reset
# ---------------------------------------------------------------------------

def test_sim_robot_reset(robot):
    """reset() restores initial state and zeros time."""
    # Step to advance time
    robot.step()
    assert robot.mj_data.time > 0.0

    robot.reset()
    assert robot.mj_data.time == 0.0
    # ctrl should be zeroed
    np.testing.assert_array_equal(robot.mj_data.ctrl, 0.0)


# ---------------------------------------------------------------------------
# Test: step
# ---------------------------------------------------------------------------

def test_sim_robot_step(robot):
    """step() advances simulation time."""
    t0 = robot.mj_data.time
    robot.step()
    t1 = robot.mj_data.time
    assert t1 > t0


# ---------------------------------------------------------------------------
# Test: gravity
# ---------------------------------------------------------------------------

def test_sim_robot_gravity(robot):
    """Robot falls under gravity when no commands are applied."""
    robot.reset()
    initial_z = robot.get_state().base_position[2]

    # Step several policy ticks with zero ctrl
    for _ in range(50):
        robot.step()

    final_z = robot.get_state().base_position[2]
    assert final_z < initial_z, "Robot should fall under gravity"


# ---------------------------------------------------------------------------
# Test: damping holds
# ---------------------------------------------------------------------------

def test_sim_robot_damping_holds(robot):
    """Damping command sets position actuator to pure damping (kp=0, kv=kd).

    With position actuators, a damping command sets:
      ctrl = 0 (target position), gainprm = 0 (kp), biasprm[2] = -kd
    The actuator then computes: force = 0*(ctrl-q) - kd*dq = -kd*dq
    """
    robot.reset()
    kd_val = 10.0
    cmd = RobotCommand.damping(29, kd=kd_val)
    robot.send_command(cmd)
    robot.step()  # applies command

    for cfg_i in range(29):
        mj_i = robot._cfg_to_mj[cfg_i]
        # ctrl = target position = 0 (from damping command)
        np.testing.assert_allclose(
            robot.mj_data.ctrl[mj_i], 0.0, atol=1e-12,
            err_msg=f"Damping ctrl (target pos) mismatch at joint {cfg_i}"
        )
        # kp = 0 (no position tracking)
        np.testing.assert_allclose(
            robot.mj_model.actuator_gainprm[mj_i, 0], 0.0, atol=1e-12,
            err_msg=f"Damping kp mismatch at joint {cfg_i}"
        )
        # kv = kd (velocity damping)
        np.testing.assert_allclose(
            robot.mj_model.actuator_biasprm[mj_i, 2], -kd_val, atol=1e-12,
            err_msg=f"Damping kv mismatch at joint {cfg_i}"
        )


# ---------------------------------------------------------------------------
# Test: send_command shape
# ---------------------------------------------------------------------------

def test_sim_robot_send_command_shape(robot):
    """send_command accepts n_dof-shaped command arrays."""
    cmd = RobotCommand(
        joint_positions=np.zeros(29),
        joint_velocities=np.zeros(29),
        joint_torques=np.zeros(29),
        kp=np.full(29, 100.0),
        kd=np.full(29, 10.0),
    )
    # Should not raise
    robot.send_command(cmd)


# ---------------------------------------------------------------------------
# Test: IMU upright
# ---------------------------------------------------------------------------

def test_sim_robot_imu_upright(robot):
    """After reset, IMU quaternion is approximately identity (upright)."""
    robot.reset()
    state = robot.get_state()
    # wxyz format, identity = [1, 0, 0, 0]
    np.testing.assert_allclose(state.imu_quaternion, [1, 0, 0, 0], atol=0.01)


# ---------------------------------------------------------------------------
# Test: connect/disconnect
# ---------------------------------------------------------------------------

def test_sim_robot_connect_disconnect(robot):
    """connect() and disconnect() run without errors (mocked DDS)."""
    with patch("unitree_launcher.robot.sim_robot.patch_unitree_threading"):
        with patch.dict("sys.modules", {
            "unitree_sdk2py.core.channel": MagicMock(),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": MagicMock(),
            "unitree_sdk2py.idl.default": MagicMock(),
        }):
            robot.connect()
            assert robot._dds_initialized is True
            robot.disconnect()
            assert robot._dds_initialized is False


# ---------------------------------------------------------------------------
# Test: Metal-specific viewer properties
# ---------------------------------------------------------------------------

def test_sim_robot_exposes_mj_model(robot):
    """mj_model property exposes MuJoCo model."""
    assert isinstance(robot.mj_model, mujoco.MjModel)
    assert robot.mj_model.nu == 29


def test_sim_robot_exposes_lock(robot):
    """lock property exposes threading.Lock."""
    assert isinstance(robot.lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# Test: impedance control values (safety-critical, value-level)
# ---------------------------------------------------------------------------

def test_sim_robot_impedance_control_values(robot):
    """Verify position actuator setup matches the PD command.

    With position actuators, send_command sets:
      ctrl[i] = q_des (target position)
      gainprm[i,0] = kp, biasprm[i,1] = -kp, biasprm[i,2] = -kd
      qfrc_applied[dof_i] = tau_ff (feedforward torque)

    MuJoCo then computes: force = kp*(q_des - q) - kd*dq + tau_ff
    """
    robot.reset()
    mujoco.mj_forward(robot.mj_model, robot.mj_data)

    # Known command values
    kp = np.full(29, 50.0)
    kd = np.full(29, 5.0)
    q_des = np.linspace(0.0, 0.5, 29)
    tau_ff = np.linspace(-1.0, 1.0, 29)

    cmd = RobotCommand(
        joint_positions=q_des.copy(),
        joint_velocities=np.zeros(29),
        joint_torques=tau_ff.copy(),
        kp=kp.copy(),
        kd=kd.copy(),
    )
    robot.send_command(cmd)
    robot.step()  # applies command

    for cfg_i in range(29):
        mj_i = robot._cfg_to_mj[cfg_i]
        dof_i = robot._dof_addr[cfg_i]

        # ctrl = target position
        np.testing.assert_allclose(
            robot.mj_data.ctrl[mj_i], q_des[cfg_i], rtol=1e-10,
            err_msg=f"ctrl mismatch at joint {cfg_i}"
        )
        # gainprm = kp
        np.testing.assert_allclose(
            robot.mj_model.actuator_gainprm[mj_i, 0], kp[cfg_i], rtol=1e-10,
            err_msg=f"kp mismatch at joint {cfg_i}"
        )
        # biasprm[1] = -kp, biasprm[2] = -kd
        np.testing.assert_allclose(
            robot.mj_model.actuator_biasprm[mj_i, 1], -kp[cfg_i], rtol=1e-10,
            err_msg=f"biasprm[1] mismatch at joint {cfg_i}"
        )
        np.testing.assert_allclose(
            robot.mj_model.actuator_biasprm[mj_i, 2], -kd[cfg_i], rtol=1e-10,
            err_msg=f"biasprm[2] mismatch at joint {cfg_i}"
        )
        # feedforward in qfrc_applied
        np.testing.assert_allclose(
            robot.mj_data.qfrc_applied[dof_i], tau_ff[cfg_i], rtol=1e-10,
            err_msg=f"qfrc_applied mismatch at joint {cfg_i}"
        )


# ---------------------------------------------------------------------------
# Test: sensor mapping correctness
# ---------------------------------------------------------------------------

def test_sim_robot_sensor_mapping_correctness(robot):
    """get_state() matches raw sensordata layout for 29-DOF."""
    robot.reset()
    mujoco.mj_forward(robot.mj_model, robot.mj_data)

    state = robot.get_state()
    sd = robot.mj_data.sensordata
    nm = 29
    dms = 3 * nm

    # Joint positions = sensordata[0:29] (identity mapping for 29-DOF)
    np.testing.assert_array_equal(state.joint_positions, sd[0:29])
    # Joint velocities = sensordata[29:58]
    np.testing.assert_array_equal(state.joint_velocities, sd[29:58])
    # Joint torques = sensordata[58:87]
    np.testing.assert_array_equal(state.joint_torques, sd[58:87])
    # IMU quaternion = sensordata[87:91]
    np.testing.assert_array_equal(state.imu_quaternion, sd[87:91])
    # IMU gyro = sensordata[91:94]
    np.testing.assert_array_equal(state.imu_angular_velocity, sd[91:94])
    # IMU accel = sensordata[94:97]
    np.testing.assert_array_equal(state.imu_linear_acceleration, sd[94:97])
    # Base position = sensordata[97:100]
    np.testing.assert_array_equal(state.base_position, sd[97:100])
    # Base velocity = sensordata[100:103]
    np.testing.assert_array_equal(state.base_velocity, sd[100:103])


# ---------------------------------------------------------------------------
# Test: 23-DOF variant
# ---------------------------------------------------------------------------

def test_sim_robot_23dof(robot_23dof):
    """23-DOF config: n_dof=23, shapes (23,), step works."""
    assert robot_23dof.n_dof == 23

    state = robot_23dof.get_state()
    assert state.joint_positions.shape == (23,)
    assert state.joint_velocities.shape == (23,)
    assert state.joint_torques.shape == (23,)

    # Step should work
    robot_23dof.step()
    assert robot_23dof.mj_data.time > 0.0


# ---------------------------------------------------------------------------
# Test: substep count
# ---------------------------------------------------------------------------

def test_sim_robot_substep_count(robot):
    """With sim_freq=200, policy_freq=50, step advances by 4*0.005=0.02s."""
    robot.reset()
    t0 = robot.mj_data.time
    robot.step()
    t1 = robot.mj_data.time
    expected_dt = 4 * 0.005  # 4 substeps * 0.005s
    np.testing.assert_allclose(t1 - t0, expected_dt, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: base position
# ---------------------------------------------------------------------------

def test_sim_robot_base_position(robot):
    """After reset, base_position Z is initial height; falls under gravity."""
    robot.reset()
    state0 = robot.get_state()
    # Initial height should be positive (robot standing)
    assert state0.base_position[2] > 0.5

    # Step with gravity, no commands
    for _ in range(20):
        robot.step()

    state1 = robot.get_state()
    assert state1.base_position[2] < state0.base_position[2]


# ---------------------------------------------------------------------------
# Test: reset with custom state
# ---------------------------------------------------------------------------

def test_sim_robot_reset_custom_state(robot):
    """reset(initial_state) sets joint positions to custom values."""
    custom_pos = np.linspace(-0.1, 0.1, 29)
    custom_state = RobotState(
        timestamp=0.0,
        joint_positions=custom_pos.copy(),
        joint_velocities=np.zeros(29),
        joint_torques=np.zeros(29),
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_angular_velocity=np.zeros(3),
        imu_linear_acceleration=np.zeros(3),
        base_position=np.zeros(3),
        base_velocity=np.zeros(3),
    )
    robot.reset(initial_state=custom_state)

    state = robot.get_state()
    np.testing.assert_allclose(state.joint_positions, custom_pos, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: DDS publish mock
# ---------------------------------------------------------------------------

def test_sim_robot_dds_publish_mock(robot):
    """DDS publish thread calls Write() with motor states populated."""
    # Create a mock LowState message
    mock_msg = MagicMock()
    mock_msg.motor_state = [MagicMock() for _ in range(35)]
    mock_msg.imu_state = MagicMock()
    mock_msg.imu_state.quaternion = [0.0] * 4
    mock_msg.imu_state.gyroscope = [0.0] * 3
    mock_msg.imu_state.accelerometer = [0.0] * 3

    mock_pub = MagicMock()

    robot._low_state_msg = mock_msg
    robot._low_state_pub = mock_pub

    # Call the publish method directly
    robot._publish_low_state()

    # Verify Write was called
    mock_pub.Write.assert_called_once_with(mock_msg)

    # Verify motor states were populated
    for i in range(29):
        assert mock_msg.motor_state[i].q == float(robot.mj_data.sensordata[i])


# ---------------------------------------------------------------------------
# Test: get_state returns copies (not references to internal data)
# ---------------------------------------------------------------------------

def test_sim_robot_get_state_returns_copies(robot):
    """get_state() arrays are independent copies of sensor data."""
    state = robot.get_state()
    original_pos = state.joint_positions.copy()

    # Mutate the returned array
    state.joint_positions[:] = 999.0

    # Get state again — should not be affected
    state2 = robot.get_state()
    np.testing.assert_array_equal(state2.joint_positions, original_pos)


# ---------------------------------------------------------------------------
# Test: send_command wrong shape
# ---------------------------------------------------------------------------

def test_send_command_wrong_shape(robot):
    """send_command with wrong-shaped arrays should raise on step, not silently corrupt."""
    cmd = RobotCommand(
        joint_positions=np.zeros(10),  # wrong: 10 != 29
        joint_velocities=np.zeros(10),
        joint_torques=np.zeros(10),
        kp=np.full(10, 100.0),
        kd=np.full(10, 10.0),
    )
    robot.send_command(cmd)
    with pytest.raises((IndexError, ValueError)):
        robot.step()
