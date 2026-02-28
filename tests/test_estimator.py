"""Tests for the estimation module: Lie groups, InEKF, FK, contacts, and facade."""
from __future__ import annotations

import numpy as np
import pytest

from unitree_launcher.config import G1_29DOF_JOINTS, Q_HOME_29DOF
from unitree_launcher.estimation.lie_group import (
    adjoint_se2_3,
    quat_to_rotation_matrix,
    se2_3_exp,
    skew,
    so3_exp,
    so3_left_jacobian,
    so3_log,
    unskew,
)
from unitree_launcher.estimation.inekf import RightInvariantEKF
from unitree_launcher.estimation.contact import ContactDetector, SchmittTrigger
from unitree_launcher.estimation.kinematics import G1Kinematics
from unitree_launcher.estimation.state_estimator import StateEstimator
from unitree_launcher.robot.base import RobotState


# ======================================================================
# Lie Group Tests
# ======================================================================


class TestSkew:
    def test_skew_unskew_roundtrip(self):
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(unskew(skew(v)), v)

    def test_skew_antisymmetric(self):
        v = np.array([0.5, -1.2, 3.7])
        S = skew(v)
        assert np.allclose(S, -S.T)

    def test_skew_cross_product(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert np.allclose(skew(a) @ b, np.cross(a, b))


class TestSO3:
    def test_exp_identity(self):
        R = so3_exp(np.zeros(3))
        assert np.allclose(R, np.eye(3))

    def test_exp_small_angle(self):
        phi = np.array([1e-9, 0.0, 0.0])
        R = so3_exp(phi)
        assert np.allclose(R, np.eye(3), atol=1e-6)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_exp_90_deg_z(self):
        phi = np.array([0.0, 0.0, np.pi / 2])
        R = so3_exp(phi)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert np.allclose(R, expected, atol=1e-10)

    def test_exp_log_roundtrip(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            phi = rng.uniform(-np.pi + 0.1, np.pi - 0.1, 3)
            R = so3_exp(phi)
            phi_back = so3_log(R)
            R_back = so3_exp(phi_back)
            assert np.allclose(R, R_back, atol=1e-10)

    def test_log_identity(self):
        phi = so3_log(np.eye(3))
        assert np.allclose(phi, np.zeros(3), atol=1e-12)

    def test_exp_is_rotation(self):
        rng = np.random.default_rng(123)
        phi = rng.standard_normal(3)
        R = so3_exp(phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestSO3LeftJacobian:
    def test_identity_at_zero(self):
        J = so3_left_jacobian(np.zeros(3))
        assert np.allclose(J, np.eye(3), atol=1e-6)

    def test_numerical_gradient(self):
        """Verify J * delta_phi ≈ Log(Exp(phi + delta_phi) * Exp(-phi))."""
        phi = np.array([0.3, -0.5, 0.1])
        J = so3_left_jacobian(phi)
        eps = 1e-6
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            R1 = so3_exp(phi + delta)
            R0 = so3_exp(phi)
            diff = so3_log(R1 @ R0.T)
            J_col = diff / eps
            assert np.allclose(J[:, i], J_col, atol=1e-4)


class TestSE23:
    def test_exp_identity(self):
        X = se2_3_exp(np.zeros(9))
        assert np.allclose(X, np.eye(5))

    def test_exp_translation_only(self):
        xi = np.array([0, 0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        X = se2_3_exp(xi)
        assert np.allclose(X[0:3, 0:3], np.eye(3))
        assert np.allclose(X[0:3, 3], [1, 2, 3])
        assert np.allclose(X[0:3, 4], [4, 5, 6])

    def test_adjoint_shape(self):
        X = se2_3_exp(np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0]))
        Ad = adjoint_se2_3(X)
        assert Ad.shape == (9, 9)

    def test_adjoint_identity(self):
        Ad = adjoint_se2_3(np.eye(5))
        assert np.allclose(Ad, np.eye(9))


class TestQuatToRotation:
    def test_identity_quaternion(self):
        R = quat_to_rotation_matrix(np.array([1.0, 0, 0, 0]))
        assert np.allclose(R, np.eye(3))

    def test_90_deg_z(self):
        """wxyz quaternion for 90-deg rotation about Z."""
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        R = quat_to_rotation_matrix(q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert np.allclose(R, expected, atol=1e-10)

    def test_orthogonal(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        R = quat_to_rotation_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


# ======================================================================
# Contact Detector Tests
# ======================================================================


class TestSchmittTrigger:
    def test_initially_off(self):
        t = SchmittTrigger(high_threshold=50, low_threshold=20)
        assert t.state is False

    def test_goes_high_after_debounce(self):
        t = SchmittTrigger(high_threshold=50, low_threshold=20,
                           high_time=0.02, low_time=0.04)
        dt = 0.01  # 10ms
        # Below threshold -> stays off
        assert t.update(30.0, dt) is False
        # Above threshold but not long enough
        assert t.update(60.0, dt) is False
        # Now accumulated 20ms >= high_time
        assert t.update(60.0, dt) is True

    def test_goes_low_after_debounce(self):
        t = SchmittTrigger(high_threshold=50, low_threshold=20,
                           high_time=0.0, low_time=0.04)
        # Force on immediately
        t.state = True
        dt = 0.01
        # Below low threshold
        assert t.update(10.0, dt) is True   # not enough time
        assert t.update(10.0, dt) is True   # 20ms
        assert t.update(10.0, dt) is True   # 30ms
        assert t.update(10.0, dt) is False  # 40ms >= low_time

    def test_resets_timer_on_bounce(self):
        t = SchmittTrigger(high_threshold=50, low_threshold=20,
                           high_time=0.03, low_time=0.04)
        dt = 0.01
        t.update(60.0, dt)  # 10ms above
        t.update(60.0, dt)  # 20ms above
        t.update(30.0, dt)  # drops below -> timer resets
        t.update(60.0, dt)  # restart: 10ms
        assert t.state is False
        t.update(60.0, dt)  # 20ms
        t.update(60.0, dt)  # 30ms >= high_time
        assert t.state is True


class TestContactDetector:
    def test_both_standing(self):
        det = ContactDetector(
            lever_arm=0.025, high_threshold=40, low_threshold=15,
            high_time=0.0, low_time=0.04
        )
        torques = np.zeros(29)
        # Standing: ankle torque ≈ body weight * lever arm / 2
        # ~40kg * 9.81 / 2 * 0.025 ≈ 4.9 Nm per ankle
        # -> GRF = 4.9 / 0.025 = 196 N (well above threshold)
        torques[4] = 5.0   # left ankle pitch
        torques[10] = 5.0  # right ankle pitch
        left, right = det.update(torques, 0.02)
        assert left is True
        assert right is True

    def test_both_airborne(self):
        det = ContactDetector(
            lever_arm=0.025, high_threshold=40, low_threshold=15,
            high_time=0.0, low_time=0.0
        )
        torques = np.zeros(29)
        left, right = det.update(torques, 0.02)
        assert left is False
        assert right is False

    def test_reset_both_in_contact(self):
        det = ContactDetector()
        det.reset(both_in_contact=True)
        assert det.left_contact is True
        assert det.right_contact is True

    def test_reset_not_in_contact(self):
        det = ContactDetector()
        det.reset(both_in_contact=False)
        assert det.left_contact is False
        assert det.right_contact is False


# ======================================================================
# Kinematics Tests
# ======================================================================


class TestG1Kinematics:
    @pytest.fixture
    def fk(self):
        return G1Kinematics()

    @pytest.fixture
    def home_q(self):
        return np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])

    def test_left_foot_home_pose(self, fk, home_q):
        """At home pose, left foot should be roughly below pelvis, ~0.65m down."""
        p = fk.left_foot_position(home_q)
        assert p.shape == (3,)
        # Foot should be below pelvis (negative Z)
        assert p[2] < -0.4
        # Should be roughly at standing height: pelvis at 0.793 + foot at ~-0.65
        # so foot Z ≈ -0.65 in pelvis frame (world Z ≈ 0.14)
        assert -0.75 < p[2] < -0.5
        # Laterally offset to the left (positive Y in pelvis frame)
        assert p[1] > 0

    def test_right_foot_home_pose(self, fk, home_q):
        """At home pose, right foot should mirror left (negative Y)."""
        p_left = fk.left_foot_position(home_q)
        p_right = fk.right_foot_position(home_q)
        # Mirror in Y
        assert np.isclose(p_left[1], -p_right[1], atol=0.01)
        # Same Z depth
        assert np.isclose(p_left[2], p_right[2], atol=0.01)

    def test_zero_config(self, fk):
        """At zero joint angles, feet should be directly below hips."""
        q = np.zeros(29)
        p_left = fk.left_foot_position(q)
        p_right = fk.right_foot_position(q)
        # Both should be below pelvis
        assert p_left[2] < -0.5
        assert p_right[2] < -0.5

    def test_left_jacobian_shape(self, fk, home_q):
        J = fk.left_foot_jacobian(home_q)
        assert J.shape == (3, 6)

    def test_right_jacobian_shape(self, fk, home_q):
        J = fk.right_foot_jacobian(home_q)
        assert J.shape == (3, 6)

    def test_jacobian_numerical_consistency(self, fk, home_q):
        """Jacobian times small joint velocity should match FK displacement."""
        J = fk.left_foot_jacobian(home_q)
        dq = np.array([0.01, 0, 0, 0, 0, 0])
        p0 = fk.left_foot_position(home_q)
        q_pert = home_q.copy()
        q_pert[0] += 0.01
        p1 = fk.left_foot_position(q_pert)
        dp_actual = p1 - p0
        dp_jacobian = J @ dq
        assert np.allclose(dp_actual, dp_jacobian, atol=1e-4)

    def test_subset_input(self, fk):
        """FK should work with 6-element leg subset."""
        q_leg = np.array([-0.312, 0.0, 0.0, 0.669, -0.33, 0.0])
        p = fk.left_foot_position(q_leg)
        assert p.shape == (3,)


# ======================================================================
# InEKF Tests
# ======================================================================


class TestRightInvariantEKF:
    @pytest.fixture
    def ekf(self):
        ekf = RightInvariantEKF()
        ekf.initialize(
            R0=np.eye(3),
            v0=np.zeros(3),
            p0=np.array([0.0, 0.0, 0.793]),
        )
        return ekf

    def test_initialize(self, ekf):
        assert ekf.initialized
        assert np.allclose(ekf.rotation, np.eye(3))
        assert np.allclose(ekf.velocity, np.zeros(3))
        assert np.allclose(ekf.position, [0, 0, 0.793])
        assert ekf.n_contacts == 0

    def test_predict_gravity_only(self, ekf):
        """With only gravity measured, velocity should increase downward."""
        # Accelerometer measures -gravity in body frame when stationary
        # If upright: accel = [0, 0, 9.81] (specific force, not including gravity)
        omega = np.zeros(3)
        accel = np.array([0.0, 0.0, 9.81])  # body-frame measurement
        dt = 0.02

        # After predict: a_world = R @ accel + g = [0,0,9.81] + [0,0,-9.81] = 0
        ekf.predict(omega, accel, dt)

        # Velocity should be approximately zero (gravity cancelled)
        assert np.allclose(ekf.velocity, np.zeros(3), atol=0.01)
        # Position should be approximately unchanged
        assert np.allclose(ekf.position, [0, 0, 0.793], atol=0.01)

    def test_predict_free_fall(self, ekf):
        """In free fall, accelerometer reads zero; velocity increases down."""
        omega = np.zeros(3)
        accel = np.zeros(3)  # free fall: no specific force
        dt = 0.1

        ekf.predict(omega, accel, dt)

        # v = g * dt = [0, 0, -9.81 * 0.1] = [0, 0, -0.981]
        assert np.isclose(ekf.velocity[2], -9.81 * dt, atol=0.01)

    def test_augment_marginalize_contact(self, ekf):
        assert ekf.n_contacts == 0
        cid = ekf.augment_state(np.array([0.0, 0.1, -0.65]))
        assert ekf.n_contacts == 1

        # Augment a second contact
        cid2 = ekf.augment_state(np.array([0.0, -0.1, -0.65]))
        assert ekf.n_contacts == 2

        # Marginalize first
        ekf.marginalize_contact(cid)
        assert ekf.n_contacts == 1

        # Marginalize second
        ekf.marginalize_contact(cid2)
        assert ekf.n_contacts == 0

    def test_marginalize_nonexistent(self, ekf):
        """Marginalize with invalid ID should be a no-op."""
        ekf.marginalize_contact(999)
        assert ekf.n_contacts == 0

    def test_correct_kinematics_basic(self, ekf):
        """Correction with a contact should not diverge."""
        p_foot = np.array([0.0, 0.1, -0.65])
        cid = ekf.augment_state(p_foot)

        # Position before correction
        p0 = ekf.position.copy()

        # Small predict to create some error
        ekf.predict(np.zeros(3), np.array([0.0, 0.0, 9.81]), 0.02)

        # Correct
        ekf.correct_kinematics(cid, p_foot)

        # Should still be close to initial
        assert np.linalg.norm(ekf.position - p0) < 0.1

    def test_covariance_stays_symmetric(self, ekf):
        p_foot = np.array([0.0, 0.1, -0.65])
        cid = ekf.augment_state(p_foot)

        for _ in range(10):
            ekf.predict(np.zeros(3), np.array([0.0, 0.0, 9.81]), 0.02)
            ekf.correct_kinematics(cid, p_foot)

        P = ekf.covariance
        assert np.allclose(P, P.T, atol=1e-10)

    def test_rotation_stays_valid(self, ekf):
        """After many predict/correct cycles, R should still be orthogonal."""
        p_foot = np.array([0.0, 0.1, -0.65])
        cid = ekf.augment_state(p_foot)

        rng = np.random.default_rng(99)
        for _ in range(50):
            omega = rng.standard_normal(3) * 0.1
            accel = np.array([0.0, 0.0, 9.81]) + rng.standard_normal(3) * 0.01
            ekf.predict(omega, accel, 0.02)
            ekf.correct_kinematics(cid, p_foot)

        R = ekf.rotation
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


# ======================================================================
# StateEstimator Facade Tests
# ======================================================================


class TestStateEstimator:
    @pytest.fixture
    def home_state(self):
        """Standing state at home pose."""
        home = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        return RobotState(
            timestamp=0.0,
            joint_positions=home.copy(),
            joint_velocities=np.zeros(29),
            joint_torques=np.zeros(29),
            imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            imu_angular_velocity=np.zeros(3),
            imu_linear_acceleration=np.array([0.0, 0.0, 9.81]),
            base_position=np.array([0.0, 0.0, 0.793]),
            base_velocity=np.zeros(3),
        )

    @pytest.fixture
    def mock_config(self):
        class MockControl:
            policy_frequency = 50
        class MockConfig:
            control = MockControl()
        return MockConfig()

    def test_initialize_from_sim_state(self, mock_config, home_state):
        est = StateEstimator(mock_config)
        est.initialize(home_state)
        assert est.initialized
        assert np.allclose(est.base_position, [0, 0, 0.793], atol=0.01)

    def test_initialize_from_nan_state(self, mock_config, home_state):
        """When base_position/velocity are NaN (real robot), use defaults."""
        home_state.base_position = np.full(3, np.nan)
        home_state.base_velocity = np.full(3, np.nan)
        est = StateEstimator(mock_config)
        est.initialize(home_state)
        assert est.initialized
        assert np.allclose(est.base_position, [0, 0, 0.793], atol=0.01)

    def test_update_standing(self, mock_config, home_state):
        """Standing still: estimator should remain near initial pose."""
        est = StateEstimator(mock_config)

        # Give ankle torques to detect contacts
        home_state.joint_torques[4] = 5.0   # left ankle pitch
        home_state.joint_torques[10] = 5.0  # right ankle pitch

        for _ in range(50):
            est.update(home_state, dt=0.02)

        assert np.linalg.norm(est.base_position - [0, 0, 0.793]) < 0.1
        assert np.linalg.norm(est.base_velocity) < 0.5

    def test_populate_close_to_gt_in_sim(self, mock_config, home_state):
        """In sim, estimator output should be very close to ground truth."""
        est = StateEstimator(mock_config)
        home_state.joint_torques[4] = 5.0   # ankle torques for contact
        home_state.joint_torques[10] = 5.0
        # Run past warmup + blend
        for _ in range(60):
            est.update(home_state, dt=0.02)
        new_state = est.populate_robot_state(home_state)
        assert np.allclose(new_state.base_position, home_state.base_position, atol=0.05)
        assert np.allclose(new_state.base_velocity, home_state.base_velocity, atol=0.05)
        assert new_state is not home_state

    def test_contact_properties(self, mock_config, home_state):
        est = StateEstimator(mock_config)
        est.initialize(home_state)
        # After init, both feet should be in contact
        assert est.left_contact is True
        assert est.right_contact is True

    def test_auto_initialize_on_first_update(self, mock_config, home_state):
        """StateEstimator auto-initializes on first update call."""
        est = StateEstimator(mock_config)
        assert not est.initialized
        est.update(home_state)
        assert est.initialized

    def test_warmup_skips_ekf(self, mock_config, home_state):
        """During warmup, EKF is not run; output uses raw state."""
        est = StateEstimator(mock_config)
        home_state.joint_torques[:] = 0.0

        for i in range(24):  # warmup = 25 ticks, tick 0 is initialize
            est.update(home_state, dt=0.02)
            out = est.populate_robot_state(home_state)
            # During warmup, output should be raw state
            assert np.allclose(out.base_position, [0, 0, 0.793]), \
                f"step {i}: should use raw state during warmup"
            assert np.allclose(out.base_velocity, np.zeros(3)), \
                f"step {i}: should use raw velocity during warmup"

    def test_populate_nan_fallback_on_divergence(self, mock_config, home_state):
        """On real robot, falls back to defaults when EKF diverges."""
        est = StateEstimator(mock_config)
        home_state.base_position = np.full(3, np.nan)
        home_state.base_velocity = np.full(3, np.nan)
        for _ in range(30):
            est.update(home_state, dt=0.02)
        est._ekf._X[0:3, 3] = np.array([0, 0, 50.0])  # diverged velocity
        out = est.populate_robot_state(home_state)
        # Falls back to defaults (p0, zero vel)
        assert not np.any(np.isnan(out.base_position))
        assert not np.any(np.isnan(out.base_velocity))

    def test_blend_alpha_ramp(self, mock_config, home_state):
        """Blend alpha ramps from 0 to 1 after warmup."""
        est = StateEstimator(mock_config)

        # Run through warmup + re-init tick + one normal tick
        for _ in range(27):
            est.update(home_state, dt=0.02)

        alpha = est._blend_alpha()
        assert 0.0 < alpha < 0.5, f"Expected small alpha, got {alpha}"

        for _ in range(25):
            est.update(home_state, dt=0.02)
        alpha = est._blend_alpha()
        assert alpha == 1.0, f"Expected alpha=1.0, got {alpha}"

    def test_real_robot_nan_fill(self, mock_config, home_state):
        """On real robot (NaN base), estimator fills values after warmup."""
        est = StateEstimator(mock_config)

        for i in range(40):
            state = RobotState(
                timestamp=i * 0.02,
                joint_positions=home_state.joint_positions.copy(),
                joint_velocities=np.zeros(29), joint_torques=np.zeros(29),
                imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                imu_angular_velocity=np.zeros(3),
                imu_linear_acceleration=np.array([0.0, 0.0, 9.81]),
                base_position=np.full(3, np.nan),  # NaN = real robot
                base_velocity=np.full(3, np.nan),
            )
            est.update(state, dt=0.02)
            out = est.populate_robot_state(state)
            assert np.all(np.isfinite(out.base_position)), f"NaN pos at step {i}"
            assert np.all(np.isfinite(out.base_velocity)), f"NaN vel at step {i}"


# ======================================================================
# FK validation against MuJoCo (optional -- requires mujoco)
# ======================================================================


class TestFKvsMuJoCo:
    """Validate FK output against MuJoCo's site_xpos."""

    @pytest.fixture
    def mj_env(self):
        """Load the 29-DOF model and return (model, data)."""
        try:
            import mujoco
        except ImportError:
            pytest.skip("mujoco not installed")
        model_path = str(
            __import__("pathlib").Path(__file__).resolve().parent.parent
            / "assets" / "robots" / "g1" / "g1_29dof.xml"
        )
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        return model, data

    def _get_site_pos(self, model, data, site_name):
        import mujoco
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return data.site_xpos[site_id].copy()

    def _set_joint_positions(self, model, data, q_dict):
        """Set joint positions from a dict and run mj_forward."""
        import mujoco
        for name, value in q_dict.items():
            joint_name = name + "_joint"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            qpos_addr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_addr] = value
        mujoco.mj_forward(model, data)

    def test_home_pose_left_foot(self, mj_env):
        """FK left foot at home pose should match MuJoCo within 1mm."""
        model, data = mj_env
        import mujoco

        self._set_joint_positions(model, data, Q_HOME_29DOF)
        mujoco.mj_forward(model, data)

        # MuJoCo site position is in world frame
        site_world = self._get_site_pos(model, data, "left_foot")

        # Pelvis position (floating base)
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_pos = data.xpos[pelvis_id].copy()
        pelvis_mat = data.xmat[pelvis_id].reshape(3, 3).copy()

        # Site in pelvis frame
        site_pelvis_mj = pelvis_mat.T @ (site_world - pelvis_pos)

        # Our FK
        fk = G1Kinematics()
        q = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        site_pelvis_fk = fk.left_foot_position(q)

        error = np.linalg.norm(site_pelvis_fk - site_pelvis_mj)
        assert error < 0.001, f"FK error {error:.6f}m > 1mm: fk={site_pelvis_fk}, mj={site_pelvis_mj}"

    def test_home_pose_right_foot(self, mj_env):
        """FK right foot at home pose should match MuJoCo within 1mm."""
        model, data = mj_env
        import mujoco

        self._set_joint_positions(model, data, Q_HOME_29DOF)
        mujoco.mj_forward(model, data)

        site_world = self._get_site_pos(model, data, "right_foot")
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_pos = data.xpos[pelvis_id].copy()
        pelvis_mat = data.xmat[pelvis_id].reshape(3, 3).copy()
        site_pelvis_mj = pelvis_mat.T @ (site_world - pelvis_pos)

        fk = G1Kinematics()
        q = np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS])
        site_pelvis_fk = fk.right_foot_position(q)

        error = np.linalg.norm(site_pelvis_fk - site_pelvis_mj)
        assert error < 0.001, f"FK error {error:.6f}m > 1mm: fk={site_pelvis_fk}, mj={site_pelvis_mj}"

    def test_random_configs(self, mj_env):
        """FK at random joint configs should match MuJoCo within 1mm."""
        model, data = mj_env
        import mujoco

        fk = G1Kinematics()
        rng = np.random.default_rng(42)

        for _ in range(10):
            # Random joint positions within limits (small range for legs)
            q_dict = {}
            for j in G1_29DOF_JOINTS:
                q_dict[j] = rng.uniform(-0.3, 0.3)
            # Keep legs in reasonable range
            for prefix in ["left_", "right_"]:
                q_dict[f"{prefix}knee"] = rng.uniform(0.0, 1.5)
                q_dict[f"{prefix}hip_pitch"] = rng.uniform(-0.5, 0.3)
                q_dict[f"{prefix}ankle_pitch"] = rng.uniform(-0.5, 0.3)

            self._set_joint_positions(model, data, q_dict)
            mujoco.mj_forward(model, data)

            q_arr = np.array([q_dict[j] for j in G1_29DOF_JOINTS])

            for side, site_name in [("left", "left_foot"), ("right", "right_foot")]:
                site_world = self._get_site_pos(model, data, site_name)
                pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                pelvis_pos = data.xpos[pelvis_id].copy()
                pelvis_mat = data.xmat[pelvis_id].reshape(3, 3).copy()
                site_pelvis_mj = pelvis_mat.T @ (site_world - pelvis_pos)

                if side == "left":
                    site_pelvis_fk = fk.left_foot_position(q_arr)
                else:
                    site_pelvis_fk = fk.right_foot_position(q_arr)

                error = np.linalg.norm(site_pelvis_fk - site_pelvis_mj)
                assert error < 0.001, (
                    f"{side} FK error {error:.6f}m > 1mm at random config"
                )
