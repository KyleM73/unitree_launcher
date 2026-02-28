"""Tests for the mink-based trajectory planner."""
from __future__ import annotations

import importlib
from unittest import mock

import numpy as np
import pytest

from unitree_launcher.config import G1_29DOF_JOINTS, Q_HOME_29DOF


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def home_q():
    """Home positions in config order."""
    return np.array([Q_HOME_29DOF[j] for j in G1_29DOF_JOINTS], dtype=np.float64)


@pytest.fixture
def zero_q():
    """All-zeros configuration."""
    return np.zeros(29, dtype=np.float64)


# ---------------------------------------------------------------------------
# Module-level tests (no mink required)
# ---------------------------------------------------------------------------

class TestCollisionPairs:
    """Verify the collision pair constant is well-formed."""

    def test_pairs_are_list_of_tuples(self):
        from unitree_launcher.trajectory import G1_SELF_COLLISION_PAIRS

        assert isinstance(G1_SELF_COLLISION_PAIRS, list)
        for pair in G1_SELF_COLLISION_PAIRS:
            assert len(pair) == 2
            geoms_a, geoms_b = pair
            assert isinstance(geoms_a, list)
            assert isinstance(geoms_b, list)
            assert len(geoms_a) > 0
            assert len(geoms_b) > 0

    def test_all_geom_names_are_strings(self):
        from unitree_launcher.trajectory import G1_SELF_COLLISION_PAIRS

        for geoms_a, geoms_b in G1_SELF_COLLISION_PAIRS:
            for name in geoms_a + geoms_b:
                assert isinstance(name, str)

    def test_no_duplicate_pairs(self):
        from unitree_launcher.trajectory import G1_SELF_COLLISION_PAIRS

        seen = set()
        for geoms_a, geoms_b in G1_SELF_COLLISION_PAIRS:
            key = (tuple(sorted(geoms_a)), tuple(sorted(geoms_b)))
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)

    def test_expected_categories_covered(self):
        """Ensure all planned collision categories have at least one pair."""
        from unitree_launcher.trajectory import G1_SELF_COLLISION_PAIRS

        all_geom_a = set()
        all_geom_b = set()
        for geoms_a, geoms_b in G1_SELF_COLLISION_PAIRS:
            all_geom_a.update(geoms_a)
            all_geom_b.update(geoms_b)
        all_geoms = all_geom_a | all_geom_b

        # Arms present
        assert "left_elbow_collision" in all_geoms
        assert "right_hand_collision" in all_geoms
        # Torso present
        assert "torso_collision1" in all_geoms
        # Head present
        assert "head_collision" in all_geoms
        # Legs present
        assert "left_thigh" in all_geoms
        assert "right_shin" in all_geoms
        # Pelvis present
        assert "pelvis_collision" in all_geoms


class TestTrajectoryPlan:
    """Tests for the TrajectoryPlan dataclass."""

    def test_dataclass_fields(self):
        from unitree_launcher.trajectory import TrajectoryPlan

        wp = np.zeros((10, 29))
        plan = TrajectoryPlan(waypoints=wp, dt_plan=0.02, converged=True)
        assert plan.waypoints.shape == (10, 29)
        assert plan.dt_plan == 0.02
        assert plan.converged is True

    def test_not_converged(self):
        from unitree_launcher.trajectory import TrajectoryPlan

        plan = TrajectoryPlan(
            waypoints=np.zeros((5, 29)), dt_plan=0.01, converged=False,
        )
        assert plan.converged is False


class TestResampleTrajectory:
    """Tests for resample_trajectory (no mink needed)."""

    def test_output_shape(self):
        from unitree_launcher.trajectory import TrajectoryPlan, resample_trajectory

        wp = np.random.randn(50, 29)
        plan = TrajectoryPlan(waypoints=wp, dt_plan=0.02, converged=True)
        result = resample_trajectory(plan, target_duration=5.0, target_dt=0.02)
        assert result.shape == (250, 29)

    def test_single_waypoint_replicates(self):
        from unitree_launcher.trajectory import TrajectoryPlan, resample_trajectory

        wp = np.ones((1, 29)) * 0.5
        plan = TrajectoryPlan(waypoints=wp, dt_plan=0.02, converged=True)
        result = resample_trajectory(plan, target_duration=1.0, target_dt=0.02)
        # Single waypoint -> all output steps are that waypoint
        np.testing.assert_allclose(result, 0.5)

    def test_endpoints_match(self):
        from unitree_launcher.trajectory import TrajectoryPlan, resample_trajectory

        n_plan = 100
        wp = np.linspace(0, 1, n_plan).reshape(-1, 1).repeat(29, axis=1)
        plan = TrajectoryPlan(waypoints=wp, dt_plan=0.01, converged=True)
        result = resample_trajectory(plan, target_duration=2.0, target_dt=0.02)
        # First output should match first waypoint
        np.testing.assert_allclose(result[0], wp[0], atol=1e-10)

    def test_empty_plan_raises(self):
        from unitree_launcher.trajectory import TrajectoryPlan, resample_trajectory

        plan = TrajectoryPlan(waypoints=np.zeros((0, 29)), dt_plan=0.02, converged=False)
        with pytest.raises(ValueError, match="Empty"):
            resample_trajectory(plan, target_duration=1.0, target_dt=0.02)


# ---------------------------------------------------------------------------
# Gantry wrapper tests
# ---------------------------------------------------------------------------

class TestPlanInterpolationTrajectory:
    """Tests for the gantry.py convenience wrapper."""

    def test_raises_when_mink_missing(self, home_q, zero_q):
        """When mink is not installed, ImportError propagates."""
        with mock.patch.dict("sys.modules", {"mink": None}):
            from unitree_launcher.gantry import plan_interpolation_trajectory
            with pytest.raises((ImportError, ModuleNotFoundError)):
                plan_interpolation_trajectory(zero_q, home_q)

    def test_raises_on_planning_error(self, home_q, zero_q):
        """When planning raises, error propagates."""
        with mock.patch(
            "unitree_launcher.trajectory.plan_trajectory",
            side_effect=RuntimeError("test error"),
        ):
            from unitree_launcher.gantry import plan_interpolation_trajectory
            with pytest.raises(RuntimeError, match="test error"):
                plan_interpolation_trajectory(zero_q, home_q)

    def test_default_goal_is_home(self, zero_q):
        """When q_goal is None, uses home positions."""
        from unitree_launcher.gantry import plan_interpolation_trajectory, build_home_positions

        home_q = build_home_positions()

        with mock.patch("unitree_launcher.trajectory.plan_trajectory") as mock_plan:
            from unitree_launcher.trajectory import TrajectoryPlan
            mock_plan.return_value = TrajectoryPlan(
                waypoints=np.tile(home_q, (10, 1)),
                dt_plan=0.02,
                converged=True,
            )
            result = plan_interpolation_trajectory(zero_q)
            call_args = mock_plan.call_args
            np.testing.assert_array_equal(call_args[0][1], home_q)


# ---------------------------------------------------------------------------
# Integration test (requires mink — skip if not installed)
# ---------------------------------------------------------------------------

mink_available = importlib.util.find_spec("mink") is not None


@pytest.mark.skipif(not mink_available, reason="mink not installed")
@pytest.mark.slow
class TestPlanTrajectoryIntegration:
    """Integration tests that actually run the mink planner."""

    def test_identity_plan_converges_immediately(self, home_q):
        """Planning from home to home should converge in very few steps."""
        from unitree_launcher.trajectory import plan_trajectory

        plan = plan_trajectory(home_q, home_q, max_steps=10)
        assert plan.converged
        assert len(plan.waypoints) <= 10

    def test_small_perturbation_converges(self, home_q):
        """Small offset from home should converge."""
        from unitree_launcher.trajectory import plan_trajectory

        start_q = home_q.copy()
        start_q[15] += 0.1  # Slightly perturb left shoulder pitch
        plan = plan_trajectory(start_q, home_q, max_steps=200)
        assert plan.converged
        # Final waypoint should be close to home
        np.testing.assert_allclose(
            plan.waypoints[-1], home_q, atol=0.02,
        )

    def test_waypoints_shape(self, home_q, zero_q):
        """Waypoints should have correct shape."""
        from unitree_launcher.trajectory import plan_trajectory

        plan = plan_trajectory(zero_q, home_q, max_steps=50)
        assert plan.waypoints.ndim == 2
        assert plan.waypoints.shape[1] == 29
        assert plan.waypoints.shape[0] <= 50

    def test_full_pipeline(self, home_q, zero_q):
        """End-to-end: plan + resample produces correct shape."""
        from unitree_launcher.trajectory import plan_trajectory, resample_trajectory

        plan = plan_trajectory(zero_q, home_q, max_steps=100)
        result = resample_trajectory(plan, target_duration=5.0, target_dt=0.02)
        assert result.shape == (250, 29)
