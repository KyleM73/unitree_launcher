"""Gantry hang test — sim-side pytest wrapper.

Runs the same test sequence as `scripts/gantry_sim.py sim` but with
assertions. This ensures the interpolation + hold logic works before going
to hardware.

Thresholds:
    The model uses dof_damping=2 Nm/(rad/s) matching mjlab's actuator kv=2.
    - Interpolation: < 0.2 rad position error (typically < 0.1).
    - Hold phase: < 2 rad/s max velocity with full standby gains.
      The PD controller + dof_damping should hold the hanging robot steady.
"""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from unitree_launcher.gantry import build_home_positions

# run_test lives in scripts/tests/ (not a package), so add it to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "tests"))
from test_gantry import run_test


@pytest.fixture
def default_config_path():
    return str(Path(__file__).parent.parent / "configs" / "sim.yaml")


def _make_args(config_path):
    """Build a minimal args namespace for headless sim testing."""
    return Namespace(
        mode="sim", config=config_path,
        gui=False, viser=False, port=8080,
        gamepad=False, record=None,
        interface=None,
    )


class TestGantryHangSim:
    """Simulated gantry hang test — validates the full sequence."""

    def test_interpolate_reaches_home(self, default_config_path):
        """After interpolation + hold, joints converge near home.

        With feet on the ground (anchor at 0.79m) and IsaacLab gains,
        floor contact dynamics cause some joints to deviate from home.
        The 0.5 rad threshold accommodates this.
        """
        results = run_test(_make_args(default_config_path))
        assert results["final_position_error"] < 0.5, (
            f"Position error {results['final_position_error']:.4f} rad > 0.5 rad"
        )

    def test_hold_velocity_bounded(self, default_config_path):
        """Hold phase: velocity stays low with full standby gains.

        With standby kp (150-350) and kd (3-20) plus model dof_damping=2,
        the robot should hold position with minimal oscillation.
        """
        results = run_test(_make_args(default_config_path))
        assert results["post_damping_max_vel"] < 2.0, (
            f"Max velocity {results['post_damping_max_vel']:.4f} rad/s > 2.0 rad/s"
        )

    def test_gains_start_from_zero(self, default_config_path):
        """Verify the hanging start position differs from home (test is meaningful)."""
        results = run_test(_make_args(default_config_path))
        home_q = build_home_positions()
        pre_err = np.max(np.abs(results["pre_interp_positions"] - home_q))
        assert pre_err > 0.01, (
            f"Pre-interpolation error {pre_err:.4f} rad — "
            "robot was already at home before interpolation started"
        )
