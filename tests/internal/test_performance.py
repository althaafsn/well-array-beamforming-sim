from __future__ import annotations

import numpy as np
import pytest

from well_array_sim.internal import load_internal_scenario, simulate_axial_scan
from well_array_sim.internal.geometry3d import (
    GROUND_TRUTH_N_THETA,
    GROUND_TRUTH_Z_STEP_M,
    clear_geometry_cache,
    ground_truth_cylinder_pointcloud,
)


@pytest.fixture
def scenario():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    return load_internal_scenario(root / "scenarios" / "internal_pipe_default.yaml")


def test_cached_gt_cylinder_is_deterministic(scenario) -> None:
    pipe3d = scenario.pipe_3d
    clear_geometry_cache()
    first = ground_truth_cylinder_pointcloud(pipe3d, radius_m=pipe3d.inner_radius_m)
    second = ground_truth_cylinder_pointcloud(pipe3d, radius_m=pipe3d.inner_radius_m)
    assert first.shape == second.shape
    assert np.allclose(first, second)


def test_ground_truth_cylinder_is_dense(scenario) -> None:
    pipe3d = scenario.pipe_3d
    pts = ground_truth_cylinder_pointcloud(pipe3d, radius_m=pipe3d.inner_radius_m)
    n_z = max(2, int(round(pipe3d.length_m / GROUND_TRUTH_Z_STEP_M)) + 1)
    assert pts.shape[0] == GROUND_TRUTH_N_THETA * n_z


def test_axial_default_scans_every_z_station(scenario) -> None:
    """Axial scan runs an independent ray+SAFT shot at each (z, θ)."""
    z_vals = np.array([0.0, 0.05, 0.1])
    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_vals,
        angle_step_deg=30.0,
        z_step_m=0.05,
        inference=scenario.inference,
    )
    assert scan.inferred_distance_m.shape == (3, 12)
    assert scan.p_rx is not None


def test_axial_scan_is_deterministic(scenario) -> None:
    z_vals = scenario.z_stations()[::10]
    kwargs = dict(
        angle_step_deg=30.0,
        z_step_m=0.01,
        inference=scenario.inference,
    )
    first = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_vals,
        **kwargs,
    )
    second = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_vals,
        **kwargs,
    )
    assert np.allclose(first.inferred_distance_m, second.inferred_distance_m)
    assert np.allclose(first.measured_echo_us, second.measured_echo_us)
    assert np.allclose(first.peak_amplitude, second.peak_amplitude)
