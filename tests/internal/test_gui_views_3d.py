from __future__ import annotations

import math

import pytest
from matplotlib.figure import Figure

from well_array_sim.internal import load_internal_scenario, simulate_axial_scan
from well_array_sim.internal.geometry3d import (
    GROUND_TRUTH_N_THETA,
    GROUND_TRUTH_Z_STEP_M,
    ground_truth_axial_pointcloud_3d,
)
from well_array_sim.internal.gui_views import build_figure_for_view


@pytest.fixture
def scenario():
    return load_internal_scenario("scenarios/internal_pipe_default.yaml")


def test_build_figure_for_view_axial(scenario) -> None:
    axial = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        scenario.z_stations()[::10],
        angle_step_deg=30.0,
        z_step_m=0.01,
        inference=scenario.inference,
    )
    fig = build_figure_for_view(
        mode="axial",
        view="Inferred point cloud",
        scenario=scenario,
        pulse_echo_result=None,
        axial_result=axial,
        corrosion_snapshot=None,
        show_inferred=True,
        show_ground_truth=True,
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1

    expected_n = ground_truth_axial_pointcloud_3d(
        scenario.pipe_3d,
        wall_profile=scenario.wall_profile,
        nominal_inner_radius_m=scenario.pipe.inner_radius_m,
    ).shape[0]
    assert expected_n == GROUND_TRUTH_N_THETA * (
        max(2, int(round(scenario.pipe_3d.length_m / GROUND_TRUTH_Z_STEP_M)) + 1)
    )
    assert expected_n > len(axial.angles_deg) * len(axial.z_stations_m)
