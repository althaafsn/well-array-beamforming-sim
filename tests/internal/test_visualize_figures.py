from __future__ import annotations

import itertools
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from well_array_sim.internal.gui_views import MODE_AXIAL, MODE_SINGLE, build_figure_for_view
from well_array_sim.internal import load_internal_scenario, simulate_axial_scan, simulate_pulse_echo_2d


@pytest.fixture
def scenario():
    return load_internal_scenario("scenarios/internal_pipe_default.yaml")


@pytest.fixture
def pulse_result(scenario):
    return simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(45.0),
        inference=scenario.inference,
    )


@pytest.mark.parametrize(
    ("show_inferred", "show_ground_truth"),
    list(itertools.product([True, False], repeat=2)),
)
def test_figure_pulse_echo_overlay_combinations(
    pulse_result,
    show_inferred: bool,
    show_ground_truth: bool,
) -> None:
    from well_array_sim.internal.visualize import figure_pulse_echo

    fig = figure_pulse_echo(
        pulse_result,
        show_inferred=show_inferred,
        show_ground_truth=show_ground_truth,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_build_figure_for_view_pulse_echo(scenario, pulse_result) -> None:
    fig = build_figure_for_view(
        mode=MODE_SINGLE,
        view="Pulse echo",
        scenario=scenario,
        pulse_echo_result=pulse_result,
        axial_result=None,
        corrosion_snapshot=None,
        show_inferred=True,
        show_ground_truth=False,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_figure_axial_cylinder_map_builds(scenario) -> None:
    from well_array_sim.internal.visualize import figure_axial_cylinder_map

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
    fig = figure_axial_cylinder_map(axial, scenario)
    assert fig.axes
    plt.close(fig)


def test_figure_axial_radius_map_builds(scenario) -> None:
    from well_array_sim.internal.visualize import figure_axial_radius_map

    axial = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0, 0.1]),
        angle_step_deg=30.0,
        z_step_m=0.01,
        inference=scenario.inference,
    )
    fig = figure_axial_radius_map(axial, scenario)
    assert fig.axes
    plt.close(fig)


@pytest.mark.parametrize(
    ("view", "show_inferred", "show_ground_truth"),
    [
        ("SAFT point cloud", True, False),
        ("SAFT point cloud", False, True),
        ("Radius map", True, True),
        ("Radius map", False, False),
    ],
)
def test_axial_overlay_combinations(
    scenario,
    view: str,
    show_inferred: bool,
    show_ground_truth: bool,
) -> None:
    axial = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=90.0,
        inference=scenario.inference,
    )
    fig = build_figure_for_view(
        mode=MODE_AXIAL,
        view=view,
        scenario=scenario,
        pulse_echo_result=None,
        axial_result=axial,
        corrosion_snapshot=None,
        show_inferred=show_inferred,
        show_ground_truth=show_ground_truth,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_build_figure_for_view_axial(scenario) -> None:
    axial = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0, 0.1]),
        angle_step_deg=30.0,
        z_step_m=0.01,
        inference=scenario.inference,
    )
    fig = build_figure_for_view(
        mode=MODE_AXIAL,
        view="SAFT point cloud",
        scenario=scenario,
        pulse_echo_result=None,
        axial_result=axial,
        corrosion_snapshot=None,
        show_inferred=True,
        show_ground_truth=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
