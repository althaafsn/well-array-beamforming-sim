from __future__ import annotations

import math

import pytest
from matplotlib.figure import Figure

from well_array_sim.internal import load_internal_scenario, simulate_pulse_echo_2d
from well_array_sim.internal.gui_views import build_figure_for_view


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


def test_build_figure_pulse_echo(scenario, pulse_result) -> None:
    fig = build_figure_for_view(
        mode="single",
        view="Pulse echo",
        scenario=scenario,
        pulse_echo_result=pulse_result,
        axial_result=None,
        corrosion_snapshot=None,
        show_inferred=True,
        show_ground_truth=True,
    )
    assert isinstance(fig, Figure)


def test_build_figure_packet_scene(scenario, pulse_result) -> None:
    t_s = pulse_result.ground_truth_distance_m / pulse_result.fluid_vp
    fig = build_figure_for_view(
        mode="single",
        view="Packet scene",
        scenario=scenario,
        pulse_echo_result=pulse_result,
        axial_result=None,
        corrosion_snapshot=None,
        show_inferred=True,
        show_ground_truth=True,
        packet_time_s=t_s,
    )
    assert isinstance(fig, Figure)
