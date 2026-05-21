from __future__ import annotations

import math

from well_array_sim.internal import load_internal_scenario, simulate_pulse_echo_2d


def test_saft_finds_bump_at_45_deg() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_wavy_wall.yaml")
    result = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(45.0),
        wall_profile=scenario.wall_profile,
        echo=scenario.echo,
        inference=scenario.inference,
    )
    assert result.ground_truth_distance_m > scenario.pipe.inner_radius_m
    assert abs(result.inferred_distance_m - result.ground_truth_distance_m) < 0.008


def test_saft_nominal_radius_away_from_bump() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_wavy_wall.yaml")
    result = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(200.0),
        wall_profile=scenario.wall_profile,
        echo=scenario.echo,
        inference=scenario.inference,
    )
    assert abs(result.ground_truth_distance_m - scenario.pipe.inner_radius_m) < 1e-6
    assert abs(result.inferred_distance_m - scenario.pipe.inner_radius_m) < 0.020
