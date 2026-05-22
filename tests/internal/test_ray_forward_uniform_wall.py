from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from well_array_sim.internal import load_internal_scenario, simulate_pulse_echo_2d


def test_inferred_distance_matches_uniform_wall() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_default.yaml")
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
    assert abs(result.inferred_distance_m - scenario.pipe.inner_radius_m) < 0.002


def test_echo_arrival_time_matches_round_trip() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_default.yaml")
    result = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(0.0),
        inference=scenario.inference,
    )
    expected_us = 2.0 * result.ground_truth_distance_m / result.fluid_vp * 1e6
    dt_us = float(result.time_us[1] - result.time_us[0])
    lag = int(round(expected_us / dt_us))
    echo_slice = result.p_rx[lag : lag + 80]
    assert np.max(np.abs(echo_slice)) > 0.1


def test_matched_filter_profile_peaks_near_ground_truth() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_default.yaml")
    result = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(45.0),
        inference=scenario.inference,
    )
    peak_r = float(result.range_profile_r_m[int(np.argmax(result.range_profile_I))])
    assert abs(peak_r - result.ground_truth_distance_m) < 0.002
    assert abs(result.inferred_distance_m - result.ground_truth_distance_m) < 0.002


def test_received_waveform_has_echo_energy() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_default.yaml")
    result = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(90.0),
        inference=scenario.inference,
    )
    assert np.max(np.abs(result.p_rx)) > 0.01
