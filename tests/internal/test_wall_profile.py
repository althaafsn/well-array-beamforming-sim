from __future__ import annotations

import math

import numpy as np
import pytest

from well_array_sim.internal import (
    BoreFluid,
    Pipe2D,
    PointTransducer,
    SteelWall,
    load_internal_scenario,
    simulate_pulse_echo_2d,
)
from well_array_sim.internal.wall_profile import (
    EchoConfig,
    build_wall_profile,
    echo_amplitude_scale,
    inner_radius_at,
)


def _uniform_setup():
    scenario = load_internal_scenario("scenarios/internal_pipe_default.yaml")
    pipe = Pipe2D(inner_radius_m=0.1092375, wall_thickness_m=0.013)
    fluid = BoreFluid(rho=900, vp=1300)
    steel = SteelWall(rho=7850, vp=5778)
    transducer = PointTransducer(center_freq_hz=250000, bandwidth=0.7)
    timing = scenario.timing
    return pipe, fluid, steel, transducer, timing, scenario.inference


def test_uniform_pipe_regression_without_profile() -> None:
    pipe, fluid, steel, transducer, timing, inference = _uniform_setup()
    result = simulate_pulse_echo_2d(
        pipe,
        fluid,
        steel,
        transducer,
        timing,
        math.radians(45.0),
        inference=inference,
    )
    assert result.ground_truth_distance_m == pytest.approx(pipe.inner_radius_m)
    assert result.inferred_distance_m == pytest.approx(pipe.inner_radius_m, abs=0.005)


def test_bump_profile_increases_local_radius() -> None:
    raw = {
        "wall_profile": {
            "n_theta": 72,
            "bump": {"theta_deg": 45.0, "width_deg": 30.0, "delta_m": 0.006},
        }
    }
    profile = build_wall_profile(raw, z_stations=np.array([0.0]), nominal_inner_radius_m=0.109)
    assert profile is not None
    r_bump = inner_radius_at(profile, z_m=0.0, theta_rad=math.radians(45.0), nominal_inner_radius_m=0.109)
    r_away = inner_radius_at(profile, z_m=0.0, theta_rad=math.radians(200.0), nominal_inner_radius_m=0.109)
    assert r_bump == pytest.approx(0.115, abs=1e-6)
    assert r_away == pytest.approx(0.109, abs=1e-6)


def test_bump_shorter_tof_when_steered_at_bump() -> None:
    pipe, fluid, steel, transducer, timing, inference = _uniform_setup()
    raw = {
        "wall_profile": {
            "n_theta": 72,
            "bump": {"theta_deg": 45.0, "width_deg": 30.0, "delta_m": 0.006},
        },
        "echo": {"snr_db": 40.0, "noise_seed": 1},
    }
    profile = build_wall_profile(raw, z_stations=np.array([0.0]), nominal_inner_radius_m=pipe.inner_radius_m)
    echo = EchoConfig(snr_db=40.0, noise_seed=1)
    at_bump = simulate_pulse_echo_2d(
        pipe,
        fluid,
        steel,
        transducer,
        timing,
        math.radians(45.0),
        wall_profile=profile,
        echo=echo,
        inference=inference,
    )
    away = simulate_pulse_echo_2d(
        pipe,
        fluid,
        steel,
        transducer,
        timing,
        math.radians(200.0),
        wall_profile=profile,
        echo=echo,
        inference=inference,
    )
    assert at_bump.ground_truth_distance_m > pipe.inner_radius_m
    bump_us = 2.0 * at_bump.ground_truth_distance_m / fluid.vp * 1e6
    away_us = 2.0 * away.ground_truth_distance_m / fluid.vp * 1e6
    assert bump_us > away_us
    assert at_bump.inferred_distance_m == pytest.approx(at_bump.ground_truth_distance_m, abs=0.008)


def test_amplitude_exponent_scales_echo() -> None:
    pipe, _, _, _, _, _ = _uniform_setup()
    raw = {
        "wall_profile": {
            "n_theta": 72,
            "bump": {"theta_deg": 45.0, "width_deg": 30.0, "delta_m": 0.006},
        }
    }
    profile = build_wall_profile(raw, z_stations=np.array([0.0]), nominal_inner_radius_m=pipe.inner_radius_m)
    scale_bump = echo_amplitude_scale(
        profile,
        EchoConfig(amplitude_exponent=1.0),
        z_m=0.0,
        theta_rad=math.radians(45.0),
        nominal_inner_radius_m=pipe.inner_radius_m,
    )
    scale_clean = echo_amplitude_scale(
        profile,
        EchoConfig(amplitude_exponent=1.0),
        z_m=0.0,
        theta_rad=math.radians(200.0),
        nominal_inner_radius_m=pipe.inner_radius_m,
    )
    assert scale_bump < scale_clean


def test_trace_noise_changes_received_waveform() -> None:
    pipe, fluid, steel, transducer, timing, inference = _uniform_setup()
    clean = simulate_pulse_echo_2d(
        pipe,
        fluid,
        steel,
        transducer,
        timing,
        math.radians(45.0),
        inference=inference,
    )
    noisy = simulate_pulse_echo_2d(
        pipe,
        fluid,
        steel,
        transducer,
        timing,
        math.radians(45.0),
        echo=EchoConfig(snr_db=10.0, noise_seed=42),
        inference=inference,
    )
    assert not np.allclose(clean.p_rx, noisy.p_rx)
    assert clean.ground_truth_distance_m == noisy.ground_truth_distance_m


def test_wavy_wall_scenario_loads() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_wavy_wall.yaml")
    assert scenario.wall_profile is not None
    assert scenario.echo is not None
    assert scenario.echo.snr_db == 60.0
