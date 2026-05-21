from __future__ import annotations

import math

import numpy as np

from well_array_sim.internal import load_internal_scenario, simulate_pulse_echo_2d
from well_array_sim.internal.ray_forward import _reflection_coeff
from well_array_sim.internal.wave_packet import (
    _arrival_index,
    spawn_packets_from_pulse,
    synthesize_received_trace,
)
from well_array_sim.core.pulse import gaussian_tone_burst, make_time_axis


def test_superposition_matches_shifted_tx_pulse() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_default.yaml")
    timing = scenario.timing
    time_s = make_time_axis(timing["t_end_us"], timing["dt_us"])
    p_tx = gaussian_tone_burst(
        time_s,
        f0_hz=scenario.transducer.center_freq_hz,
        bandwidth=scenario.transducer.bandwidth,
    )
    wall_r = scenario.pipe.inner_radius_m
    rc = _reflection_coeff(scenario.fluid, scenario.steel)
    packets = spawn_packets_from_pulse(
        time_s,
        p_tx,
        theta_rad=math.radians(30.0),
        wall_distance_m=wall_r,
        fluid_vp=scenario.fluid.vp,
        center_freq_hz=scenario.transducer.center_freq_hz,
        bandwidth=scenario.transducer.bandwidth,
    )
    p_rx = synthesize_received_trace(
        time_s,
        packets,
        reflection_coeff=rc,
        amplitude_scale=1.0,
    )
    expected = np.zeros_like(p_rx)
    for packet in packets:
        idx = _arrival_index(time_s, packet.arrival_time_s)
        if 0 <= idx < len(expected):
            expected[idx] += rc * packet.amplitude

    dt_s = float(time_s[1] - time_s[0])
    lag = int(round(2.0 * wall_r / scenario.fluid.vp / dt_s))
    shifted = np.zeros_like(p_rx)
    shifted[lag : lag + len(p_tx)] = rc * p_tx[: max(0, len(p_rx) - lag)]

    assert np.allclose(p_rx, expected, atol=1e-10)
    assert np.allclose(p_rx, shifted, atol=1e-10)

    sim = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(30.0),
        inference=scenario.inference,
    )
    assert np.allclose(sim.p_rx, expected, atol=1e-10)
