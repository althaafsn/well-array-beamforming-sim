from __future__ import annotations

import numpy as np

from well_array_sim.internal.imaging import inference_range_grid, saft_range_profile
from well_array_sim.internal.pipe import BoreFluid, Pipe2D, SteelWall
from well_array_sim.internal.pulse_echo_result import PulseEchoResult
from well_array_sim.internal.scenario import InferenceConfig
from well_array_sim.internal.transducer import PointTransducer
from well_array_sim.internal.wall_profile import (
    EchoConfig,
    WallProfile,
    echo_amplitude_scale,
    inner_radius_at,
)
from well_array_sim.internal.wave_packet import (
    add_trace_noise,
    beam_direction_xy,
    spawn_packets_from_pulse,
    synthesize_received_trace,
    trajectory_from_packets,
)


def _reflection_coeff(fluid: BoreFluid, steel: SteelWall) -> float:
    z_fluid = fluid.impedance
    z_steel = steel.impedance
    return (z_steel - z_fluid) / (z_steel + z_fluid)


def simulate_pulse_echo_2d(
    pipe: Pipe2D,
    fluid: BoreFluid,
    steel: SteelWall,
    transducer: PointTransducer,
    timing: dict[str, float],
    theta_rad: float,
    *,
    z_m: float = 0.0,
    wall_profile: WallProfile | None = None,
    echo: EchoConfig | None = None,
    inference: InferenceConfig | None = None,
) -> PulseEchoResult:
    """
    Ray-packet forward model + blind SAFT inference at one (θ, z) station.

    Forward physics uses true wall radius for reflection timing (sim world).
    Inference uses only p_rx, p_tx, and the blind range grid.
    """
    time_s, p_tx = transducer.transmit_pulse(timing)
    ground_truth_m = inner_radius_at(
        wall_profile,
        z_m=z_m,
        theta_rad=theta_rad,
        nominal_inner_radius_m=pipe.inner_radius_m,
    )
    amplitude_scale = echo_amplitude_scale(
        wall_profile,
        echo,
        z_m=z_m,
        theta_rad=theta_rad,
        nominal_inner_radius_m=pipe.inner_radius_m,
    )
    reflection_coeff = _reflection_coeff(fluid, steel)

    packets = spawn_packets_from_pulse(
        time_s,
        p_tx,
        theta_rad=theta_rad,
        wall_distance_m=ground_truth_m,
        fluid_vp=fluid.vp,
        center_freq_hz=transducer.center_freq_hz,
        bandwidth=transducer.bandwidth,
    )
    p_rx = synthesize_received_trace(
        time_s,
        packets,
        reflection_coeff=reflection_coeff,
        amplitude_scale=amplitude_scale,
    )

    nominal_packets = spawn_packets_from_pulse(
        time_s,
        p_tx,
        theta_rad=theta_rad,
        wall_distance_m=pipe.inner_radius_m,
        fluid_vp=fluid.vp,
        center_freq_hz=transducer.center_freq_hz,
        bandwidth=transducer.bandwidth,
    )
    p_rx_nominal = synthesize_received_trace(
        time_s,
        nominal_packets,
        reflection_coeff=reflection_coeff,
        amplitude_scale=1.0,
    )
    reference_peak = float(np.max(np.abs(p_rx_nominal)))

    noise_rng = None
    if echo is not None and echo.noise_seed is not None:
        noise_rng = np.random.default_rng(echo.noise_seed)
    snr_db = echo.snr_db if echo is not None else None
    p_rx = add_trace_noise(
        p_rx,
        snr_db=snr_db,
        rng=noise_rng,
        reference_peak=reference_peak,
    )

    if inference is None:
        r_min = max(0.01, pipe.inner_radius_m * 0.5)
        r_max = pipe.inner_radius_m * 1.5
        r_step = 0.0005
    else:
        r_min = inference.r_min_m
        r_max = inference.r_max_m
        r_step = inference.r_step_m

    r_grid = inference_range_grid(r_min_m=r_min, r_max_m=r_max, r_step_m=r_step)
    range_profile_I, inferred_m = saft_range_profile(
        p_rx,
        p_tx,
        time_s,
        r_grid,
        fluid.vp,
    )

    return PulseEchoResult(
        theta_rad=theta_rad,
        z_m=z_m,
        beam_dir_xy=beam_direction_xy(theta_rad),
        wall_distance_m=pipe.inner_radius_m,
        ground_truth_distance_m=ground_truth_m,
        inferred_distance_m=inferred_m,
        reflection_coeff=reflection_coeff,
        time_us=time_s * 1e6,
        p_tx=p_tx.copy(),
        p_rx=p_rx.copy(),
        range_profile_r_m=r_grid,
        range_profile_I=range_profile_I,
        trajectory=trajectory_from_packets(packets),
        fluid_vp=fluid.vp,
    )
