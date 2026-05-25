"""Optional Rust backend for axial scan export kernels."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from well_array_sim.internal.axial_scan import AxialScanResult
from well_array_sim.internal.pipe import BoreFluid, Pipe2D, SteelWall
from well_array_sim.internal.transducer import PointTransducer
from well_array_sim.internal.wall_profile import EchoConfig, WallProfile

if TYPE_CHECKING:
    from well_array_sim.internal.scenario import InferenceConfig

_RUST_MODULE = None
_RUST_IMPORT_TRIED = False


def _load_rust_module():
    global _RUST_MODULE, _RUST_IMPORT_TRIED
    if _RUST_IMPORT_TRIED:
        return _RUST_MODULE
    _RUST_IMPORT_TRIED = True
    try:
        import well_array_sim_core as core

        if core.extension_available():
            _RUST_MODULE = core
    except ImportError:
        _RUST_MODULE = None
    return _RUST_MODULE


def rust_available() -> bool:
    return _load_rust_module() is not None


def rust_enabled() -> bool:
    if os.environ.get("WELL_ARRAY_SIM_USE_RUST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return rust_available()
    return False


def wall_profile_arrays(
    profile: WallProfile | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if profile is None:
        return None, None, None, None
    amp = profile.amplitude_multiplier
    return (
        np.ascontiguousarray(profile.z_m, dtype=float),
        np.ascontiguousarray(profile.theta_rad, dtype=float),
        np.ascontiguousarray(profile.inner_radius_m, dtype=float),
        None if amp is None else np.ascontiguousarray(amp, dtype=float),
    )


def simulate_axial_scan_rust(
    pipe: Pipe2D,
    fluid: BoreFluid,
    steel: SteelWall,
    transducer: PointTransducer,
    timing: dict[str, float],
    z_stations: np.ndarray,
    *,
    angle_step_deg: float = 1.0,
    z_step_m: float = 0.01,
    wall_profile: WallProfile | None = None,
    echo: EchoConfig | None = None,
    inference: InferenceConfig | None = None,
    store_waveforms: bool = True,
) -> AxialScanResult | None:
    core = _load_rust_module()
    if core is None:
        return None

    z_list = np.ascontiguousarray(z_stations, dtype=float)
    angles_deg = np.ascontiguousarray(np.arange(0.0, 360.0, angle_step_deg, dtype=float))
    wall_z, wall_theta, wall_r, wall_amp = wall_profile_arrays(wall_profile)

    if inference is None:
        r_min = max(0.01, pipe.inner_radius_m * 0.5)
        r_max = pipe.inner_radius_m * 1.5
        r_step = 0.0005
        inference_mode = "angular_saft"
        angular_window_deg = 15.0
        coherent_sum = True
    else:
        r_min = inference.r_min_m
        r_max = inference.r_max_m
        r_step = inference.r_step_m
        inference_mode = inference.mode
        angular_window_deg = inference.angular_window_deg
        coherent_sum = inference.coherent_sum

    amplitude_exponent = 0.0 if echo is None else echo.amplitude_exponent
    snr_db = None if echo is None else echo.snr_db
    noise_seed = None if echo is None else echo.noise_seed

    raw = core.simulate_axial_scan_rust(
        float(pipe.inner_radius_m),
        float(fluid.rho),
        float(fluid.vp),
        float(steel.rho),
        float(steel.vp),
        float(transducer.center_freq_hz),
        float(transducer.bandwidth),
        float(timing["t_end_us"]),
        float(timing["dt_us"]),
        8e-6,
        z_list,
        angles_deg,
        wall_z,
        wall_theta,
        wall_r,
        wall_amp,
        float(amplitude_exponent),
        snr_db,
        noise_seed,
        inference_mode,
        float(r_min),
        float(r_max),
        float(r_step),
        float(angular_window_deg),
        bool(coherent_sum),
        bool(store_waveforms),
    )

    p_rx = raw["p_rx"]
    if p_rx is None:
        p_rx_arr = None
    else:
        p_rx_arr = np.asarray(p_rx, dtype=float)

    return AxialScanResult(
        z_stations_m=np.asarray(raw["z_stations_m"], dtype=float),
        angles_deg=np.asarray(raw["angles_deg"], dtype=float),
        inferred_distance_m=np.asarray(raw["inferred_distance_m"], dtype=float),
        measured_echo_us=np.asarray(raw["measured_echo_us"], dtype=float),
        peak_amplitude=np.asarray(raw["peak_amplitude"], dtype=float),
        wall_distance_m=float(raw["wall_distance_m"]),
        ground_truth_distance_m=np.asarray(raw["ground_truth_distance_m"], dtype=float),
        angle_step_deg=float(raw["angle_step_deg"]),
        z_step_m=float(raw["z_step_m"]),
        engine=str(raw["engine"]),
        time_us=np.asarray(raw["time_us"], dtype=float),
        p_tx=np.asarray(raw["p_tx"], dtype=float),
        p_rx=p_rx_arr,
    )
