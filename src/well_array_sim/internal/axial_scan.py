from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from well_array_sim.internal.pipe import BoreFluid, Pipe2D, SteelWall
from well_array_sim.internal.transducer import PointTransducer
from well_array_sim.internal.wall_profile import EchoConfig, WallProfile

if TYPE_CHECKING:
    from well_array_sim.internal.scenario import InferenceConfig


def z_stations_m(
    length_m: float,
    z_step_m: float = 0.01,
    z_start_m: float = 0.0,
    z_end_m: float | None = None,
) -> np.ndarray:
    """Axial station grid along the pipe (inclusive end when step-aligned)."""
    z_end = length_m if z_end_m is None else float(z_end_m)
    return np.arange(z_start_m, z_end + 0.5 * z_step_m, z_step_m, dtype=float)


@dataclass(frozen=True)
class AxialScanResult:
    """Ray-packet + blind SAFT sweep over (z, θ)."""

    z_stations_m: np.ndarray
    angles_deg: np.ndarray
    inferred_distance_m: np.ndarray
    measured_echo_us: np.ndarray
    peak_amplitude: np.ndarray
    wall_distance_m: float
    ground_truth_distance_m: np.ndarray
    angle_step_deg: float
    z_step_m: float
    engine: str = "ray"
    time_us: np.ndarray | None = None
    p_tx: np.ndarray | None = None
    p_rx: np.ndarray | None = None

    @property
    def error_mm(self) -> np.ndarray:
        return (self.inferred_distance_m - self.ground_truth_distance_m) * 1000.0

    @property
    def n_z(self) -> int:
        return len(self.z_stations_m)

    @property
    def n_theta(self) -> int:
        return len(self.angles_deg)


def simulate_axial_scan(
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
) -> AxialScanResult:
    """
    Ray-packet pulse-echo + blind SAFT at each (z, θ) station.

    Independent 2D slices (φ = 0, no inter-station coupling). Waveforms are
    retained for downstream NDT export; visualization uses inferred R(θ, z) only.
    """
    from well_array_sim.internal.ray_forward import simulate_pulse_echo_2d

    z_list = np.asarray(z_stations, dtype=float)
    if len(z_list) == 0:
        raise ValueError("z_stations must contain at least one point")

    angles_deg = np.arange(0.0, 360.0, angle_step_deg, dtype=float)
    n_z = len(z_list)
    n_theta = len(angles_deg)

    inferred = np.zeros((n_z, n_theta), dtype=float)
    measured_us = np.zeros((n_z, n_theta), dtype=float)
    peaks = np.zeros((n_z, n_theta), dtype=float)
    ground_truth = np.zeros((n_z, n_theta), dtype=float)

    time_us: np.ndarray | None = None
    p_tx: np.ndarray | None = None
    p_rx: np.ndarray | None = None

    for iz, z_m in enumerate(z_list):
        for it, angle_deg in enumerate(angles_deg):
            shot = simulate_pulse_echo_2d(
                pipe,
                fluid,
                steel,
                transducer,
                timing,
                float(np.deg2rad(angle_deg)),
                z_m=float(z_m),
                wall_profile=wall_profile,
                echo=echo,
                inference=inference,
            )
            inferred[iz, it] = shot.inferred_distance_m
            ground_truth[iz, it] = shot.ground_truth_distance_m
            measured_us[iz, it] = 2.0 * shot.inferred_distance_m / fluid.vp * 1e6
            peaks[iz, it] = float(np.max(np.abs(shot.p_rx)))

            if store_waveforms:
                if time_us is None:
                    time_us = shot.time_us.copy()
                    p_tx = shot.p_tx.copy()
                    p_rx = np.zeros((n_z, n_theta, len(time_us)), dtype=float)
                p_rx[iz, it] = shot.p_rx

    return AxialScanResult(
        z_stations_m=z_list,
        angles_deg=angles_deg,
        inferred_distance_m=inferred,
        measured_echo_us=measured_us,
        peak_amplitude=peaks,
        wall_distance_m=pipe.inner_radius_m,
        ground_truth_distance_m=ground_truth,
        angle_step_deg=angle_step_deg,
        z_step_m=z_step_m,
        engine="ray",
        time_us=time_us,
        p_tx=p_tx,
        p_rx=p_rx,
    )
