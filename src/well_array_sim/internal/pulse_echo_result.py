from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from well_array_sim.internal.wave_packet import PacketTrajectory


@dataclass(frozen=True)
class PulseEchoResult:
    """Ray-packet pulse-echo simulation at one (θ, z) station."""

    theta_rad: float
    z_m: float
    beam_dir_xy: np.ndarray
    wall_distance_m: float
    ground_truth_distance_m: float
    inferred_distance_m: float
    reflection_coeff: float
    time_us: np.ndarray
    p_tx: np.ndarray
    p_rx: np.ndarray
    range_profile_r_m: np.ndarray
    range_profile_I: np.ndarray
    trajectory: PacketTrajectory
    fluid_vp: float

    @property
    def error_mm(self) -> float:
        return (self.inferred_distance_m - self.ground_truth_distance_m) * 1000.0

    @property
    def measured_echo_us(self) -> float:
        dt_us = float(self.time_us[1] - self.time_us[0]) if len(self.time_us) > 1 else 0.5
        idx = int(round(2.0 * self.inferred_distance_m / self.fluid_vp / (dt_us * 1e-6)))
        idx = max(0, min(idx, len(self.time_us) - 1))
        return float(self.time_us[idx])
