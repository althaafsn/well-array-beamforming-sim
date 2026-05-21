"""Map axial scan results to blind observation grid (no ground truth)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from well_array_sim.export.schema import FORBIDDEN_OBSERVATION_COLUMNS, OBSERVATION_GRID_COLUMNS
from well_array_sim.internal.axial_scan import AxialScanResult


def build_observation_grid_table(
    scan: AxialScanResult,
    *,
    z_local_m: np.ndarray | None = None,
) -> pa.Table:
    rows: dict[str, list] = {col: [] for col in OBSERVATION_GRID_COLUMNS}
    z_values = scan.z_stations_m if z_local_m is None else z_local_m
    for iz, z_m in enumerate(z_values):
        for it, theta_deg in enumerate(scan.angles_deg):
            rows["z_local_m"].append(float(z_m))
            rows["theta_deg"].append(float(theta_deg))
            rows["inferred_inner_radius_m"].append(float(scan.inferred_distance_m[iz, it]))
            rows["echo_time_us"].append(float(scan.measured_echo_us[iz, it]))
            rows["peak_amplitude"].append(float(scan.peak_amplitude[iz, it]))
            rows["engine"].append(scan.engine)
    forbidden = set(rows.keys()) & FORBIDDEN_OBSERVATION_COLUMNS
    if forbidden:
        raise ValueError(f"Observation grid must not contain ground-truth columns: {forbidden}")
    return pa.table(rows)
