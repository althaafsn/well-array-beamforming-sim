"""Waveform sidecar for partition observation bundles."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from well_array_sim.export.schema import WAVEFORM_COLUMNS
from well_array_sim.internal.axial_scan import AxialScanResult


def build_waveforms_table(
    scan: AxialScanResult,
    *,
    z_local_m: np.ndarray | None = None,
) -> pa.Table:
    if scan.time_us is None or scan.p_rx is None or scan.p_tx is None:
        raise ValueError("AxialScanResult missing waveform arrays")
    dt_us = float(scan.time_us[1] - scan.time_us[0]) if len(scan.time_us) > 1 else 1.0
    sample_rate_hz = 1e6 / dt_us if dt_us > 0 else 0.0
    t0_us = float(scan.time_us[0])
    rows: dict[str, list] = {col: [] for col in WAVEFORM_COLUMNS}
    z_values = scan.z_stations_m if z_local_m is None else z_local_m
    for iz, z_m in enumerate(z_values):
        for it, theta_deg in enumerate(scan.angles_deg):
            trace = scan.p_rx[iz, it]
            rows["z_local_m"].append(float(z_m))
            rows["theta_deg"].append(float(theta_deg))
            rows["sample_rate_hz"].append(sample_rate_hz)
            rows["t0_us"].append(t0_us)
            rows["n_samples"].append(int(len(trace)))
            rows["p_tx"].append(scan.p_tx.tolist())
            rows["p_rx"].append(trace.tolist())
    return pa.table(rows)
