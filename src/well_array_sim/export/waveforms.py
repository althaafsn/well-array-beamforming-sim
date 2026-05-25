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
    z_values = np.asarray(
        scan.z_stations_m if z_local_m is None else z_local_m,
        dtype=float,
    )
    angles_deg = np.asarray(scan.angles_deg, dtype=float)
    p_tx = np.asarray(scan.p_tx, dtype=float)
    p_rx = np.asarray(scan.p_rx, dtype=float)

    n_z = len(z_values)
    n_theta = len(angles_deg)
    n_rows = n_z * n_theta
    n_samples = int(p_tx.shape[-1] if p_tx.ndim > 1 else len(p_tx))

    dt_us = float(scan.time_us[1] - scan.time_us[0]) if len(scan.time_us) > 1 else 1.0
    sample_rate_hz = 1e6 / dt_us if dt_us > 0 else 0.0
    t0_us = float(scan.time_us[0])

    z_col = np.repeat(z_values, n_theta)
    theta_col = np.tile(angles_deg, n_z)
    flat_tx = np.tile(p_tx, n_rows)
    flat_rx = p_rx.reshape(n_rows, n_samples).ravel()

    columns = {
        "z_local_m": pa.array(z_col, type=pa.float64()),
        "theta_deg": pa.array(theta_col, type=pa.float64()),
        "sample_rate_hz": pa.array(np.full(n_rows, sample_rate_hz), type=pa.float64()),
        "t0_us": pa.array(np.full(n_rows, t0_us), type=pa.float64()),
        "n_samples": pa.array(np.full(n_rows, n_samples, dtype=np.int64), type=pa.int64()),
        "p_tx": pa.FixedSizeListArray.from_arrays(flat_tx, n_samples),
        "p_rx": pa.FixedSizeListArray.from_arrays(flat_rx, n_samples),
    }
    return pa.table([columns[name] for name in WAVEFORM_COLUMNS], names=WAVEFORM_COLUMNS)
