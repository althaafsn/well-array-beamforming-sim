"""Tests for vectorized waveform table export."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from well_array_sim.export.schema import WAVEFORM_COLUMNS
from well_array_sim.export.waveforms import build_waveforms_table
from well_array_sim.internal.axial_scan import AxialScanResult


def _synthetic_scan(*, n_z: int = 2, n_theta: int = 3, n_samples: int = 8) -> AxialScanResult:
    z = np.linspace(0.0, 0.05, n_z)
    angles = np.array([0.0, 90.0, 180.0], dtype=float)[:n_theta]
    p_tx = np.arange(n_samples, dtype=float)
    p_rx = np.zeros((n_z, n_theta, n_samples), dtype=float)
    for iz in range(n_z):
        for it in range(n_theta):
            p_rx[iz, it, :] = float(iz * 100 + it)
    return AxialScanResult(
        z_stations_m=z,
        angles_deg=angles,
        inferred_distance_m=np.zeros((n_z, n_theta)),
        measured_echo_us=np.zeros((n_z, n_theta)),
        peak_amplitude=np.zeros((n_z, n_theta)),
        wall_distance_m=0.1,
        ground_truth_distance_m=np.zeros((n_z, n_theta)),
        angle_step_deg=90.0,
        z_step_m=0.05,
        time_us=np.arange(n_samples, dtype=float),
        p_tx=p_tx,
        p_rx=p_rx,
    )


def test_waveforms_table_columns_and_row_count() -> None:
    scan = _synthetic_scan()
    table = build_waveforms_table(scan)
    assert table.column_names == WAVEFORM_COLUMNS
    assert table.num_rows == 2 * 3


def test_waveforms_table_row_order_and_traces() -> None:
    scan = _synthetic_scan()
    table = build_waveforms_table(scan)

    z_col = table["z_local_m"].to_pylist()
    theta_col = table["theta_deg"].to_pylist()
    p_tx_col = table["p_tx"].to_pylist()
    p_rx_col = table["p_rx"].to_pylist()

    assert z_col == [0.0, 0.0, 0.0, 0.05, 0.05, 0.05]
    assert theta_col == [0.0, 90.0, 180.0, 0.0, 90.0, 180.0]
    assert p_tx_col[0] == scan.p_tx.tolist()
    assert p_tx_col[-1] == scan.p_tx.tolist()
    assert p_rx_col[0] == scan.p_rx[0, 0].tolist()
    assert p_rx_col[3] == scan.p_rx[1, 0].tolist()
    assert p_rx_col[1] == scan.p_rx[0, 1].tolist()


def test_waveforms_table_z_local_override() -> None:
    scan = _synthetic_scan(n_z=2, n_theta=2)
    z_local = np.array([0.01, 0.02])
    table = build_waveforms_table(scan, z_local_m=z_local)
    assert table["z_local_m"].to_pylist() == [0.01, 0.01, 0.02, 0.02]


def test_waveforms_table_scalar_metadata() -> None:
    scan = _synthetic_scan(n_z=1, n_theta=1, n_samples=4)
    table = build_waveforms_table(scan)
    assert table["sample_rate_hz"].to_pylist() == [1e6]
    assert table["t0_us"].to_pylist() == [0.0]
    assert table["n_samples"].to_pylist() == [4]
    p_tx_type = table.schema.field("p_tx").type
    assert pa.types.is_list(p_tx_type) or pa.types.is_fixed_size_list(p_tx_type)
