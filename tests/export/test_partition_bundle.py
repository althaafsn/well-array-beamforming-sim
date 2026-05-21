"""Tests for partition observation export bundles."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from well_array_sim.export.bundle import export_partition_observation_bundle
from well_array_sim.export.manifest import read_manifest
from well_array_sim.export.schema import EXPORT_SCHEMA_VERSION, OBSERVATION_GRID_COLUMNS, STATE_GRID_COLUMNS, WAVEFORM_COLUMNS


SCENARIO = Path("scenarios/internal_pipe_corrosion_default.yaml")


@pytest.fixture(scope="module")
def scenario_path() -> Path:
    if not SCENARIO.exists():
        pytest.skip("corrosion scenario missing")
    return SCENARIO


def _export(tmp_path: Path, scenario_path: Path, year: int, **kwargs) -> Path:
    defaults = {
        "partition_index": 0,
        "z_step_m": 0.20,
        "angle_step_deg": 90.0,
    }
    defaults.update(kwargs)
    return export_partition_observation_bundle(
        scenario_path=scenario_path,
        segment_id="test_seg",
        observation_year=year,
        out_root=tmp_path,
        **defaults,
    )


def _read_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).read().column_names


def _read_column_max(path: Path, column: str) -> float:
    table = pq.read_table(path, columns=[column])
    values = table[column].to_pylist()
    return float(max(values))


def test_manifest_schema_version(tmp_path: Path, scenario_path: Path) -> None:
    bundle_dir = _export(tmp_path, scenario_path, 0)
    manifest = read_manifest(bundle_dir / "manifest.json")
    assert manifest["export_schema_version"] == EXPORT_SCHEMA_VERSION
    assert manifest["bundle_type"] == "pipe_partition_observation"
    assert manifest["segment_id"] == "test_seg"
    assert manifest["partition_id"] == "test_seg@p0000"
    assert manifest["observation_year"] == 0
    assert "waveforms" in manifest["artifacts"]


def test_waveforms_always_exported(tmp_path: Path, scenario_path: Path) -> None:
    bundle_dir = _export(tmp_path, scenario_path, 0)
    assert (bundle_dir / "waveforms.parquet").exists()
    cols = _read_columns(bundle_dir / "waveforms.parquet")
    for column in WAVEFORM_COLUMNS:
        assert column in cols


def test_same_columns_year_0_and_10(tmp_path: Path, scenario_path: Path) -> None:
    dir0 = _export(tmp_path, scenario_path, 0)
    dir10 = _export(tmp_path, scenario_path, 10)
    state0_cols = _read_columns(dir0 / "state_grid.parquet")
    state10_cols = _read_columns(dir10 / "state_grid.parquet")
    obs0_cols = _read_columns(dir0 / "observation_grid.parquet")
    obs10_cols = _read_columns(dir10 / "observation_grid.parquet")
    assert state0_cols == state10_cols == STATE_GRID_COLUMNS
    assert obs0_cols == obs10_cols == OBSERVATION_GRID_COLUMNS
    assert _read_column_max(dir10 / "state_grid.parquet", "metal_loss_m") >= _read_column_max(
        dir0 / "state_grid.parquet", "metal_loss_m"
    )


def test_observation_grid_has_no_ground_truth_columns(tmp_path: Path, scenario_path: Path) -> None:
    bundle_dir = _export(tmp_path, scenario_path, 5)
    cols = set(_read_columns(bundle_dir / "observation_grid.parquet"))
    forbidden = {"ground_truth_distance_m", "ground_truth_inner_radius_m", "error_mm"}
    assert forbidden.isdisjoint(cols)


def test_chainage_offset_uses_local_z_in_tables(tmp_path: Path, scenario_path: Path) -> None:
    bundle_dir = _export(
        tmp_path,
        scenario_path,
        0,
        partition_index=3,
        chainage_start_m=1.2,
        axial_length_m=0.4,
    )
    manifest = read_manifest(bundle_dir / "manifest.json")
    assert manifest["chainage_start_m"] == pytest.approx(1.2)
    assert manifest["chainage_end_m"] == pytest.approx(1.6)

    obs = pq.read_table(bundle_dir / "observation_grid.parquet", columns=["z_local_m"])
    z_local = obs["z_local_m"].to_pylist()
    assert max(z_local) <= 0.4 + 1e-6
    assert min(z_local) >= -1e-6

    state = pq.read_table(bundle_dir / "state_grid.parquet", columns=["z_local_m", "wall_z_m"])
    wall_z = state["wall_z_m"].to_pylist()
    assert min(wall_z) >= 1.2 - 1e-6
