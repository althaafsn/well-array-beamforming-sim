"""Tests for BC segment multi-year study workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from well_array_sim.io.bc_pipelines import get_segment_by_permit_id, scenario_yaml_from_bc_segment
from well_array_sim.segment_study import run_segment_study


def test_bc_scenario_scales_inference_to_pipe_radius() -> None:
    segment = get_segment_by_permit_id(13)
    data = scenario_yaml_from_bc_segment(segment, max_length_m=10.0)
    inner_r = data["pipe"]["inner_radius_m"]
    assert data["pipe"]["outside_diameter_mm"] == 762.0
    assert data["inference"]["r_max_m"] > inner_r
    assert data["inference"]["r_min_m"] < inner_r
    assert data["corrosion"] is not None
    assert "transducer" in data
    assert data["scan"]["angle_step_deg"] == 1.0
    assert data["scan"]["z_step_m"] == 0.01


def test_segment_study_exports_partition_bundles_and_plots(tmp_path: Path) -> None:
    result = run_segment_study(
        13,
        out_root=tmp_path / "study",
        years=[0],
        max_length_m=1.2,
        partition_length_m=0.4,
        z_step_m=0.2,
        angle_step_deg=90.0,
        plot_waveforms=True,
    )
    assert result.scenario_path.exists()
    assert result.summary_path is not None
    assert result.partition_count == 3
    assert len(result.bundle_dirs) == 3
    bundle = result.bundle_dirs[0]
    assert (bundle / "manifest.json").exists()
    assert (bundle / "waveforms.parquet").exists()
    assert (bundle / "observation_grid.parquet").exists()
    assert any(path.name.endswith(".png") for path in result.plot_paths)

    with result.summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["permit_id"] == 13
    assert summary["years"] == [0]
    assert summary["sim_length_m"] == pytest.approx(1.2)
    assert summary["partition_count"] == 3
    assert summary["study_summary_schema_version"] == "1.0.0"
    assert summary["grid"]["z_step_m"] == pytest.approx(0.2)
    assert summary["grid"]["angle_step_deg"] == pytest.approx(90.0)
    assert summary["partition_length_m"] == pytest.approx(0.4)
    assert len(summary["bundles"]) == 3
    assert len(summary["year_results"]) == 1
    assert len(summary["year_results"][0]["partitions"]) == 3
