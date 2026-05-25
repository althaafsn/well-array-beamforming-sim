"""Tests for multi-partition export helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from well_array_sim.export.bundle import export_partition_plan
from well_array_sim.export.partitions import partition_plan


SCENARIO = Path("scenarios/internal_pipe_corrosion_default.yaml")


def test_export_partition_plan_writes_all_bundles(tmp_path: Path) -> None:
    if not SCENARIO.exists():
        pytest.skip("corrosion scenario missing")

    partitions = partition_plan(1.2, partition_length_m=0.4)
    paths = export_partition_plan(
        scenario_path=SCENARIO,
        segment_id="plan_test",
        partitions=partitions,
        years=[0, 5],
        out_root=tmp_path,
        z_step_m=0.20,
        angle_step_deg=90.0,
    )
    assert len(paths) == len(partitions) * 2
    for path in paths:
        assert (path / "waveforms.parquet").exists()


def test_export_partition_plan_parallel_workers(tmp_path: Path) -> None:
    if not SCENARIO.exists():
        pytest.skip("corrosion scenario missing")

    partitions = partition_plan(1.2, partition_length_m=0.4)
    kwargs = {
        "scenario_path": SCENARIO,
        "segment_id": "plan_parallel",
        "partitions": partitions,
        "years": [0, 5],
        "out_root": tmp_path,
        "z_step_m": 0.20,
        "angle_step_deg": 90.0,
    }
    sequential = export_partition_plan(**kwargs, workers=1)
    parallel = export_partition_plan(
        scenario_path=SCENARIO,
        segment_id="plan_parallel_p2",
        partitions=partitions,
        years=[0, 5],
        out_root=tmp_path / "parallel",
        z_step_m=0.20,
        angle_step_deg=90.0,
        workers=2,
    )
    assert len(parallel) == len(sequential)
    for path in parallel:
        assert (path / "manifest.json").exists()
        assert (path / "waveforms.parquet").exists()
