"""CLI smoke tests for partition export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "internal_pipe_corrosion_default.yaml"


@pytest.mark.parametrize("entrypoint", ["module", "script"])
def test_export_partition_cli_smoke(tmp_path: Path, entrypoint: str) -> None:
    if not SCENARIO.exists():
        pytest.skip("corrosion scenario missing")

    out_root = tmp_path / "bundles"
    common = [
        "--scenario",
        str(SCENARIO),
        "--segment-id",
        "cli_smoke",
        "--observation-year",
        "0",
        "--out-root",
        str(out_root),
        "--z-step-m",
        "0.20",
        "--angle-step-deg",
        "90.0",
    ]
    if entrypoint == "module":
        cmd = [sys.executable, "-m", "well_array_sim.cli", "export-partition", *common]
    else:
        cmd = ["well-array-sim-export", *common]

    subprocess.run(cmd, check=True, cwd=str(ROOT))
    bundle = out_root / "segment_id=cli_smoke" / "partition_id=cli_smoke@p0000" / "year=0"
    assert (bundle / "manifest.json").exists()
    assert (bundle / "waveforms.parquet").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert "waveforms" in manifest["artifacts"]


def test_export_all_partitions_cli(tmp_path: Path) -> None:
    if not SCENARIO.exists():
        pytest.skip("corrosion scenario missing")

    out_root = tmp_path / "bundles"
    cmd = [
        sys.executable,
        "-m",
        "well_array_sim.cli",
        "export-partition",
        "--scenario",
        str(SCENARIO),
        "--segment-id",
        "cli_all",
        "--all-partitions",
        "--years",
        "0",
        "--out-root",
        str(out_root),
        "--z-step-m",
        "0.20",
        "--angle-step-deg",
        "90.0",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    assert (out_root / "segment_id=cli_all" / "partition_id=cli_all@p0000" / "year=0").exists()
