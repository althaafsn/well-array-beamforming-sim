"""CLI discoverability: root help and legacy flag routing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "well_array_sim.cli", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_root_help_lists_workflows() -> None:
    result = _run_cli()
    assert result.returncode == 0
    for token in ("sim", "bc", "export-partition", "What this simulates", "Examples"):
        assert token in result.stdout


def test_legacy_bare_sim_flags_still_work() -> None:
    result = _run_cli(
        "sim",
        "--scenario",
        "scenarios/internal_pipe_default.yaml",
        "--angle-deg",
        "45",
        "--out",
        "/tmp/well_array_cli_help_smoke",
    )
    assert result.returncode == 0
    assert "Range profile:" in result.stdout
