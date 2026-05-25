#!/usr/bin/env bash
# Pre-release verification: tests + smoke bc run + bundle artifact checks.
# Optional: ./scripts/verify_release.sh --with-rust
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WITH_RUST=0
if [[ "${1:-}" == "--with-rust" ]]; then
  WITH_RUST=1
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ "$WITH_RUST" == "1" ]]; then
  echo "==> build Rust extension"
  ./scripts/build_rust.sh
  export WELL_ARRAY_SIM_USE_RUST=1
fi

echo "==> pytest"
python -m pytest -q
if [[ "$WITH_RUST" == "1" ]]; then
  python -m pytest tests/internal/test_rust_parity.py -q
fi

SMOKE="$ROOT/outputs/smoke"
rm -rf "$SMOKE"

echo "==> smoke bc run"
well-array-sim bc run \
  --permit-id 13 \
  --years 0 \
  --max-length-m 1.2 \
  --max-partitions 3 \
  --z-step-m 0.20 \
  --angle-step-deg 90 \
  --no-plots \
  --out "$SMOKE"

SUMMARY="$SMOKE/study_summary.json"
BUNDLE="$SMOKE/bundles/segment_id=13/partition_id=13@p0000/year=0"

test -f "$SUMMARY" || { echo "missing $SUMMARY"; exit 1; }
test -f "$BUNDLE/manifest.json" || { echo "missing manifest"; exit 1; }
test -f "$BUNDLE/waveforms.parquet" || { echo "missing waveforms"; exit 1; }
test -f "$BUNDLE/state_grid.parquet" || { echo "missing state_grid"; exit 1; }
test -f "$BUNDLE/observation_grid.parquet" || { echo "missing observation_grid"; exit 1; }

if [[ -d "$SMOKE/plots" ]] && [[ -n "$(find "$SMOKE/plots" -name '*.png' 2>/dev/null)" ]]; then
  echo "unexpected PNG plots with --no-plots"
  exit 1
fi

echo "==> package + CLI smoke"
python - <<'PY'
import re
import subprocess
import sys
from pathlib import Path

root = Path(".")
pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
assert match is not None
version = match.group(1)
assert version == "0.2.0", version

from well_array_sim.export.schema import SIM_ENGINE_VERSION, EXPORT_SCHEMA_VERSION

assert SIM_ENGINE_VERSION == "0.2.0"
assert EXPORT_SCHEMA_VERSION == "1.0.0"

help_out = subprocess.run(
    [sys.executable, "-m", "well_array_sim.cli"],
    capture_output=True,
    text=True,
    check=True,
).stdout
for token in ("sim", "bc", "export-partition", "What this simulates"):
    assert token in help_out, token
print(f"pyproject.toml version OK ({version})")
print("CLI workflow guide OK")
PY

python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("outputs/smoke/study_summary.json").read_text())
assert summary["study_summary_schema_version"] == "1.0.0"
assert summary["partition_count"] == 3
assert len(summary["bundles"]) == 3
manifest = json.loads(Path("outputs/smoke/bundles/segment_id=13/partition_id=13@p0000/year=0/manifest.json").read_text())
assert manifest["sim_engine_version"] == "0.2.0"
assert manifest["inference_mode"] == "angular_saft"
assert manifest["physics_mode"] == "ray_packet_pulse_echo"
assert manifest["export_schema_version"] == "1.0.0"
print("study_summary.json OK")
print("manifest.json OK (v0.2.0, angular_saft, ray_packet_pulse_echo)")
PY

echo "==> release verification passed"
if [[ "$WITH_RUST" == "1" ]]; then
  echo "==> rust backend enabled for smoke export"
fi
