#!/usr/bin/env bash
# Pre-release verification: tests + smoke bc run + bundle artifact checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> pytest"
python -m pytest -q

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

python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("outputs/smoke/study_summary.json").read_text())
assert summary["study_summary_schema_version"] == "1.0.0"
assert summary["partition_count"] == 3
assert len(summary["bundles"]) == 3
print("study_summary.json OK")
PY

echo "==> release verification passed"
