#!/usr/bin/env bash
# Export segment 1987 with localized middle-band corrosion and annual years 0–10.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/data/bundles/progression_1987}"
SCENARIO_DIR="$OUT/scenarios"
mkdir -p "$SCENARIO_DIR"

cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
SCENARIO="$SCENARIO_DIR/segment_1987.yaml"
well-array-sim bc scenario --permit-id 1987 --max-length-m 0.4 --out "$SCENARIO"

python3 <<PY
import yaml
from pathlib import Path

path = Path("$SCENARIO")
data = yaml.safe_load(path.read_text(encoding="utf-8"))
cor = data.setdefault("corrosion", {})
cor["V_corr_mm_per_yr"] = 0.01
cor["pit_lambda_per_m2_yr"] = 12.0
cor["snapshot_years"] = list(range(11))
cor["hotspot"] = {
    "z_center_m": 0.2,
    "z_half_width_m": 0.04,
    "pit_lambda_multiplier": 100.0,
}
path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
print("patched hotspot scenario:", path)
PY

well-array-sim export-partition \
  --scenario "$SCENARIO" \
  --segment-id 1987 \
  --all-partitions \
  --years 0,1,2,3,4,5,6,7,8,9,10 \
  --out-root "$OUT" \
  --z-step-m 0.10 \
  --angle-step-deg 10.0

echo "Progression bundles written to $OUT"
echo "Sync to platform: rsync -a $OUT/ ../acoustic-ndt-platform/data/bundles/progression_1987/"
