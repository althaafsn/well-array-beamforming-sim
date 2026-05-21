#!/usr/bin/env bash
# Export seed partition observation bundles (BC-faithful scenarios, coarse grid for size).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/data/bundles/seed}"
SCENARIO_DIR="$OUT/scenarios"
mkdir -p "$SCENARIO_DIR"

cd "$ROOT"
for PERMIT in 1987 4394; do
  SCENARIO="$SCENARIO_DIR/segment_${PERMIT}.yaml"
  well-array-sim bc scenario --permit-id "$PERMIT" --max-length-m 0.4 --out "$SCENARIO"
  well-array-sim export-partition \
    --scenario "$SCENARIO" \
    --segment-id "$PERMIT" \
    --all-partitions \
    --years 0,5,10 \
    --out-root "$OUT" \
    --z-step-m 0.10 \
    --angle-step-deg 10.0
done

echo "Seed bundles written to $OUT"
echo "Copy to acoustic-ndt-platform: cp -a $OUT/* ../acoustic-ndt-platform/data/bundles/seed/"
