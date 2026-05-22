#!/usr/bin/env bash
# Regenerate PNGs under docs/images/ for the README gallery.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

OUT="$ROOT/docs/images"
WAVY="$ROOT/scenarios/internal_pipe_wavy_wall.yaml"
CORR="$ROOT/scenarios/internal_pipe_corrosion_default.yaml"
mkdir -p "$OUT"

echo "==> one shot (wavy wall @ 45°)"
well-array-sim sim --scenario "$WAVY" --angle-deg 45 --out "$OUT/one_shot"

echo "==> pipe sweep (wavy wall, 30° steps)"
well-array-sim sim --scenario "$WAVY" --axial-scan --angle-step-deg 30 --out "$OUT/_axial_sweep"
cp "$OUT/_axial_sweep_axial_radius_map.png" "$OUT/axial_radius_map.png"
cp "$OUT/_axial_sweep_axial_point_cloud.png" "$OUT/axial_point_cloud.png"
rm -f "$OUT/_axial_sweep"*

echo "==> corrosion thickness @ 10 yr"
well-array-sim sim --scenario "$CORR" --corrosion-snapshots --out "$OUT/_corrosion"
cp "$OUT/_corrosion_corrosion_10yr_map.png" "$OUT/corrosion_thickness_10yr.png"
rm -f "$OUT/_corrosion"*

echo "==> wrote $(find "$OUT" -maxdepth 1 -name '*.png' | wc -l) PNGs under docs/images/"
