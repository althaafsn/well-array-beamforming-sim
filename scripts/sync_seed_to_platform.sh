#!/usr/bin/env bash
# Export v0.2 seed bundles from this repo and copy to acoustic-ndt-platform.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLATFORM="${1:-$ROOT/../acoustic-ndt-platform}"
SEED_OUT="$ROOT/data/bundles/seed"
PLATFORM_SEED="$PLATFORM/data/bundles/seed"

if [[ ! -d "$PLATFORM" ]]; then
  echo "Platform repo not found: $PLATFORM" >&2
  exit 1
fi

cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> export seed bundles (sim v0.2)"
"$ROOT/scripts/export_seed_bundles.sh" "$SEED_OUT"

echo "==> verify v0.2 manifest"
python - <<'PY'
import json
from pathlib import Path

root = Path("data/bundles/seed")
manifest_path = root / "segment_id=1987/partition_id=1987@p0000/year=0/manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["sim_engine_version"] == "0.2.0", manifest.get("sim_engine_version")
assert manifest["inference_mode"] == "angular_saft"
assert manifest["physics_mode"] == "ray_packet_pulse_echo"
assert manifest["pipe_nominal"]["outer_diameter_mm"] == 762.0
bundle_dir = manifest_path.parent
for name in ("state_grid.parquet", "observation_grid.parquet", "waveforms.parquet"):
    assert (bundle_dir / name).exists(), name
print("manifest OK (v0.2.0, angular_saft, waveforms present)")
PY

echo "==> sync to platform seed"
mkdir -p "$PLATFORM_SEED"
if ! cp -a "$SEED_OUT/"* "$PLATFORM_SEED/" 2>/dev/null; then
  echo "cp failed (often Docker/airflow UID 50000 on data/bundles). Retrying via docker..." >&2
  docker run --rm -v "$ROOT:/sim" -v "$PLATFORM:/platform" alpine sh -c "
    chown -R $(id -u):$(id -g) /platform/data/bundles 2>/dev/null || true
    rm -rf /platform/data/bundles/staging
    mkdir -p /platform/data/bundles/seed
    cp -a /sim/data/bundles/seed/* /platform/data/bundles/seed/
  "
fi
rm -rf "$PLATFORM/data/bundles/staging" 2>/dev/null || true

echo "Seed synced to $PLATFORM_SEED"
echo "Ingest on platform:"
echo "  cd $PLATFORM"
echo "  python -m acoustic_ndt.ingest.observation_bundle --bundle-root data/bundles/seed --skip-minio"
