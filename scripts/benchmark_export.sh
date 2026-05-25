#!/usr/bin/env bash
# Time one partition export (Python vs optional Rust backend).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

SCENARIO="${SCENARIO:-scenarios/internal_pipe_corrosion_default.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/benchmark}"
SEGMENT="${SEGMENT:-bench-export}"
YEAR="${YEAR:-0}"

run_export() {
  local label="$1"
  local use_rust="$2"
  local workers="$3"
  local out="${OUT_ROOT}/${label}"
  rm -rf "$out"
  if [[ "$use_rust" == "1" ]]; then
    export WELL_ARRAY_SIM_USE_RUST=1
  else
    unset WELL_ARRAY_SIM_USE_RUST
  fi
  local start end elapsed
  start=$(date +%s.%N)
  well-array-sim export-partition \
    --scenario "$SCENARIO" \
    --segment-id "${SEGMENT}-${label}" \
    --observation-year "$YEAR" \
    --out-root "$out" \
    --z-step-m 0.01 \
    --angle-step-deg 1.0 \
    --workers "$workers" >/dev/null
  end=$(date +%s.%N)
  elapsed=$(python - <<PY
start = float("$start")
end = float("$end")
print(f"{end - start:.2f}")
PY
)
  echo "${label}_seconds=${elapsed}"
}

run_plan_export() {
  local label="$1"
  local use_rust="$2"
  local workers="$3"
  local out="${OUT_ROOT}/${label}"
  rm -rf "$out"
  if [[ "$use_rust" == "1" ]]; then
    export WELL_ARRAY_SIM_USE_RUST=1
  else
    unset WELL_ARRAY_SIM_USE_RUST
  fi
  local start end elapsed
  start=$(date +%s.%N)
  well-array-sim export-partition \
    --scenario "$SCENARIO" \
    --segment-id "${SEGMENT}-${label}" \
    --all-partitions \
    --max-partitions 4 \
    --years 0 \
    --out-root "$out" \
    --z-step-m 0.20 \
    --angle-step-deg 90.0 \
    --workers "$workers" >/dev/null
  end=$(date +%s.%N)
  elapsed=$(python - <<PY
start = float("$start")
end = float("$end")
print(f"{end - start:.2f}")
PY
)
  echo "${label}_seconds=${elapsed}"
}

mkdir -p "$OUT_ROOT"
echo "==> export benchmark (scenario=$SCENARIO, full 41x360 grid)"
echo "    writing to ${OUT_ROOT}/"
run_export python 0 1
if python - <<'PY'
from well_array_sim.internal._rust_backend import rust_available
raise SystemExit(0 if rust_available() else 1)
PY
then
  run_export rust 1 1
else
  echo "rust_seconds=skipped (run ./scripts/build_rust.sh first)"
fi

BENCH_WORKERS="${BENCH_WORKERS:-}"
if [[ -n "$BENCH_WORKERS" ]]; then
  echo "==> parallel plan benchmark (4 partitions, coarse grid, workers=$BENCH_WORKERS)"
  if python - <<'PY'
from well_array_sim.internal._rust_backend import rust_available
raise SystemExit(0 if rust_available() else 1)
PY
  then
    run_plan_export rust_workers1 1 1
    run_plan_export "rust_workers${BENCH_WORKERS}" 1 "$BENCH_WORKERS"
  else
    echo "rust_workers_seconds=skipped (run ./scripts/build_rust.sh first)"
  fi
fi
