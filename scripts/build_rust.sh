#!/usr/bin/env bash
# Build and install the optional Rust extension into the active venv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install -q "maturin>=1.4"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
maturin develop --release --manifest-path rust/well_array_sim_core/Cargo.toml

python - <<'PY'
import well_array_sim_core
assert well_array_sim_core.extension_available()
print("well_array_sim_core OK")
PY
