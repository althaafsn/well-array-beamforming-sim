# Internal Pipe Ultrasonic Sim (v0.2)

Python simulator for **internal** fluid-filled pipe pulse-echo NDT: a rotating monostatic tool transmits ultrasound, receives wall echoes, and **blindly estimates inner radius** (angular SAFT migration across 360° sweeps). Optional corrosion evolves the true wall geometry over time. BC pipeline segment studies export 0.4 m partition bundles for **acoustic-ndt-platform** ingest.

**Release reference** (schemas, benchmarks, scope limits): [RELEASE.md](RELEASE.md)

---

## What this simulates

| Layer | What happens | Where it lives |
|-------|----------------|----------------|
| **Forward physics** | Ray-packet pulse-echo: TX pulse → fluid path → wall echo → RX waveform | `internal/ray_forward.py` |
| **Blind inference** | Estimate wall distance from RX alone (no access to true wall shape) | `internal/imaging.py`, `internal/saft_polar.py` |
| **Ground truth** | True inner wall geometry (uniform, wavy, or corrosion-evolved) | scenario YAML, `internal/corrosion/` |
| **Pipeline study** | Map a real BC pipeline segment → simulate a short window → export bundles | `bc run`, `export-partition` |

**Mental model:** imagine a tool sitting on the pipe centreline, rotating in azimuth while moving along the pipe axis. Each shot produces a waveform; inference stitches shots into a wall radius map. Corrosion changes the true wall between observation years — inference never reads that directly.

---

## Quick start

```bash
well-array-sim --help                              # list commands + examples
well-array-sim sim --angle-deg 45                  # one shot, ~seconds
well-array-sim-gui                                 # interactive explorer
```

Install first if needed — see [Install](#install) below.

---

## Install

```bash
cd well-array-beamforming-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

Requires **Python ≥ 3.10**. GUI needs system tkinter (`sudo apt install python3-tk` on Debian/Ubuntu).

Verify:

```bash
pytest -q          # 97 tests
well-array-sim --help
```

---

## Pick a workflow

| Goal | Command | Output folder | Typical runtime* |
|------|---------|---------------|------------------|
| **Try one ultrasound shot** | `well-array-sim sim --angle-deg 45` | `outputs/radial_demo_*` | seconds |
| **Smoke test** (~1 min) | See [Example 1](#example-1-smoke-test-1-minute) | `outputs/smoke/` | ~1 min |
| **BC pipeline segment study** (main batch workflow) | `well-array-sim bc run …` | `outputs/<name>/` | ~15 min (demo) to ~8 h (default 40 m) |
| **Interactive exploration** | `well-array-sim-gui` | `outputs/` on save | interactive |
| **Export bundles only** (no BC GeoJSON) | `well-array-sim export-partition …` | `--out-root` you choose | same per-partition cost as `bc run` |

\*Full-resolution defaults (1 cm × 1°) on the [benchmark laptop](#runtime--benchmarks) in [RELEASE.md](RELEASE.md). Use coarser `--z-step-m` / `--angle-step-deg` for faster iteration.

---

## Example 1: Smoke test (~1 minute)

Three 0.4 m partitions, one year, coarse grid — confirms install + export path:

```bash
well-array-sim bc run \
  --permit-id 13 \
  --years 0 \
  --max-length-m 1.2 \
  --max-partitions 3 \
  --z-step-m 0.20 \
  --angle-step-deg 90 \
  --no-plots \
  --out outputs/smoke
```

Check: `outputs/smoke/bundles/…/manifest.json` and `waveforms.parquet` exist; `outputs/smoke/study_summary.json` lists 3 bundles.

---

## Example 2: BC segment demo (~15 minutes)

Transmission segment 1987, 3 partitions × default corrosion years, sample PNGs for partition 0:

```bash
well-array-sim bc run \
  --permit-id 1987 \
  --max-length-m 40 \
  --max-partitions 3 \
  --out outputs/seg_1987_demo
```

Omit `--years` to use scenario template years `[0, 2, 5, 10]`. Pass `--years 0,5,10` to override.

---

## Example 3: Default local study (~8 hours)

First **40 m** of a BC segment at full resolution (100 partitions × 4 years = 400 bundles):

```bash
well-array-sim bc run \
  --permit-id 1987 \
  --max-length-m 40 \
  --out outputs/seg_1987
```

---

## BC pipeline segment study (`bc run`) — step by step

`bc` = **British Columbia** pipeline GIS data bundled in `data/raw/`. It is not a generic "beamforming" command — it maps real pipeline segments to pipe geometry and runs the UT simulator along a short centre-line window.

### 1. Browse BC pipeline data

Bundled GeoJSON: [`data/raw/bc_pipeline_segments.geojson`](data/raw/bc_pipeline_segments.geojson) (5,940 segments).

```bash
well-array-sim bc list --line-type Transmission --limit 10
well-array-sim bc show --permit-id 1987
well-array-sim bc categories
well-array-sim bc summary
```

`permit_id` = `OG_PIPELINE_SEGMENT_PERMIT_ID` from BC records.

### 2. Run study

```bash
well-array-sim bc run \
  --permit-id 1987 \
  --years 0,5,10 \
  --max-length-m 40 \
  --out outputs/seg_1987
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-length-m` | `40` | Simulate only the first N metres along the BC centre-line |
| `--partition-length-m` | `0.40` | Axial chunk size per bundle |
| `--max-partitions` | all | Cap partitions (useful for dev) |
| `--years` | from scenario `corrosion.snapshot_years` | Observation years to export |
| `--z-step-m` | `0.01` (1 cm) | Axial scan spacing |
| `--angle-step-deg` | `1.0` | Azimuth step (360° sweep) |
| `--max-plot-partitions` | `1` | PNG previews for first N partitions per year |
| `--no-plots` | off | Skip PNGs; bundles still include waveforms |

Build scenario YAML only (no simulation):

```bash
well-array-sim bc scenario --permit-id 1987 --max-length-m 40 --out outputs/seg_1987.yaml
```

### 3. Where output goes

All paths are under the `--out` directory you pass to `bc run`:

```
outputs/seg_1987/
├── segment_1987_scenario.yaml     # pipe + corrosion params from BC line type
├── study_summary.json             # run index (written when bc run finishes)
├── bundles/
│   └── segment_id=1987/
│       └── partition_id=1987@p0000/
│           └── year=0/
│               ├── manifest.json
│               ├── state_grid.parquet
│               ├── observation_grid.parquet
│               └── waveforms.parquet    ← always included
└── plots/                             # optional PNG previews
    └── year=0/partition=0000/
        ├── radius_map.png
        └── waveform_z0_theta45.png
```

- **`manifest.json` + 3 parquets** — platform-ready partition bundle (one per partition × year).
- **`study_summary.json`** — local run index only; not used by platform ingest. Appears **after the full run completes**.

Partition IDs: `p0000` = chainage 0.0–0.4 m, `p0001` = 0.4–0.8 m, etc.

---

## Export-only (`export-partition`)

For custom scenarios or batch jobs without BC GeoJSON:

```bash
# One partition, one year
well-array-sim export-partition \
  --scenario scenarios/internal_pipe_corrosion_default.yaml \
  --segment-id demo \
  --observation-year 0 \
  --out-root /tmp/bundles

# All partitions in scenario pipe.length_m, multiple years
well-array-sim export-partition \
  --scenario outputs/seg_1987.yaml \
  --segment-id 1987 \
  --all-partitions \
  --years 0,5,10 \
  --out-root /tmp/bundles
```

Same flags via: `well-array-sim-export` (no `export-partition` prefix).

Seed bundles for **acoustic-ndt-platform**:

```bash
./scripts/export_seed_bundles.sh
```

---

## Other entry points

| Command | Purpose |
|---------|---------|
| `well-array-sim sim --scenario … --angle-deg 45 --out outputs/demo` | One shot at one azimuth → PNG/NPZ |
| `well-array-sim sim --axial-scan --out outputs/axial_demo` | Full 360° tool sweep on one scenario |
| `well-array-sim-gui` | Desktop tkinter app (one shot / pipe sweep / corrosion views) |
| `well-array-sim om summary` | CER O&M CSV stats (metadata only — not UT simulation) |
| `python -m well_array_sim.om_gui` | O&M data browser |

Scenario YAML reference: [`scenarios/internal_pipe_default.yaml`](scenarios/internal_pipe_default.yaml). Corrosion demo: [`scenarios/internal_pipe_corrosion_default.yaml`](scenarios/internal_pipe_corrosion_default.yaml).

---

## Runtime & benchmarks

Measured on **HP OMEN Laptop 15-en1xxx** (AMD Ryzen 7 5800H, 8 cores / 16 threads, 61 GiB RAM, Linux, Python 3.14). See [RELEASE.md](RELEASE.md) for full table and scaling notes.

| Run | Bundles | Approx. time |
|-----|---------|--------------|
| Smoke test (Example 1) | 3 | ~1 min |
| Demo 3 partitions × 4 years | 12 | ~15 min |
| Default 40 m × 4 years (full grid) | 400 | ~8 h |
| Full 746 km segment × 1 year | ~1.87 M | not practical on one laptop |

One **0.4 m partition × 1 year** at full grid (41 z × 360 θ ≈ 14,760 shots): **~80 s** on the benchmark machine.

---

## Platform integration

Partition bundles follow schema v1 for ingest by **acoustic-ndt-platform** (MinIO + DuckDB). Manifest field reference: [RELEASE.md](RELEASE.md#partition-bundle-manifestjson).

---

## Project layout

```
well-array-beamforming-sim/
├── scenarios/           # YAML scenarios
├── data/raw/            # BC GeoJSON, CER O&M CSV
├── scripts/             # export_seed_bundles.sh, benchmarks
├── src/well_array_sim/
│   ├── cli.py           # well-array-sim, export-partition
│   ├── segment_study.py # bc run orchestrator
│   ├── export/          # partition bundle export
│   ├── internal/        # ray physics, matched filter, angular SAFT, corrosion
│   └── gui.py
├── tests/               # pytest (97 tests)
├── outputs/             # gitignored; your runs go here
├── README.md            # this file — how to use
└── RELEASE.md           # schemas, benchmarks, limits
```

---

## Inference modes

| `inference.mode` | Description |
|------------------|-------------|
| **`angular_saft`** (default) | After a full 360° sweep at each axial station, migrate all shot correlations into a polar focus image `I(r, φ)` and extract radius per direction. This is true angular synthetic-aperture SAFT. |
| **`matched_filter`** | Per-shot pulse-echo ranging (cross-correlate RX with TX). Used for single-angle CLI/GUI and optional axial parity tests. |

Scenario block example:

```yaml
inference:
  mode: angular_saft
  r_min_m: 0.07
  r_max_m: 0.14
  r_step_m: 0.0005
  angular_window_deg: 15.0   # angular aperture (FWHM) for stitching neighbouring shots
  coherent_sum: true
```

Single-angle CLI (`well-array-sim sim`) and GUI **Single** mode use **matched-filter** ranging regardless of YAML mode. Only **Axial** / export use `angular_saft` when configured.

---

## Tests

```bash
pytest -q
```

---

## Release v0.2.0

**Status:** ready to tag after `./scripts/verify_release.sh` passes (97 tests + smoke export).

Pre-tag verification (tests + smoke export + manifest checks):

```bash
chmod +x scripts/verify_release.sh
./scripts/verify_release.sh
```

Optional: regenerate local seed bundles for platform ingest:

```bash
./scripts/export_seed_bundles.sh
# copies to ../acoustic-ndt-platform/data/bundles/seed/ if needed
```

Tag when verification passes — full steps in [RELEASE.md](RELEASE.md#release-checklist-v020).

---

## Known limitations (v0.2)

2D ray slice (not full wave equation), concentric monostatic tool, synthetic corrosion rates, category-based pipe specs from BC line type (not row-level O&M). Full BC segment length is never simulated in one run — use `--max-length-m`. Angular SAFT stitches shots in angle only (not a multi-element linear transducer array yet). Details: [RELEASE.md](RELEASE.md#scope-limits).
