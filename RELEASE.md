# Release v0.1.0

**well-array-beamforming-sim** v0.1 — research prototype for internal pipe ultrasonic pulse-echo simulation (2D ray-packet forward model + blind SAFT) with BC pipeline segment studies and partition export for **acoustic-ndt-platform** ingest.

**How to use:** [README.md](README.md)

---

## What is included

| Feature | Entry point |
|---------|-------------|
| Single-angle / axial ray + SAFT | `well-array-sim sim`, `well-array-sim-gui` |
| Corrosion time evolution | scenarios with `corrosion:` block |
| BC segment browse / scenario build | `well-array-sim bc list|show|scenario` |
| Multi-year partition study + plots | `well-array-sim bc run` |
| Platform bundle export | `well-array-sim export-partition`, `well-array-sim-export` |
| CER O&M CSV browse | `well-array-sim om …`, `well-array-sim-om` |

**Not included:** full-wave 3D propagation, eccentered tools, ops-driven corrosion, integrity alerting, distributed batch orchestration.

---

## Benchmark environment

All runtime figures below were measured on:

| | |
|---|---|
| **Machine** | HP OMEN Laptop 15-en1xxx (`althaaf-OMEN-Laptop-15-en1xxx`) |
| **CPU** | AMD Ryzen 7 5800H (8 cores / 16 threads) |
| **RAM** | 61 GiB |
| **OS** | Linux 7.0.0-15-generic x86_64 |
| **Python** | 3.14.4 (project requires ≥ 3.10) |
| **Parallelism** | Single-threaded Python (no multiprocessing yet) |

Re-run a one-partition benchmark locally:

```bash
well-array-sim export-partition \
  --scenario scenarios/internal_pipe_corrosion_default.yaml \
  --segment-id bench \
  --observation-year 0 \
  --out-root /tmp/bench
# expect ~80 s at default 1 cm × 1° grid
```

---

## Runtime reference

**Per bundle cost** = one `(partition, year)` export. Each 0.4 m partition at default grid runs **41 axial × 360 azimuth ≈ 14,760** ray+SAFT shots and always writes waveforms.

| Grid | Shots / partition | Time / bundle |
|------|-------------------|---------------|
| **Default** `z_step_m=0.01`, `angle_step_deg=1` | ~14,760 | **~80 s** |
| Coarse `z_step_m=0.20`, `angle_step_deg=90` | ~12 | **~0.1 s** |

**Common `bc run` totals** (default grid):

| Command profile | Partitions | Years | Bundles | Approx. time |
|-----------------|------------|-------|---------|--------------|
| Smoke ([README Example 1](README.md#example-1-smoke-test-1-minute)) | 3 | 1 | 3 | ~1 min |
| Demo `--max-partitions 3`, template years | 3 | 4 | 12 | ~15 min |
| Default `--max-length-m 40`, template years | 100 | 4 | 400 | ~8 h |
| `--max-length-m 40 --years 0,5,10` | 100 | 3 | 300 | ~6.5 h |

**Full BC segment** (e.g. permit 1987 ≈ 746 km → ~1.87 M partitions/year) is **not** intended for a single laptop run. Use `--max-length-m`, `--max-partitions`, or coarser scan flags for development.

**BC dataset scale** (for planning):

| Statistic | Length | Partitions @ 0.4 m | 4-year bundles @ ~80 s |
|-----------|--------|--------------------|-------------------------|
| Median segment | ~1.35 km | ~3,400 | ~12 days |
| P90 segment | ~6.8 km | ~17,100 | ~59 days |
| Segment 1987 | ~746 km | ~1.87 M | ~17+ years |

---

## Output files

### From `bc run` (`--out <dir>/`)

| File / folder | Created when | Used by |
|---------------|--------------|---------|
| `segment_<id>_scenario.yaml` | Start of run | Reproducibility; input to `export-partition` |
| `bundles/…/manifest.json` + 3 parquets | Each partition × year | **Platform ingest**, local analysis |
| `study_summary.json` | **End** of full `bc run` | Local index only |
| `plots/…/*.png` | During run (unless `--no-plots`) | Human preview |

### From `export-partition` only

Creates `bundles/…` only (no `study_summary.json`, no plots).

---

## Partition bundle (`manifest.json`)

One manifest per bundle directory. **Required for platform ingest.**

| Field | Value / purpose |
|-------|-----------------|
| `export_schema_version` | `"1.0.0"` |
| `bundle_type` | `"pipe_partition_observation"` |
| `segment_id`, `partition_id`, `partition_index` | Identity (`partition_id` = `{segment_id}@p{index:04d}`) |
| `chainage_start_m`, `chainage_end_m`, `axial_length_m` | Axial window along sim pipe (local z is 0…0.4 m inside partition) |
| `observation_year` | Corrosion snapshot year |
| `grid` | `{n_z, n_theta, z_step_m, theta_step_deg}` |
| `artifacts` | Filenames: `state_grid.parquet`, `observation_grid.parquet`, `waveforms.parquet` |
| `summary_scalars` | Quick thickness/radius stats for catalog |

**Also written** (provenance): `run_id`, `sim_engine`, `sim_engine_version`, `physics_mode`, `scenario_ref`, `scenario_params_hash`, `pipe_nominal`, `degradation_inputs`.

### Parquet columns

Defined in `src/well_array_sim/export/schema.py`:

| File | Columns | Content |
|------|---------|---------|
| `state_grid.parquet` | 11 | Ground-truth wall geometry per `(z_local, θ)` |
| `observation_grid.parquet` | 6 | Blind inference only (no GT columns) |
| `waveforms.parquet` | 7 | TX/RX pressure traces per shot |

**Waveforms are mandatory** — every bundle includes `waveforms.parquet`. The old `--no-waveforms` flag was removed in v0.1.

---

## Study summary (`study_summary.json`)

Written only by **`bc run`**, at the end of a successful run. **Not ingested by the platform.**

| Field | Purpose |
|-------|---------|
| `study_summary_schema_version` | `"1.0.0"` |
| `permit_id`, `segment_id`, `bc_length_m`, `sim_length_m`, `max_length_m` | BC segment + sim window |
| `partition_length_m`, `partition_count`, `years` | Partition plan |
| `scenario_path`, `grid`, `bc_source` | Provenance |
| `year_results[]` | Per-year list of partitions with `bundle_dir` + `summary_scalars` |
| `bundles[]`, `plots[]` | Flat path lists for convenience |

---

## Defaults

| Setting | Default | Notes |
|---------|---------|-------|
| Scan `z_step_m` | `0.01` m (1 cm) | BC scenarios and templates |
| Scan `angle_step_deg` | `1.0`° | Full 360° sweep |
| Partition length | `0.40` m | `--partition-length-m` |
| `bc run` sim window | `40` m | `--max-length-m` |
| Corrosion snapshot years | `[0, 2, 5, 10]` | Override with `bc run --years` |

Pipe OD/wall/SMYS for BC segments come from **line type category** (Transmission → 762 mm OD, 13 mm wall), not individual GeoJSON row fields.

---

## Scope limits (v0.1)

- **2D slice**, single ray — not full wave equation or 3D bore propagation
- **Sim window cap** — never simulates an entire multi-hundred-km BC segment in one command
- **Category-based pipe specs** — not row-level CER O&M diameter fields
- **Synthetic corrosion** — fixed YAML rates; not De Waard-from-ops
- **No integrity alerts** in export — analysis belongs in the platform
- **Single-threaded** export loop

---

## Requirements & tests

- Python ≥ 3.10
- Dependencies: `requirements.txt` / `pyproject.toml`

```bash
pip install -e ".[dev]"
pytest -q    # 87 tests
```

Test coverage: ray forward, blind SAFT, corrosion, partition export, BC pipelines, segment study, CLI smoke tests.

---

## Release checklist (v0.1.0)

```bash
pip install -e ".[dev]"
./scripts/verify_release.sh          # pytest + smoke bc run
./scripts/export_seed_bundles.sh     # optional: data/bundles/seed/
git tag -a v0.1.0 -m "v0.1.0 prototype release"
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.

---

## Console scripts

| Script | Alias |
|--------|-------|
| `well-array-sim` | Main CLI (`bc`, `sim`, `om`, `export-partition`) |
| `well-array-sim-export` | Partition export only |
| `well-array-sim-gui` | Desktop simulator GUI |
| `well-array-sim-om` | O&M data GUI |
