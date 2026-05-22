# Changelog

All notable changes to **well-array-beamforming-sim** are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-05-22

### Added

- **Angular SAFT migration** (`internal/saft_polar.py`): after a full 360° sweep at each axial station, weighted correlation stacking on the lag axis builds a polar focus image `I(r, φ)` and extracts blind radius per direction
- `inference.mode: angular_saft` (default) with `angular_window_deg` and `coherent_sum` config keys
- Legacy `inference.mode: matched_filter` for per-shot ranging (no angular combine)
- Manifest fields `inference_mode` (`angular_saft` | `matched_filter`) and `physics_mode` (`ray_packet_pulse_echo`)
- CLI workflow guide (`well-array-sim` with no args lists commands + examples)
- 10 new tests (`test_saft_polar.py`, `test_cli_help.py`); **97 total**

### Changed

- Package version **0.2.0**; `sim_engine_version` in export manifests
- Removed misleading `inference.mode: saft` YAML alias (raises migration error)
- Renamed `saft_range_profile` → `matched_filter_range_profile`; plot/view/CLI output labels updated
- `simulate_axial_scan` defers inference to batch angular SAFT when `mode: angular_saft`
- `simulate_pulse_echo_2d` accepts `run_inference=False` to skip per-shot correlate during axial sweeps
- BC scenario builder and default YAML scenarios default to `angular_saft`
- Single-angle CLI / GUI **One shot** mode still use per-shot matched-filter inference
- GUI modes relabeled (One shot / Pipe sweep / Corrosion ground truth) with inline hints

### Documentation

- README: “What this simulates” table, quick start, BC GIS clarified
- RELEASE / verify script assert v0.2 manifest fields

### Fixed

- Angular SAFT peak picking on lag axis (removed ~1 mm radius bias from early polar migration prototype)

## [0.1.0] - 2026-05-21

### Added

- 2D ray-packet forward model with per-shot matched-filter range inference (misnamed “SAFT” in v0.1; see v0.2 angular SAFT)
- Corrosion engine (uniform loss + pitting) driving wall geometry over time
- BC pipeline GeoJSON integration (5,940 segments): `bc list`, `bc show`, `bc scenario`, `bc run`
- Multi-year segment studies with **0.4 m partition export** and mandatory `waveforms.parquet`
- Partition observation bundle schema v1 (`manifest.json` + 3 parquets) for **acoustic-ndt-platform** ingest
- CLI entry points: `well-array-sim`, `well-array-sim-export`, `well-array-sim-gui`, `well-array-sim-om`
- `export-partition --all-partitions` for batch export from scenario YAML
- Desktop GUI (`PipeSimGui`) for single-angle, axial, and corrosion views
- CER O&M CSV browser and summary CLI
- 87 automated tests including CLI smoke and export schema checks

### Changed

- Scan defaults: **1 cm axial** (`z_step_m: 0.01`), **1° azimuth** (`angle_step_deg: 1.0`)
- Waveforms are always exported; removed `--no-waveforms`
- Removed legacy 16-element beamforming / Rust paths

### Documentation

- [README.md](README.md) — install, examples, output layout, runtime guide
- [RELEASE.md](RELEASE.md) — v0.1 schemas, benchmark machine (OMEN Ryzen 5800H), runtime tables
- [scripts/verify_release.sh](scripts/verify_release.sh) — pre-tag verification

### Known limitations (v0.1)

- Sim window capped (`--max-length-m`, default 40 m); not full BC segment length
- Single-threaded export (~80 s per full-resolution partition on benchmark laptop)
- 2D ray slice; category-based pipe specs from BC line type

[0.2.0]: RELEASE.md
[0.1.0]: RELEASE.md
