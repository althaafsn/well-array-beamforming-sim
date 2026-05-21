# Changelog

All notable changes to **well-array-beamforming-sim** are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-21

### Added

- 2D ray-packet forward model with blind SAFT range inference (`simulate_pulse_echo_2d`, `simulate_axial_scan`)
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

[0.1.0]: RELEASE.md
