# Raw datasets

## BC pipeline segment permits (BCGW)

| Path | Description |
|------|-------------|
| `bc_pipeline_segments.geojson` | 5,940 BC pipeline centre-lines (canonical ingest name) |
| `BCGW_02001F02_1779126151284_10220/` | Original BC Data Catalogue download (GeoJSON + metadata) |

Source: [BC Oil and Gas Pipeline Segment Permits](https://catalogue.data.gov.bc.ca/dataset/ecf567ea-4901-4f51-a5b0-35959ca96c47)

Used by `acoustic-ndt-platform` ingest (`bronze.bronze_bc_pipelines`) and `well-array-sim bc …` commands.

Pipe geometry defaults (OD, wall, SMYS, MOP) are inferred from BC `LINE_TYPE_DESC`:

| Category | BC line types | Assumed OD | Wall |
|----------|---------------|------------|------|
| Gathering / Flowline | Gathering, Flow | 168 mm (6") | 4.8 mm |
| Mid-Size Feeder | Intermediate, Fuel Gas, Distribution, … | 324 mm (12") | 7.9 mm |
| Major Transmission | Transmission | 762 mm (30") | 13 mm |

```bash
well-array-sim bc categories
well-array-sim bc summary
well-array-sim bc list --line-type Transmission --limit 10
well-array-sim bc show --permit-id 1987
well-array-sim bc scenario --permit-id 13 --out scenarios/bc_13.yaml
well-array-sim bc run --permit-id 13 --years 0 --max-length-m 1.2 --max-partitions 3 \
  --z-step-m 0.20 --angle-step-deg 90 --no-plots --out outputs/smoke
```

See [README.md](../../README.md) for full usage and [RELEASE.md](../../RELEASE.md) for runtime benchmarks.

## CER Operation & Maintenance Activity

| File | Source |
|------|--------|
| `operation-and-maintenance-activity.csv` | Canada Energy Regulator (CER) O&M event notifications |
| `operation-and-maintenance-activity-data-dictionary.csv` | Column definitions |

```bash
well-array-sim om summary
python -m well_array_sim.om_gui
```

Only ~7% of O&M rows include pipeline outside diameter — useful for analysis, not required for BC scenario creation.
