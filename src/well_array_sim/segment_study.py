"""Run multi-year acoustic studies for BC pipeline segments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import yaml

from well_array_sim.export.bundle import export_partition_plan
from well_array_sim.export.partitions import partition_plan
from well_array_sim.export.schema import DEFAULT_AXIAL_LENGTH_M, STUDY_SUMMARY_SCHEMA_VERSION
from well_array_sim.internal import load_internal_scenario
from well_array_sim.io.bc_pipelines import (
    BcPipelineSegment,
    format_segment_detail,
    get_segment_by_permit_id,
    resolved_sim_length_m,
    scenario_yaml_from_bc_segment,
    write_scenario_from_bc_segment,
)


@dataclass
class SegmentStudyResult:
    permit_id: int
    segment_id: str
    bc_length_m: float
    sim_length_m: float
    scenario_path: Path
    out_root: Path
    years: list[int]
    partition_count: int = 0
    bundle_dirs: list[Path] = field(default_factory=list)
    plot_paths: list[Path] = field(default_factory=list)
    summary_path: Path | None = None


def default_years_from_scenario(scenario_path: Path) -> list[int]:
    with scenario_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    corrosion = data.get("corrosion") or {}
    years = corrosion.get("snapshot_years") or [0, 5, 10]
    return [int(round(float(year))) for year in years]


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - float(target))))


def plot_bundle_waveform(
    bundle_dir: Path,
    out_path: Path,
    *,
    z_m: float = 0.0,
    theta_deg: float = 45.0,
) -> Path:
    """Plot TX/RX waveforms for the nearest (z, θ) shot in a bundle."""
    wf_path = bundle_dir / "waveforms.parquet"
    if not wf_path.exists():
        raise FileNotFoundError(f"No waveforms.parquet in {bundle_dir}")
    table = pq.read_table(wf_path)
    z_vals = np.asarray(table["z_local_m"].to_pylist(), dtype=float)
    theta_vals = np.asarray(table["theta_deg"].to_pylist(), dtype=float)
    row_idx = _nearest_index(z_vals, z_m) if len(z_vals) else 0
    if len(theta_vals):
        same_z = np.where(np.isclose(z_vals, z_vals[row_idx]))[0]
        theta_subset = theta_vals[same_z]
        row_idx = int(same_z[_nearest_index(theta_subset, theta_deg)])

    sample_rate_hz = float(table["sample_rate_hz"][row_idx].as_py())
    t0_us = float(table["t0_us"][row_idx].as_py())
    n_samples = int(table["n_samples"][row_idx].as_py())
    p_tx = np.asarray(table["p_tx"][row_idx].as_py(), dtype=float)
    p_rx = np.asarray(table["p_rx"][row_idx].as_py(), dtype=float)
    z_shot = float(z_vals[row_idx])
    theta_shot = float(theta_vals[row_idx])
    time_us = t0_us + np.arange(n_samples) * (1e6 / sample_rate_hz if sample_rate_hz > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_us, p_tx, label="TX pulse", color="0.35", lw=1.2)
    ax.plot(time_us, p_rx, label="RX echo", color="tab:blue", lw=1.0)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Pressure (a.u.)")
    ax.set_title(f"Waveforms @ z={z_shot:.2f} m, θ={theta_shot:.0f}°")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_bundle_radius_map(bundle_dir: Path, out_path: Path) -> Path:
    """Heatmap of blind inferred inner radius from observation_grid.parquet."""
    obs_path = bundle_dir / "observation_grid.parquet"
    table = pq.read_table(obs_path)
    z_vals = np.unique(np.asarray(table["z_local_m"].to_pylist(), dtype=float))
    theta_vals = np.unique(np.asarray(table["theta_deg"].to_pylist(), dtype=float))
    radius_mm = np.asarray(table["inferred_inner_radius_m"].to_pylist(), dtype=float) * 1000.0
    grid = radius_mm.reshape(len(z_vals), len(theta_vals))

    extent = [
        float(theta_vals.min()),
        float(theta_vals.max() + (theta_vals[1] - theta_vals[0] if len(theta_vals) > 1 else 1.0)),
        float(z_vals.min()),
        float(z_vals.max() + (z_vals[1] - z_vals[0] if len(z_vals) > 1 else 0.1)),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax.set_xlabel("Steer angle θ (deg)")
    ax.set_ylabel("Axial position z (m)")
    ax.set_title("Blind inferred inner radius R(θ, z) [mm]")
    fig.colorbar(im, ax=ax, label="Radius (mm)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_segment_study(
    permit_id: int,
    *,
    geojson_path: Path | str | None = None,
    out_root: Path | str,
    years: list[int] | None = None,
    max_length_m: float = 40.0,
    partition_length_m: float = DEFAULT_AXIAL_LENGTH_M,
    max_partitions: int | None = None,
    z_step_m: float | None = None,
    angle_step_deg: float | None = None,
    plot_waveforms: bool = True,
    max_plot_partitions: int = 1,
    sample_z_m: float = 0.0,
    sample_theta_deg: float = 45.0,
) -> SegmentStudyResult:
    """
    Build a BC-derived scenario, simulate corrosion + acoustics over years, and export bundles.

    The full BC centre-line length is capped to ``max_length_m`` for runtime. Each partition
    covers ``partition_length_m`` metres (default 0.4 m) along the sim window. Every bundle
    includes waveforms for later inspection.
    """
    segment = get_segment_by_permit_id(permit_id, path=geojson_path)
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scenario_path = out_root / f"segment_{permit_id}_scenario.yaml"
    write_scenario_from_bc_segment(
        segment,
        scenario_path,
        max_length_m=max_length_m,
    )
    scenario = load_internal_scenario(scenario_path)
    resolved_z_step_m = scenario.z_step_m if z_step_m is None else float(z_step_m)
    resolved_angle_step_deg = scenario.angle_step_deg if angle_step_deg is None else float(angle_step_deg)
    year_list = years if years is not None else default_years_from_scenario(scenario_path)
    sim_length_m = resolved_sim_length_m(segment, max_length_m=max_length_m)
    segment_id = str(permit_id)
    partitions = partition_plan(
        sim_length_m,
        partition_length_m=partition_length_m,
        max_partitions=max_partitions,
    )

    bundle_dirs = export_partition_plan(
        scenario_path=scenario_path,
        segment_id=segment_id,
        partitions=partitions,
        years=year_list,
        out_root=out_root / "bundles",
        z_step_m=resolved_z_step_m,
        angle_step_deg=resolved_angle_step_deg,
    )

    plot_paths: list[Path] = []
    summaries: list[dict] = []
    bundle_idx = 0
    for year in year_list:
        year_summaries: list[dict] = []
        for partition in partitions:
            bundle_dir = bundle_dirs[bundle_idx]
            bundle_idx += 1

            with open(bundle_dir / "manifest.json", encoding="utf-8") as handle:
                manifest = json.load(handle)
            year_summaries.append(
                {
                    "partition_index": partition.partition_index,
                    "chainage_start_m": partition.chainage_start_m,
                    "axial_length_m": partition.axial_length_m,
                    "bundle_dir": str(bundle_dir),
                    "summary_scalars": manifest.get("summary_scalars", {}),
                }
            )

            if partition.partition_index < max(0, int(max_plot_partitions)):
                plots_dir = out_root / "plots" / f"year={year}" / f"partition={partition.partition_index:04d}"
                plot_paths.append(plot_bundle_radius_map(bundle_dir, plots_dir / "radius_map.png"))
                if plot_waveforms:
                    plot_paths.append(
                        plot_bundle_waveform(
                            bundle_dir,
                            plots_dir / f"waveform_z{sample_z_m:g}_theta{sample_theta_deg:g}.png",
                            z_m=sample_z_m,
                            theta_deg=sample_theta_deg,
                        )
                    )

        summaries.append(
            {
                "observation_year": int(year),
                "partition_count": len(partitions),
                "partitions": year_summaries,
            }
        )

    summary = {
        "study_summary_schema_version": STUDY_SUMMARY_SCHEMA_VERSION,
        "permit_id": permit_id,
        "segment_id": segment_id,
        "bc_length_m": segment.length_m,
        "sim_length_m": sim_length_m,
        "max_length_m": float(max_length_m),
        "partition_length_m": float(partition_length_m),
        "partition_count": len(partitions),
        "years": year_list,
        "scenario_path": str(scenario_path),
        "grid": {
            "z_step_m": resolved_z_step_m,
            "angle_step_deg": resolved_angle_step_deg,
        },
        "bc_source": scenario_yaml_from_bc_segment(segment, max_length_m=max_length_m)["bc_source"],
        "year_results": summaries,
        "plots": [str(path) for path in plot_paths],
        "bundles": [str(path) for path in bundle_dirs],
    }
    summary_path = out_root / "study_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return SegmentStudyResult(
        permit_id=permit_id,
        segment_id=segment_id,
        bc_length_m=segment.length_m,
        sim_length_m=sim_length_m,
        scenario_path=scenario_path,
        out_root=out_root,
        years=year_list,
        partition_count=len(partitions),
        bundle_dirs=bundle_dirs,
        plot_paths=plot_paths,
        summary_path=summary_path,
    )


def print_study_report(result: SegmentStudyResult, segment: BcPipelineSegment, *, max_length_m: float) -> None:
    print(format_segment_detail(segment, max_length_m=max_length_m))
    print("")
    print(f"Scenario:    {result.scenario_path}")
    print(f"Summary:     {result.summary_path}")
    print(f"Years:       {', '.join(str(year) for year in result.years)}")
    print(f"Partitions:  {result.partition_count} per year ({len(result.bundle_dirs)} bundles total)")
    print(f"Bundles:     {result.out_root / 'bundles'}")
    print(f"Plots:       {len(result.plot_paths)} under {result.out_root / 'plots'}")
    for path in result.plot_paths:
        print(f"  - {path}")
