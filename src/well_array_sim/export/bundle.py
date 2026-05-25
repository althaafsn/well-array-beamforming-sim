"""Orchestrate partition observation bundle export."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from well_array_sim.export.manifest import (
    build_degradation_inputs,
    build_manifest_dict,
    compute_summary_scalars,
    write_manifest,
)
from well_array_sim.export.observation_grid import build_observation_grid_table
from well_array_sim.export.partitions import PartitionSlice
from well_array_sim.export.schema import (
    DEFAULT_AXIAL_LENGTH_M,
    ArtifactNames,
    ExportConfig,
    GridSpec,
    PartitionIdentity,
)
from well_array_sim.export.state_grid import build_state_grid_table, pit_depth_grid_from_engine
from well_array_sim.export.waveforms import build_waveforms_table
from well_array_sim.internal import load_internal_scenario, simulate_axial_scan
from well_array_sim.internal.axial_scan import z_stations_m


def bundle_dir_for(out_root: Path, identity: PartitionIdentity) -> Path:
    return (
        out_root
        / f"segment_id={identity.segment_id}"
        / f"partition_id={identity.partition_id}"
        / f"year={identity.observation_year}"
    )


def export_partition_observation_bundle(
    *,
    scenario_path: Path | str,
    segment_id: str,
    observation_year: int,
    out_root: Path | str,
    partition_index: int = 0,
    chainage_start_m: float = 0.0,
    axial_length_m: float = DEFAULT_AXIAL_LENGTH_M,
    z_step_m: float | None = None,
    angle_step_deg: float | None = None,
    run_id: str = "",
) -> Path:
    scenario_path = Path(scenario_path).resolve()
    out_root = Path(out_root).resolve()
    scenario = load_internal_scenario(scenario_path)
    z_step_m = scenario.z_step_m if z_step_m is None else float(z_step_m)
    angle_step_deg = scenario.angle_step_deg if angle_step_deg is None else float(angle_step_deg)
    chainage_start_m = float(chainage_start_m)
    axial_length_m = float(axial_length_m)

    identity = PartitionIdentity(
        segment_id=str(segment_id),
        partition_index=int(partition_index),
        chainage_start_m=chainage_start_m,
        observation_year=int(observation_year),
        axial_length_m=axial_length_m,
    )

    z_stations_local = z_stations_m(axial_length_m, z_step_m=z_step_m, z_start_m=0.0)
    z_stations_abs = z_stations_local + chainage_start_m
    angles_deg = np.arange(0.0, 360.0, angle_step_deg, dtype=float)

    wall_profile = None
    pit_depth_grid = None
    if scenario.has_corrosion():
        engine = scenario.build_corrosion_engine()
        engine.run_to(float(observation_year))
        from well_array_sim.internal.corrosion.bridge import wall_profile_from_point_cloud

        wall_profile = wall_profile_from_point_cloud(engine.cloud, scenario.pipe)
        pit_depth_grid = pit_depth_grid_from_engine(
            engine.cloud,
            z_stations_local,
            angles_deg,
            chainage_start_m=chainage_start_m,
        )
    elif scenario.wall_profile is not None:
        wall_profile = scenario.wall_profile

    state_table = build_state_grid_table(
        wall_profile=wall_profile,
        pipe=scenario.pipe,
        z_stations_m=z_stations_local,
        angles_deg=angles_deg,
        pit_depth_m=pit_depth_grid,
        chainage_start_m=chainage_start_m,
    )

    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_stations_abs,
        angle_step_deg=angle_step_deg,
        z_step_m=z_step_m,
        wall_profile=wall_profile,
        echo=scenario.echo,
        inference=scenario.inference,
        store_waveforms=True,
    )
    obs_table = build_observation_grid_table(scan, z_local_m=z_stations_local)
    wf_table = build_waveforms_table(scan, z_local_m=z_stations_local)

    grid = GridSpec(
        n_z=len(z_stations_local),
        n_theta=len(angles_deg),
        z_step_m=z_step_m,
        theta_step_deg=angle_step_deg,
    )
    summaries = compute_summary_scalars(state_table)
    degradation_inputs = build_degradation_inputs(scenario, observation_year)
    export_cfg = ExportConfig(
        identity=identity,
        scenario_path=str(scenario_path),
        z_step_m=z_step_m,
        angle_step_deg=angle_step_deg,
        run_id=run_id,
        degradation_inputs=degradation_inputs,
    )
    artifacts = ArtifactNames()
    manifest = build_manifest_dict(
        identity=identity,
        export_cfg=export_cfg,
        scenario=scenario,
        grid=grid,
        summaries=summaries,
        artifacts=artifacts,
    )

    bundle_dir = bundle_dir_for(out_root, identity)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(state_table, bundle_dir / artifacts.state_grid)
    pq.write_table(obs_table, bundle_dir / artifacts.observation_grid)
    pq.write_table(wf_table, bundle_dir / artifacts.waveforms)
    write_manifest(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def _export_one_partition_task(task: dict) -> Path:
    """Picklable worker entry point for parallel export."""
    return export_partition_observation_bundle(**task)


def export_partition_years(
    *,
    scenario_path: Path | str,
    segment_id: str,
    years: list[int],
    out_root: Path | str,
    **kwargs,
) -> list[Path]:
    paths: list[Path] = []
    for year in years:
        paths.append(
            export_partition_observation_bundle(
                scenario_path=scenario_path,
                segment_id=segment_id,
                observation_year=year,
                out_root=out_root,
                **kwargs,
            )
        )
    return paths


def export_partition_plan(
    *,
    scenario_path: Path | str,
    segment_id: str,
    partitions: Sequence[PartitionSlice],
    years: Sequence[int],
    out_root: Path | str,
    z_step_m: float | None = None,
    angle_step_deg: float | None = None,
    workers: int = 1,
) -> list[Path]:
    """Export every partition in ``partitions`` for each observation year."""
    segment_id = str(segment_id)
    tasks: list[dict] = []
    for year in years:
        for partition in partitions:
            tasks.append(
                {
                    "scenario_path": scenario_path,
                    "segment_id": segment_id,
                    "observation_year": int(year),
                    "out_root": out_root,
                    "partition_index": partition.partition_index,
                    "chainage_start_m": partition.chainage_start_m,
                    "axial_length_m": partition.axial_length_m,
                    "z_step_m": z_step_m,
                    "angle_step_deg": angle_step_deg,
                    "run_id": f"{segment_id}@p{partition.partition_index:04d}_y{year}",
                }
            )

    if workers <= 1 or len(tasks) <= 1:
        return [_export_one_partition_task(task) for task in tasks]

    max_workers = min(int(workers), len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_export_one_partition_task, tasks))
