"""manifest.json read/write for partition observation bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from well_array_sim.export.schema import (
    EXPORT_SCHEMA_VERSION,
    BUNDLE_TYPE,
    ArtifactNames,
    ExportConfig,
    GridSpec,
    PartitionIdentity,
    SummaryScalars,
)
from well_array_sim.internal.scenario import InternalScenario


def scenario_params_hash(scenario: InternalScenario) -> str:
    payload = json.dumps(scenario.raw, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_degradation_inputs(scenario: InternalScenario, observation_year: int) -> dict[str, Any]:
    out: dict[str, Any] = {"observation_year": observation_year}
    if scenario.corrosion is None:
        out["corrosion_enabled"] = False
        return out
    cfg = scenario.corrosion
    out["corrosion_enabled"] = True
    out["V_corr_mm_per_yr"] = cfg.v_corr_mm_per_yr
    out["pit_lambda_per_m2_yr"] = cfg.pit_lambda_per_m2_yr
    out["pit_alpha"] = cfg.pit_alpha
    out["snapshot_years"] = list(cfg.snapshot_years)
    return out


def compute_summary_scalars(state_table: pa.Table) -> SummaryScalars:
    t_rem_mm = pc.multiply(state_table["t_remaining_m"], 1000.0)
    metal_loss_mm = pc.multiply(state_table["metal_loss_m"], 1000.0)
    inner_r_mm = pc.multiply(state_table["inner_radius_m"], 1000.0)
    return SummaryScalars(
        t_remaining_mm_min=float(pc.min(t_rem_mm).as_py()),
        t_remaining_mm_mean=float(pc.mean(t_rem_mm).as_py()),
        metal_loss_mm_max=float(pc.max(metal_loss_mm).as_py()),
        inner_radius_mm_mean=float(pc.mean(inner_r_mm).as_py()),
    )


def build_manifest_dict(
    *,
    identity: PartitionIdentity,
    export_cfg: ExportConfig,
    scenario: InternalScenario,
    grid: GridSpec,
    summaries: SummaryScalars,
    artifacts: ArtifactNames,
) -> dict[str, Any]:
    pipe_nominal = {
        "outer_diameter_mm": float(scenario.pipe.outer_radius_m * 2 * 1000),
        "t_nominal_mm": float(scenario.pipe.wall_thickness_m * 1000),
        "fluid_vp_mps": float(scenario.fluid.vp),
        "inner_radius_mm": float(scenario.pipe.inner_radius_m * 1000),
    }
    artifact_map = {
        "state_grid": artifacts.state_grid,
        "observation_grid": artifacts.observation_grid,
        "waveforms": artifacts.waveforms,
    }
    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "bundle_type": BUNDLE_TYPE,
        "segment_id": identity.segment_id,
        "partition_id": identity.partition_id,
        "partition_index": identity.partition_index,
        "axial_length_m": identity.axial_length_m,
        "chainage_start_m": identity.chainage_start_m,
        "chainage_end_m": identity.chainage_end_m,
        "observation_year": identity.observation_year,
        "run_id": export_cfg.run_id or f"{identity.partition_id}_y{identity.observation_year}",
        "sim_engine": "well_array_sim",
        "sim_engine_version": export_cfg.sim_engine_version,
        "physics_mode": "ray_packet_pulse_echo",
        "inference_mode": (
            scenario.inference.mode
            if scenario.inference is not None
            else "matched_filter"
        ),
        "scenario_ref": export_cfg.scenario_path,
        "scenario_params_hash": scenario_params_hash(scenario),
        "grid": {
            "n_z": grid.n_z,
            "n_theta": grid.n_theta,
            "z_step_m": grid.z_step_m,
            "theta_step_deg": grid.theta_step_deg,
        },
        "pipe_nominal": pipe_nominal,
        "degradation_inputs": export_cfg.degradation_inputs,
        "artifacts": artifact_map,
        "summary_scalars": {
            "t_remaining_mm_min": summaries.t_remaining_mm_min,
            "t_remaining_mm_mean": summaries.t_remaining_mm_mean,
            "metal_loss_mm_max": summaries.metal_loss_mm_max,
            "inner_radius_mm_mean": summaries.inner_radius_mm_mean,
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
