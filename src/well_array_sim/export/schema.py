"""Partition observation export schema v1."""

from __future__ import annotations

from dataclasses import dataclass, field

EXPORT_SCHEMA_VERSION = "1.0.0"
STUDY_SUMMARY_SCHEMA_VERSION = "1.0.0"
BUNDLE_TYPE = "pipe_partition_observation"
DEFAULT_AXIAL_LENGTH_M = 0.40

STATE_GRID_COLUMNS = [
    "z_local_m",
    "theta_deg",
    "inner_radius_m",
    "t_nominal_m",
    "t_remaining_m",
    "metal_loss_m",
    "is_pitted",
    "pit_depth_m",
    "wall_x_m",
    "wall_y_m",
    "wall_z_m",
]

OBSERVATION_GRID_COLUMNS = [
    "z_local_m",
    "theta_deg",
    "inferred_inner_radius_m",
    "echo_time_us",
    "peak_amplitude",
    "engine",
]

WAVEFORM_COLUMNS = [
    "z_local_m",
    "theta_deg",
    "sample_rate_hz",
    "t0_us",
    "n_samples",
    "p_tx",
    "p_rx",
]

FORBIDDEN_OBSERVATION_COLUMNS = {
    "ground_truth_distance_m",
    "ground_truth_inner_radius_m",
    "wall_distance_m",
    "error_mm",
}


@dataclass
class PartitionIdentity:
    segment_id: str
    partition_index: int
    chainage_start_m: float
    observation_year: int
    axial_length_m: float = DEFAULT_AXIAL_LENGTH_M

    @property
    def partition_id(self) -> str:
        return f"{self.segment_id}@p{self.partition_index:04d}"

    @property
    def chainage_end_m(self) -> float:
        return self.chainage_start_m + self.axial_length_m


@dataclass
class GridSpec:
    n_z: int
    n_theta: int
    z_step_m: float
    theta_step_deg: float


@dataclass
class ArtifactNames:
    state_grid: str = "state_grid.parquet"
    observation_grid: str = "observation_grid.parquet"
    waveforms: str = "waveforms.parquet"


@dataclass
class SummaryScalars:
    t_remaining_mm_min: float
    t_remaining_mm_mean: float
    metal_loss_mm_max: float
    inner_radius_mm_mean: float


@dataclass
class ExportConfig:
    identity: PartitionIdentity
    scenario_path: str
    z_step_m: float = 0.01
    angle_step_deg: float = 1.0
    run_id: str = ""
    sim_engine_version: str = "0.1.0"
    degradation_inputs: dict = field(default_factory=dict)
