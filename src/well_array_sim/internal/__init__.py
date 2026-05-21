"""Internal pipe ray-packet simulation."""

from well_array_sim.internal.axial_scan import AxialScanResult, simulate_axial_scan, z_stations_m
from well_array_sim.internal.pipe import BoreFluid, Pipe2D, SteelWall
from well_array_sim.internal.pulse_echo_result import PulseEchoResult
from well_array_sim.internal.ray_forward import simulate_pulse_echo_2d
from well_array_sim.internal.corrosion import (
    CorrosionConfig,
    CorrosionEngine,
    CorrosionSnapshot,
    PipeWallPointCloud,
    parse_corrosion_config,
    wall_profile_from_point_cloud,
)
from well_array_sim.internal.scenario import InferenceConfig, InternalScenario, load_internal_scenario
from well_array_sim.internal.transducer import PointTransducer

__all__ = [
    "AxialScanResult",
    "BoreFluid",
    "CorrosionConfig",
    "CorrosionEngine",
    "CorrosionSnapshot",
    "InferenceConfig",
    "InternalScenario",
    "Pipe2D",
    "PipeWallPointCloud",
    "PointTransducer",
    "PulseEchoResult",
    "SteelWall",
    "load_internal_scenario",
    "parse_corrosion_config",
    "simulate_axial_scan",
    "simulate_pulse_echo_2d",
    "wall_profile_from_point_cloud",
    "z_stations_m",
]
