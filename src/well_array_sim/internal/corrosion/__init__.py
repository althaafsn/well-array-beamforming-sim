"""Time-stepping corrosion on dense inner-wall point clouds."""

from well_array_sim.internal.corrosion.bridge import wall_profile_from_point_cloud
from well_array_sim.internal.corrosion.config import CorrosionConfig, parse_corrosion_config
from well_array_sim.internal.corrosion.engine import CorrosionEngine, CorrosionSnapshot
from well_array_sim.internal.corrosion.point_cloud import PipeWallPointCloud, build_pipe_wall_point_cloud
from well_array_sim.internal.corrosion.pitting import Pit

__all__ = [
    "CorrosionConfig",
    "CorrosionEngine",
    "CorrosionSnapshot",
    "PipeWallPointCloud",
    "Pit",
    "build_pipe_wall_point_cloud",
    "parse_corrosion_config",
    "wall_profile_from_point_cloud",
]
