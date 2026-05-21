from __future__ import annotations

import numpy as np

from well_array_sim.internal.corrosion.point_cloud import PipeWallPointCloud
from well_array_sim.internal.pipe import Pipe2D
from well_array_sim.internal.wall_profile import WallProfile


def wall_profile_from_point_cloud(cloud: PipeWallPointCloud, pipe: Pipe2D) -> WallProfile:
    """
    Build WallProfile from corroded point cloud for ray/SAFT simulation.

    R_inner(θ, z) = R_nom + Total_ML(θ, z).
    """
    z_axis = cloud.z_axis()
    theta_axis = cloud.theta_axis_rad()
    inner_radius_m = cloud.inner_radius_grid()
    return WallProfile(
        z_m=z_axis,
        theta_rad=theta_axis,
        inner_radius_m=inner_radius_m,
        nominal_inner_radius_m=pipe.inner_radius_m,
        amplitude_multiplier=None,
    )
