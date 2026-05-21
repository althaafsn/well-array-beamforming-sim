"""Build state_grid rows from wall geometry (ground truth, no alerts)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from well_array_sim.export.schema import STATE_GRID_COLUMNS
from well_array_sim.internal.pipe import Pipe2D
from well_array_sim.internal.wall_profile import WallProfile


def build_state_grid_table(
    *,
    wall_profile: WallProfile | None,
    pipe: Pipe2D,
    z_stations_m: np.ndarray,
    angles_deg: np.ndarray,
    pit_depth_m: np.ndarray | None = None,
    chainage_start_m: float = 0.0,
) -> pa.Table:
    """One row per (z, theta) with geometry state only."""
    t_nominal_m = float(pipe.wall_thickness_m)
    r_nom = float(pipe.inner_radius_m)
    rows: dict[str, list] = {col: [] for col in STATE_GRID_COLUMNS}

    for iz, z_local_m in enumerate(z_stations_m):
        z_abs_m = float(z_local_m) + float(chainage_start_m)
        for it, theta_deg in enumerate(angles_deg):
            theta_rad = float(np.deg2rad(theta_deg))
            if wall_profile is not None:
                inner_r = wall_profile.lookup(z_abs_m, theta_rad)
            else:
                inner_r = r_nom
            metal_loss_m = max(0.0, inner_r - r_nom)
            t_remaining_m = max(0.0, t_nominal_m - metal_loss_m)
            pit_d = 0.0
            if pit_depth_m is not None:
                pit_d = float(pit_depth_m[iz, it])
            is_pitted = pit_d > 1e-9 or metal_loss_m > 1e-6
            rows["z_local_m"].append(float(z_local_m))
            rows["theta_deg"].append(float(theta_deg))
            rows["inner_radius_m"].append(float(inner_r))
            rows["t_nominal_m"].append(t_nominal_m)
            rows["t_remaining_m"].append(float(t_remaining_m))
            rows["metal_loss_m"].append(float(metal_loss_m))
            rows["is_pitted"].append(bool(is_pitted))
            rows["pit_depth_m"].append(float(pit_d))
            rows["wall_x_m"].append(float(inner_r * np.cos(theta_rad)))
            rows["wall_y_m"].append(float(inner_r * np.sin(theta_rad)))
            rows["wall_z_m"].append(z_abs_m)
    return pa.table(rows)


def pit_depth_grid_from_engine(
    cloud,
    z_stations_m: np.ndarray,
    angles_deg: np.ndarray,
    *,
    chainage_start_m: float = 0.0,
) -> np.ndarray:
    """Map pit-only loss (m) onto export grid via nearest neighbor."""
    n_z = len(z_stations_m)
    n_theta = len(angles_deg)
    grid = np.zeros((n_z, n_theta), dtype=float)
    z_axis = cloud.z_axis()
    theta_axis_deg = np.rad2deg(cloud.theta_axis_rad())
    pit_grid = cloud.pit_loss_m.reshape(cloud.n_z, cloud.n_theta)
    for iz, z_local_m in enumerate(z_stations_m):
        z_abs_m = float(z_local_m) + float(chainage_start_m)
        z_idx = int(np.argmin(np.abs(z_axis - z_abs_m)))
        for it, theta_deg in enumerate(angles_deg):
            dtheta = np.abs((theta_axis_deg - theta_deg + 180) % 360 - 180)
            t_idx = int(np.argmin(dtheta))
            grid[iz, it] = float(pit_grid[z_idx, t_idx])
    return grid
