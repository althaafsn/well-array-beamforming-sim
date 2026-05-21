from __future__ import annotations

from functools import lru_cache

import numpy as np

from well_array_sim.internal.pipe import Pipe3D

# Dense sampling for known pipe geometry (ground truth walls).
GROUND_TRUTH_N_THETA = 720
GROUND_TRUTH_Z_STEP_M = 0.005


def cylinder_surface(
    pipe3d: Pipe3D,
    *,
    radius_m: float,
    n_theta: int = 72,
    n_z: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric cylinder surface mesh centered on the z axis."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(0.0, pipe3d.length_m, n_z)
    tt, zz = np.meshgrid(theta, z_vals)
    x = radius_m * np.cos(tt)
    y = radius_m * np.sin(tt)
    return x, y, zz


def cylinder_pointcloud(
    pipe3d: Pipe3D,
    *,
    radius_m: float,
    n_theta: int = 24,
    n_z: int = 6,
) -> np.ndarray:
    """Sample points on a cylinder wall, shape (N, 3)."""
    x, y, z = cylinder_surface(pipe3d, radius_m=radius_m, n_theta=n_theta, n_z=n_z)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def circle_pointcloud_2d(radius_m: float, *, n_theta: int = 24) -> np.ndarray:
    """Sample points on a circle in the x-y plane, shape (N, 2)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    return np.column_stack([radius_m * np.cos(theta), radius_m * np.sin(theta)])


def _cylinder_n_z(length_m: float, z_step_m: float) -> int:
    return max(2, int(round(length_m / z_step_m)) + 1)


@lru_cache(maxsize=64)
def _cached_cylinder_pointcloud(
    length_m: float,
    radius_m: float,
    n_theta: int,
    z_step_m: float,
) -> np.ndarray:
    pipe3d = Pipe3D.from_profile(
        _dummy_profile(radius_m),
        length_m=length_m,
    )
    n_z = _cylinder_n_z(length_m, z_step_m)
    return cylinder_pointcloud(pipe3d, radius_m=radius_m, n_theta=n_theta, n_z=n_z)


@lru_cache(maxsize=32)
def _cached_circle_2d(radius_m: float, n_theta: int) -> np.ndarray:
    return circle_pointcloud_2d(radius_m, n_theta=n_theta)


@lru_cache(maxsize=32)
def _cached_polar_ring(wall_distance_m: float, n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radius = np.full(n_theta, wall_distance_m, dtype=float)
    return theta, radius


def _dummy_profile(inner_radius_m: float):
    from well_array_sim.internal.pipe import Pipe2D

    return Pipe2D(inner_radius_m=inner_radius_m, wall_thickness_m=0.001)


def ground_truth_circle_2d(radius_m: float) -> np.ndarray:
    """High-resolution ground-truth pipe wall ring in x-y."""
    return _cached_circle_2d(radius_m, GROUND_TRUTH_N_THETA)


def ground_truth_cylinder_pointcloud(pipe3d: Pipe3D, *, radius_m: float) -> np.ndarray:
    """High-resolution ground-truth cylinder wall samples."""
    return _cached_cylinder_pointcloud(
        pipe3d.length_m,
        radius_m,
        GROUND_TRUTH_N_THETA,
        GROUND_TRUTH_Z_STEP_M,
    )


def ground_truth_polar_ring(wall_distance_m: float) -> tuple[np.ndarray, np.ndarray]:
    """High-resolution constant-radius ring for polar plots (θ rad, r m)."""
    return _cached_polar_ring(wall_distance_m, GROUND_TRUTH_N_THETA)


def clear_geometry_cache() -> None:
    """Drop cached point clouds (e.g. after scenario geometry changes)."""
    _cached_cylinder_pointcloud.cache_clear()
    _cached_circle_2d.cache_clear()
    _cached_polar_ring.cache_clear()


def _axial_scan_pointcloud_from_grid(
    z_stations_m: np.ndarray,
    angles_deg: np.ndarray,
    radius_m: np.ndarray,
) -> np.ndarray:
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
    z_vals = np.asarray(z_stations_m, dtype=float)
    radii = np.asarray(radius_m, dtype=float)
    theta_grid, z_grid = np.meshgrid(angles_rad, z_vals)
    x = radii * np.cos(theta_grid)
    y = radii * np.sin(theta_grid)
    return np.column_stack([x.ravel(), y.ravel(), z_grid.ravel()])


def ground_truth_axial_pointcloud_3d(
    pipe3d: Pipe3D,
    *,
    wall_profile=None,
    nominal_inner_radius_m: float | None = None,
) -> np.ndarray:
    """
    Dense ground-truth inner wall samples R(θ, z) for axial overlays.

    Uses GROUND_TRUTH_N_THETA / GROUND_TRUTH_Z_STEP_M (not the coarser sim scan grid).
    """
    if wall_profile is None:
        return ground_truth_cylinder_pointcloud(pipe3d, radius_m=pipe3d.inner_radius_m)

    from well_array_sim.internal.wall_profile import inner_radius_at

    r_nom = pipe3d.inner_radius_m if nominal_inner_radius_m is None else nominal_inner_radius_m
    z_vals = np.arange(0.0, pipe3d.length_m + 0.5 * GROUND_TRUTH_Z_STEP_M, GROUND_TRUTH_Z_STEP_M)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, GROUND_TRUTH_N_THETA, endpoint=False)
    radii = np.zeros((len(z_vals), len(theta_rad)), dtype=float)
    for iz, z_m in enumerate(z_vals):
        for it, theta in enumerate(theta_rad):
            radii[iz, it] = inner_radius_at(
                wall_profile,
                z_m=float(z_m),
                theta_rad=float(theta),
                nominal_inner_radius_m=r_nom,
            )
    return _axial_scan_pointcloud_from_grid(z_vals, np.rad2deg(theta_rad), radii)


def axial_scan_pointcloud_3d(
    z_stations_m: np.ndarray,
    angles_deg: np.ndarray,
    radius_m: np.ndarray,
) -> np.ndarray:
    """Sample inferred wall points R(θ, z) as an (N, 3) point cloud."""
    return _axial_scan_pointcloud_from_grid(z_stations_m, angles_deg, radius_m)


def beam_pointcloud_2d(
    *,
    beam_dir_xy: np.ndarray,
    wall_distance_m: float,
    n_points: int = 12,
) -> np.ndarray:
    """Sample points along a steered radial beam in x-y (no connecting line)."""
    direction = np.asarray(beam_dir_xy, dtype=float)
    t = np.linspace(0.0, 1.0, n_points, dtype=float)
    return t[:, None] * direction[None, :] * wall_distance_m


def beam_pointcloud_3d(
    *,
    beam_dir_xy: np.ndarray,
    wall_distance_m: float,
    z_m: float,
    n_points: int = 12,
) -> np.ndarray:
    """Sample points along a flat (φ=0) steered beam at fixed z."""
    xy = beam_pointcloud_2d(
        beam_dir_xy=beam_dir_xy,
        wall_distance_m=wall_distance_m,
        n_points=n_points,
    )
    z = np.full((xy.shape[0], 1), float(z_m), dtype=float)
    return np.hstack([xy, z])


def beam_line_segment(
    *,
    beam_dir_xy: np.ndarray,
    wall_distance_m: float,
    z0: float,
    z1: float,
) -> np.ndarray:
    """Straight radial line at fixed (x, y) direction across two z levels."""
    direction = np.asarray(beam_dir_xy, dtype=float)
    wall_xy = direction * wall_distance_m
    return np.array(
        [
            [0.0, 0.0, z0],
            [wall_xy[0], wall_xy[1], z1],
        ],
        dtype=float,
    )
