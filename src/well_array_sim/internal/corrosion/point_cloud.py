from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from well_array_sim.internal.geometry3d import GROUND_TRUTH_N_THETA, GROUND_TRUTH_Z_STEP_M
from well_array_sim.internal.pipe import Pipe2D, Pipe3D


@dataclass
class PipeWallPointCloud:
    """
    Dense inner-wall surface samples with per-point metal loss.

    Coordinates stay on the nominal cylinder; corrosion updates scalars only.
    """

    xyz: np.ndarray
    theta_rad: np.ndarray
    z_m: np.ndarray
    t_nominal_m: float
    nominal_inner_radius_m: float
    length_m: float
    metal_loss_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    uniform_loss_m: float = 0.0
    pit_loss_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_theta: int = 0
    n_z: int = 0

    def __post_init__(self) -> None:
        n = len(self.xyz)
        if self.metal_loss_m.size == 0:
            self.metal_loss_m = np.zeros(n, dtype=float)
        if self.pit_loss_m.size == 0:
            self.pit_loss_m = np.zeros(n, dtype=float)

    @property
    def n_points(self) -> int:
        return len(self.xyz)

    @property
    def total_metal_loss_m(self) -> np.ndarray:
        return self.uniform_loss_m + self.pit_loss_m

    @property
    def remaining_thickness_m(self) -> np.ndarray:
        return np.maximum(0.0, self.t_nominal_m - self.total_metal_loss_m)

    @property
    def inner_radius_m(self) -> np.ndarray:
        """Local ID radius for ultrasound: R_nom + metal loss."""
        return self.nominal_inner_radius_m + self.total_metal_loss_m

    def surface_area_m2(self) -> float:
        return 2.0 * math.pi * self.nominal_inner_radius_m * self.length_m

    def metal_loss_grid(self) -> np.ndarray:
        """Reshape total metal loss to (n_z, n_theta)."""
        return self.total_metal_loss_m.reshape(self.n_z, self.n_theta)

    def inner_radius_grid(self) -> np.ndarray:
        return self.inner_radius_m.reshape(self.n_z, self.n_theta)

    def remaining_thickness_grid(self) -> np.ndarray:
        return self.remaining_thickness_m.reshape(self.n_z, self.n_theta)

    def z_axis(self) -> np.ndarray:
        return np.asarray(self.z_m.reshape(self.n_z, self.n_theta)[:, 0], dtype=float)

    def theta_axis_rad(self) -> np.ndarray:
        return np.asarray(self.theta_rad.reshape(self.n_z, self.n_theta)[0, :], dtype=float)


def build_pipe_wall_point_cloud(
    pipe3d: Pipe3D,
    *,
    n_theta: int = GROUND_TRUTH_N_THETA,
    z_step_m: float = GROUND_TRUTH_Z_STEP_M,
) -> PipeWallPointCloud:
    """Build dense nominal inner-wall mesh for corrosion evolution."""
    r_nom = pipe3d.inner_radius_m
    length_m = pipe3d.length_m
    z_vals = np.arange(0.0, length_m + 0.5 * z_step_m, z_step_m, dtype=float)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zz, tt = np.meshgrid(z_vals, theta_vals, indexing="ij")
    x = r_nom * np.cos(tt)
    y = r_nom * np.sin(tt)
    xyz = np.column_stack([x.ravel(), y.ravel(), zz.ravel()])
    return PipeWallPointCloud(
        xyz=xyz,
        theta_rad=tt.ravel().astype(float),
        z_m=zz.ravel().astype(float),
        t_nominal_m=pipe3d.profile.wall_thickness_m,
        nominal_inner_radius_m=r_nom,
        length_m=length_m,
        metal_loss_m=np.zeros(xyz.shape[0], dtype=float),
        pit_loss_m=np.zeros(xyz.shape[0], dtype=float),
        n_theta=n_theta,
        n_z=len(z_vals),
    )


def build_from_pipe2d(pipe: Pipe2D, length_m: float | None = None) -> PipeWallPointCloud:
    pipe3d = Pipe3D.from_profile(pipe, length_m=length_m)
    return build_pipe_wall_point_cloud(pipe3d)
