from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pipe2D:
    """Axisymmetric pipe cross-section centered at origin."""

    inner_radius_m: float
    wall_thickness_m: float

    @property
    def outer_radius_m(self) -> float:
        return self.inner_radius_m + self.wall_thickness_m

    def wall_point(self, theta_rad: float) -> np.ndarray:
        """Inner wall point on radial ray at angle theta."""
        return np.array(
            [
                self.inner_radius_m * np.cos(theta_rad),
                self.inner_radius_m * np.sin(theta_rad),
            ],
            dtype=float,
        )

    def outer_wall_point(self, theta_rad: float) -> np.ndarray:
        r = self.outer_radius_m
        return np.array([r * np.cos(theta_rad), r * np.sin(theta_rad)], dtype=float)

    def wall_thickness_at(self, theta_rad: float, t_rem_m: float | None = None) -> float:
        """Nominal or degraded local wall thickness (hook for future corrosion)."""
        _ = theta_rad
        if t_rem_m is not None:
            return t_rem_m
        return self.wall_thickness_m


def default_display_length_m(inner_radius_m: float) -> float:
    """Default cylinder extent along z for 3D visualization."""
    return max(4.0 * inner_radius_m, 0.25)


@dataclass(frozen=True)
class Pipe3D:
    """Axisymmetric pipe cylinder for display; physics remain in the xy cross-section."""

    profile: Pipe2D
    length_m: float

    @property
    def inner_radius_m(self) -> float:
        return self.profile.inner_radius_m

    @property
    def outer_radius_m(self) -> float:
        return self.profile.outer_radius_m

    @property
    def z_center_m(self) -> float:
        return self.length_m / 2.0

    @classmethod
    def from_profile(cls, profile: Pipe2D, length_m: float | None = None) -> Pipe3D:
        if length_m is None:
            length_m = default_display_length_m(profile.inner_radius_m)
        return cls(profile=profile, length_m=float(length_m))


@dataclass(frozen=True)
class BoreFluid:
    """Fluid filling the pipe bore where waves propagate."""

    rho: float
    vp: float

    @property
    def impedance(self) -> float:
        return self.rho * self.vp


@dataclass(frozen=True)
class SteelWall:
    """Pipe wall material for reflection coefficient."""

    rho: float
    vp: float

    @property
    def impedance(self) -> float:
        return self.rho * self.vp
