"""Hand-authored inner wall radius maps R(z, theta) for variable-wall echo synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EchoConfig:
    amplitude_exponent: float = 0.0
    snr_db: float | None = None
    noise_seed: int | None = None


@dataclass(frozen=True)
class WallProfile:
    """Nearest-neighbor lookup of local inner radius on a (z, theta) grid."""

    z_m: np.ndarray
    theta_rad: np.ndarray
    inner_radius_m: np.ndarray
    nominal_inner_radius_m: float
    amplitude_multiplier: np.ndarray | None = None

    @property
    def is_uniform(self) -> bool:
        return bool(np.allclose(self.inner_radius_m, self.nominal_inner_radius_m, rtol=0, atol=1e-9))

    def lookup(self, z_m: float, theta_rad: float) -> float:
        z_idx = int(np.argmin(np.abs(self.z_m - z_m)))
        dtheta = np.angle(np.exp(1j * (self.theta_rad - theta_rad)))
        theta_idx = int(np.argmin(np.abs(dtheta)))
        return float(self.inner_radius_m[z_idx, theta_idx])

    def amplitude_multiplier_at(self, z_m: float, theta_rad: float) -> float:
        if self.amplitude_multiplier is None:
            return 1.0
        z_idx = int(np.argmin(np.abs(self.z_m - z_m)))
        dtheta = np.angle(np.exp(1j * (self.theta_rad - theta_rad)))
        theta_idx = int(np.argmin(np.abs(dtheta)))
        return float(self.amplitude_multiplier[z_idx, theta_idx])

    def sample_xy_at_z(self, z_m: float) -> np.ndarray:
        """Return (n_theta, 2) inner wall points at one axial station."""
        z_idx = int(np.argmin(np.abs(self.z_m - z_m)))
        radii = self.inner_radius_m[z_idx]
        return np.column_stack(
            [
                radii * np.cos(self.theta_rad),
                radii * np.sin(self.theta_rad),
            ]
        )


def has_wall_profile(raw: dict[str, Any]) -> bool:
    profile = raw.get("wall_profile")
    return isinstance(profile, dict) and bool(profile)


def parse_echo_config(raw: dict[str, Any]) -> EchoConfig | None:
    echo = raw.get("echo")
    if not isinstance(echo, dict):
        return None
    snr = echo.get("snr_db")
    return EchoConfig(
        amplitude_exponent=float(echo.get("amplitude_exponent", 0.0)),
        snr_db=None if snr is None else float(snr),
        noise_seed=None if echo.get("noise_seed") is None else int(echo["noise_seed"]),
    )


def build_wall_profile(
    raw: dict[str, Any],
    *,
    z_stations: np.ndarray | None = None,
    nominal_inner_radius_m: float,
) -> WallProfile | None:
    """Build profile from YAML or return None for uniform pipe."""
    if not has_wall_profile(raw):
        return None

    cfg = raw["wall_profile"]
    n_theta = int(cfg.get("n_theta", 72))
    theta_rad = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    if z_stations is not None and len(z_stations) > 0:
        z_m = np.asarray(z_stations, dtype=float)
    elif "z_m" in cfg:
        z_m = np.asarray(cfg["z_m"], dtype=float)
    else:
        z_m = np.array([0.0], dtype=float)

    radii = _parse_radius_grid(cfg, z_m, theta_rad, nominal_inner_radius_m)
    amp_mult = _parse_amplitude_grid(cfg, z_m, theta_rad)

    return WallProfile(
        z_m=z_m,
        theta_rad=theta_rad,
        inner_radius_m=radii,
        nominal_inner_radius_m=nominal_inner_radius_m,
        amplitude_multiplier=amp_mult,
    )


def _parse_radius_grid(
    cfg: dict[str, Any],
    z_m: np.ndarray,
    theta_rad: np.ndarray,
    nominal: float,
) -> np.ndarray:
    n_z = len(z_m)
    n_theta = len(theta_rad)

    if "inner_radius_m" in cfg:
        data = np.asarray(cfg["inner_radius_m"], dtype=float)
        if data.ndim == 1:
            if data.size != n_z * n_theta:
                raise ValueError(
                    f"wall_profile.inner_radius_m flat array size {data.size} "
                    f"!= {n_z} x {n_theta}"
                )
            return data.reshape(n_z, n_theta)
        if data.shape != (n_z, n_theta):
            raise ValueError(
                f"wall_profile.inner_radius_m shape {data.shape} != ({n_z}, {n_theta})"
            )
        return data

    if "bump" in cfg:
        return _build_bump_profile(cfg["bump"], z_m, theta_rad, nominal)

    return np.full((n_z, n_theta), nominal, dtype=float)


def _build_bump_profile(
    bump: dict[str, Any],
    z_m: np.ndarray,
    theta_rad: np.ndarray,
    nominal: float,
) -> np.ndarray:
    """Demo helper: angular sector with increased inner radius."""
    delta_m = float(bump.get("delta_m", 0.005))
    center_deg = float(bump.get("theta_deg", 45.0))
    width_deg = float(bump.get("width_deg", 30.0))
    center_rad = np.deg2rad(center_deg)
    half_width = np.deg2rad(width_deg / 2.0)

    radii = np.full((len(z_m), len(theta_rad)), nominal, dtype=float)
    for iz, z_val in enumerate(z_m):
        if "z_m" in bump:
            z_targets = np.asarray(bump["z_m"], dtype=float)
            if not np.any(np.abs(z_val - z_targets) <= float(bump.get("z_tol_m", 0.015))):
                continue
        for it, theta in enumerate(theta_rad):
            dtheta = abs(np.angle(np.exp(1j * (theta - center_rad))))
            if dtheta <= half_width:
                radii[iz, it] = nominal + delta_m
    return radii


def _parse_amplitude_grid(
    cfg: dict[str, Any],
    z_m: np.ndarray,
    theta_rad: np.ndarray,
) -> np.ndarray | None:
    if "amplitude_multiplier" not in cfg:
        return None
    n_z = len(z_m)
    n_theta = len(theta_rad)
    data = np.asarray(cfg["amplitude_multiplier"], dtype=float)
    if data.ndim == 1:
        return data.reshape(n_z, n_theta)
    return data


def inner_radius_at(
    profile: WallProfile | None,
    *,
    z_m: float,
    theta_rad: float,
    nominal_inner_radius_m: float,
) -> float:
    if profile is None:
        return nominal_inner_radius_m
    return profile.lookup(z_m, theta_rad)


def echo_amplitude_scale(
    profile: WallProfile | None,
    echo: EchoConfig | None,
    *,
    z_m: float,
    theta_rad: float,
    nominal_inner_radius_m: float,
) -> float:
    local_r = inner_radius_at(
        profile,
        z_m=z_m,
        theta_rad=theta_rad,
        nominal_inner_radius_m=nominal_inner_radius_m,
    )
    scale = 1.0
    if profile is not None and profile.amplitude_multiplier is not None:
        scale *= profile.amplitude_multiplier_at(z_m, theta_rad)
    if echo is not None and echo.amplitude_exponent != 0.0:
        if local_r > 0:
            scale *= (nominal_inner_radius_m / local_r) ** echo.amplitude_exponent
    return scale
