from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Pit:
    center_index: int
    t_start_yr: float
    k: float
    n: float


def pit_depth_m(pit: Pit, time_yr: float) -> float:
    """Maximum pit depth at centre: D_max = k * (T - T_start)^n."""
    dt = time_yr - pit.t_start_yr
    if dt <= 0.0:
        return 0.0
    return pit.k * (dt**pit.n)


def pit_radius_m(depth_m: float, alpha: float) -> float:
    """Crater half-width R_pit = alpha * D_max."""
    if depth_m <= 0.0:
        return 0.0
    return alpha * depth_m


def localized_loss_m(distance_m: float, depth_m: float, radius_m: float) -> float:
    """
    Semi-ellipsoid metal loss at distance r from pit centre.

    d_ij = D_max * sqrt(1 - (r/R)^2) for r < R; else 0.
    """
    if depth_m <= 0.0 or radius_m <= 0.0:
        return 0.0
    if distance_m >= radius_m:
        return 0.0
    ratio = distance_m / radius_m
    return depth_m * math.sqrt(max(0.0, 1.0 - ratio * ratio))


def sample_pit_k(rng: np.random.Generator, mu: float, sigma: float) -> float:
    return float(rng.lognormal(mean=mu, sigma=sigma))


def sample_pit_n(rng: np.random.Generator, mu: float, sigma: float, *, min_n: float = 0.05) -> float:
    return float(max(min_n, rng.normal(loc=mu, scale=sigma)))
