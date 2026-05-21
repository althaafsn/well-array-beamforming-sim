from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributionParams:
    mu: float
    sigma: float


@dataclass(frozen=True)
class CorrosionConfig:
    """Corrosion engine parameters (internal units: metres, years)."""

    v_corr_m_per_yr: float
    pit_lambda_per_m2_yr: float
    pit_alpha: float
    dt_yr: float
    t_end_yr: float
    snapshot_years: tuple[float, ...]
    seed: int
    lognormal_k: DistributionParams
    normal_n: DistributionParams

    @property
    def v_corr_mm_per_yr(self) -> float:
        return self.v_corr_m_per_yr * 1000.0


def parse_corrosion_config(raw: dict[str, Any] | None) -> CorrosionConfig | None:
    if not isinstance(raw, dict):
        return None
    block = raw.get("corrosion")
    if not isinstance(block, dict):
        return None

    def _dist(key: str, default_mu: float, default_sigma: float) -> DistributionParams:
        sub = block.get(key)
        if isinstance(sub, dict):
            return DistributionParams(
                mu=float(sub.get("mu", default_mu)),
                sigma=float(sub.get("sigma", default_sigma)),
            )
        return DistributionParams(mu=default_mu, sigma=default_sigma)

    def _lognormal_k_params() -> DistributionParams:
        """
        Parse pit-depth growth coefficient distribution.

        Supported YAML forms:
        - log-space (metres):
          lognormal_k:
            mu_log_m: -7.8
            sigma: 0.1
        - practical mean in mm (recommended):
          lognormal_k:
            mean_mm: 0.4
            sigma: 0.1

        Backward compatibility:
        - if only `mu` is provided, interpret it as `mean_mm`.
        """
        sub = block.get("lognormal_k")
        if not isinstance(sub, dict):
            mean_mm = 0.4
            sigma = 0.1
            mean_m = mean_mm * 1e-3
            mu_log = math.log(mean_m) - 0.5 * sigma * sigma
            return DistributionParams(mu=mu_log, sigma=sigma)

        sigma = float(sub.get("sigma", 0.1))
        if "mu_log_m" in sub:
            return DistributionParams(mu=float(sub["mu_log_m"]), sigma=sigma)

        mean_mm = float(sub.get("mean_mm", sub.get("mu", 0.4)))
        mean_m = max(mean_mm * 1e-3, 1e-12)
        mu_log = math.log(mean_m) - 0.5 * sigma * sigma
        return DistributionParams(mu=mu_log, sigma=sigma)

    snapshots = block.get("snapshot_years", [0.0, 2.0, 5.0, 10.0])
    if not isinstance(snapshots, (list, tuple)):
        snapshots = [0.0, 2.0, 5.0, 10.0]

    return CorrosionConfig(
        v_corr_m_per_yr=float(block.get("V_corr_mm_per_yr", 0.05)) * 1e-3,
        pit_lambda_per_m2_yr=float(block.get("pit_lambda_per_m2_yr", 10.0)),
        pit_alpha=float(block.get("pit_alpha", 3.0)),
        dt_yr=float(block.get("dt_yr", 0.25)),
        t_end_yr=float(block.get("t_end_yr", 10.0)),
        snapshot_years=tuple(float(y) for y in snapshots),
        seed=int(block.get("seed", 42)),
        lognormal_k=_lognormal_k_params(),
        normal_n=_dist("normal_n", 0.5, 0.05),
    )
