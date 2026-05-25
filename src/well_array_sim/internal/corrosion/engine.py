from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from well_array_sim.internal.corrosion.config import CorrosionConfig
from well_array_sim.internal.corrosion.pitting import (
    Pit,
    localized_loss_m,
    pit_depth_m,
    pit_radius_m,
    sample_pit_k,
    sample_pit_n,
)
from well_array_sim.internal.corrosion.point_cloud import PipeWallPointCloud, build_pipe_wall_point_cloud
from well_array_sim.internal.pipe import Pipe3D


@dataclass
class CorrosionSnapshot:
    time_yr: float
    cloud: PipeWallPointCloud
    n_pits: int
    uniform_loss_m: float


@dataclass
class CorrosionEngine:
    config: CorrosionConfig
    cloud: PipeWallPointCloud
    time_yr: float = 0.0
    pits: list[Pit] = field(default_factory=list)
    _kdtree: cKDTree | None = field(default=None, repr=False)
    _rng: np.random.Generator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._kdtree = cKDTree(self.cloud.xyz)
        self._rng = np.random.default_rng(self.config.seed)
        self._sync_metal_loss()

    @classmethod
    def from_pipe3d(cls, pipe3d: Pipe3D, config: CorrosionConfig) -> CorrosionEngine:
        cloud = build_pipe_wall_point_cloud(pipe3d)
        return cls(config=config, cloud=cloud)

    def surface_area_m2(self) -> float:
        return self.cloud.surface_area_m2()

    def _sync_metal_loss(self) -> None:
        self.cloud.metal_loss_m = self.cloud.uniform_loss_m + self.cloud.pit_loss_m

    def _apply_pit_field(self) -> None:
        """Recompute pit_loss_m from all active pits (max over overlapping craters)."""
        n = self.cloud.n_points
        pit_loss = np.zeros(n, dtype=float)
        tree = self._kdtree
        assert tree is not None
        alpha = self.config.pit_alpha
        t_now = self.time_yr

        for pit in self.pits:
            depth = pit_depth_m(pit, t_now)
            radius = pit_radius_m(depth, alpha)
            if radius <= 0.0:
                continue
            center = self.cloud.xyz[pit.center_index]
            indices = tree.query_ball_point(center, r=radius)
            if not indices:
                continue
            idx = np.asarray(indices, dtype=int)
            dists = np.linalg.norm(self.cloud.xyz[idx] - center, axis=1)
            losses = np.array(
                [localized_loss_m(float(d), depth, radius) for d in dists],
                dtype=float,
            )
            np.maximum.at(pit_loss, idx, losses)

        self.cloud.pit_loss_m = pit_loss
        self._sync_metal_loss()

    def _spawn_pool_indices(self) -> np.ndarray:
        cfg = self.config
        n_pts = self.cloud.n_points
        if cfg.hotspot is None:
            return np.arange(n_pts, dtype=int)
        hs = cfg.hotspot
        z = self.cloud.z_m
        mask = np.abs(z - hs.z_center_m) <= hs.z_half_width_m
        indices = np.where(mask)[0]
        if indices.size == 0:
            return np.arange(n_pts, dtype=int)
        return indices.astype(int)

    def _expected_new_pits(self, dt_yr: float) -> float:
        cfg = self.config
        pool = self._spawn_pool_indices()
        n_pts = self.cloud.n_points
        base = cfg.pit_lambda_per_m2_yr * self.surface_area_m2() * dt_yr
        if cfg.hotspot is None:
            return base
        frac = pool.size / max(n_pts, 1)
        return base * frac * cfg.hotspot.pit_lambda_multiplier

    def _spawn_pits(self, n_new: int) -> None:
        if n_new <= 0:
            return
        rng = self._rng
        assert rng is not None
        cfg = self.config
        pool = self._spawn_pool_indices()
        if n_new >= pool.size:
            indices = pool
        else:
            indices = rng.choice(pool, size=n_new, replace=False)
        for idx in indices:
            self.pits.append(
                Pit(
                    center_index=int(idx),
                    t_start_yr=self.time_yr,
                    k=sample_pit_k(rng, cfg.lognormal_k.mu, cfg.lognormal_k.sigma),
                    n=sample_pit_n(rng, cfg.normal_n.mu, cfg.normal_n.sigma),
                )
            )

    def step(self, dt_yr: float) -> None:
        """Advance corrosion by dt_yr years."""
        if dt_yr <= 0.0:
            return
        cfg = self.config
        rng = self._rng
        assert rng is not None

        self.cloud.uniform_loss_m += cfg.v_corr_m_per_yr * dt_yr

        expected = self._expected_new_pits(dt_yr)
        n_new = int(rng.poisson(expected))
        self._spawn_pits(n_new)

        self.time_yr += dt_yr
        self._apply_pit_field()

    def run_to(self, target_yr: float) -> None:
        """Advance simulation to target_yr using config.dt_yr substeps."""
        if target_yr < self.time_yr:
            raise ValueError(f"target_yr {target_yr} < current time {self.time_yr}")
        dt = self.config.dt_yr
        while self.time_yr + 1e-12 < target_yr:
            remaining = target_yr - self.time_yr
            self.step(min(dt, remaining))

    def snapshot(self) -> CorrosionSnapshot:
        return CorrosionSnapshot(
            time_yr=self.time_yr,
            cloud=self.cloud,
            n_pits=len(self.pits),
            uniform_loss_m=self.cloud.uniform_loss_m,
        )

    def run_snapshots(self, years: tuple[float, ...] | None = None) -> list[CorrosionSnapshot]:
        """Run to each snapshot year in order; return state at each year."""
        if years is None:
            years = self.config.snapshot_years
        out: list[CorrosionSnapshot] = []
        for yr in sorted(years):
            self.run_to(float(yr))
            out.append(self.snapshot())
        return out
