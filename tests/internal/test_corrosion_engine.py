from __future__ import annotations

import math

import numpy as np
import pytest

from well_array_sim.internal import load_internal_scenario, simulate_pulse_echo_2d
from well_array_sim.internal.corrosion.config import CorrosionConfig, DistributionParams
from well_array_sim.internal.corrosion.engine import CorrosionEngine
from well_array_sim.internal.corrosion.pitting import Pit, localized_loss_m, pit_depth_m, pit_radius_m
from well_array_sim.internal.corrosion.point_cloud import build_pipe_wall_point_cloud
from well_array_sim.internal.corrosion.bridge import wall_profile_from_point_cloud


@pytest.fixture
def corrosion_scenario():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    return load_internal_scenario(root / "scenarios" / "internal_pipe_corrosion_default.yaml")


def _uniform_config() -> CorrosionConfig:
    return CorrosionConfig(
        v_corr_m_per_yr=0.05e-3,
        pit_lambda_per_m2_yr=0.0,
        pit_alpha=3.0,
        dt_yr=1.0,
        t_end_yr=10.0,
        snapshot_years=(0.0, 5.0),
        seed=1,
        lognormal_k=DistributionParams(0.4, 0.1),
        normal_n=DistributionParams(0.5, 0.05),
    )


def test_uniform_only() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_corrosion_default.yaml")
    cfg = CorrosionConfig(
        v_corr_m_per_yr=0.05e-3,
        pit_lambda_per_m2_yr=0.0,
        pit_alpha=3.0,
        dt_yr=2.0,
        t_end_yr=10.0,
        snapshot_years=(0.0,),
        seed=99,
        lognormal_k=DistributionParams(0.4, 0.1),
        normal_n=DistributionParams(0.5, 0.05),
    )
    engine = CorrosionEngine.from_pipe3d(scenario.pipe_3d, cfg)
    engine.run_to(4.0)
    expected = 0.05e-3 * 4.0
    assert np.allclose(engine.cloud.total_metal_loss_m, expected, rtol=1e-9)
    assert np.allclose(engine.cloud.remaining_thickness_m, scenario.pipe.wall_thickness_m - expected)


def test_poisson_spawn_count_seeded() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_corrosion_default.yaml")
    cfg = _uniform_config()
    cfg = CorrosionConfig(
        v_corr_m_per_yr=cfg.v_corr_m_per_yr,
        pit_lambda_per_m2_yr=50.0,
        pit_alpha=3.0,
        dt_yr=1.0,
        t_end_yr=1.0,
        snapshot_years=(0.0,),
        seed=12345,
        lognormal_k=cfg.lognormal_k,
        normal_n=cfg.normal_n,
    )
    engine = CorrosionEngine.from_pipe3d(scenario.pipe_3d, cfg)
    area = engine.surface_area_m2()
    expected_mean = 50.0 * area * 1.0
    engine.step(1.0)
    assert len(engine.pits) > 0
    assert len(engine.pits) < expected_mean + 5 * math.sqrt(expected_mean) + 20


def test_single_pit_profile() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_corrosion_default.yaml")
    cfg = _uniform_config()
    engine = CorrosionEngine.from_pipe3d(scenario.pipe_3d, cfg)
    center = engine.cloud.n_points // 2
    engine.pits.append(Pit(center_index=center, t_start_yr=0.0, k=0.01, n=0.5))
    engine.time_yr = 4.0
    engine._apply_pit_field()
    depth = pit_depth_m(engine.pits[0], 4.0)
    radius = pit_radius_m(depth, cfg.pit_alpha)
    loss_center = engine.cloud.pit_loss_m[center]
    assert loss_center == pytest.approx(depth, rel=1e-6)
    far_idx = (center + engine.cloud.n_points // 3) % engine.cloud.n_points
    dist = np.linalg.norm(engine.cloud.xyz[far_idx] - engine.cloud.xyz[center])
    if dist >= radius:
        assert engine.cloud.pit_loss_m[far_idx] == 0.0
    else:
        expected = localized_loss_m(dist, depth, radius)
        assert engine.cloud.pit_loss_m[far_idx] == pytest.approx(expected, rel=1e-6)


def test_overlap_uses_max() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_corrosion_default.yaml")
    cfg = _uniform_config()
    engine = CorrosionEngine.from_pipe3d(scenario.pipe_3d, cfg)
    i0 = 100
    i1 = 101
    engine.pits = [
        Pit(center_index=i0, t_start_yr=0.0, k=0.02, n=0.5),
        Pit(center_index=i1, t_start_yr=0.0, k=0.005, n=0.5),
    ]
    engine.time_yr = 10.0
    engine._apply_pit_field()
    d0_center = engine.cloud.pit_loss_m[i0]
    d1_center = engine.cloud.pit_loss_m[i1]
    assert d0_center >= d1_center
    depth0 = pit_depth_m(engine.pits[0], 10.0)
    radius0 = pit_radius_m(depth0, cfg.pit_alpha)
    dist01 = np.linalg.norm(engine.cloud.xyz[i1] - engine.cloud.xyz[i0])
    loss0_at_i1 = localized_loss_m(dist01, depth0, radius0) if dist01 < radius0 else 0.0
    assert engine.cloud.pit_loss_m[i1] == pytest.approx(max(loss0_at_i1, d1_center), rel=1e-5)


def test_thickness_floor() -> None:
    scenario = load_internal_scenario("scenarios/internal_pipe_corrosion_default.yaml")
    cfg = CorrosionConfig(
        v_corr_m_per_yr=1.0,
        pit_lambda_per_m2_yr=0.0,
        pit_alpha=3.0,
        dt_yr=1.0,
        t_end_yr=1.0,
        snapshot_years=(0.0,),
        seed=1,
        lognormal_k=DistributionParams(0.4, 0.1),
        normal_n=DistributionParams(0.5, 0.05),
    )
    engine = CorrosionEngine.from_pipe3d(scenario.pipe_3d, cfg)
    engine.run_to(1.0)
    assert np.all(engine.cloud.remaining_thickness_m >= 0.0)


def test_bridge_wall_profile(corrosion_scenario) -> None:
    engine = corrosion_scenario.build_corrosion_engine()
    engine.run_to(2.0)
    profile = wall_profile_from_point_cloud(engine.cloud, corrosion_scenario.pipe)
    r_nom = corrosion_scenario.pipe.inner_radius_m
    iz, it = 0, 0
    z_m = float(profile.z_m[iz])
    theta = float(profile.theta_rad[it])
    expected_r = float(engine.cloud.inner_radius_grid()[iz, it])
    assert profile.lookup(z_m, theta) == pytest.approx(expected_r, rel=1e-9)
    assert expected_r >= r_nom


def test_acoustic_gt_moves_with_corrosion(corrosion_scenario) -> None:
    wp0 = corrosion_scenario.wall_profile_at_year(0.0)
    wp5 = corrosion_scenario.wall_profile_at_year(5.0)
    r0 = simulate_pulse_echo_2d(
        corrosion_scenario.pipe,
        corrosion_scenario.fluid,
        corrosion_scenario.steel,
        corrosion_scenario.transducer,
        corrosion_scenario.timing,
        math.radians(45.0),
        z_m=0.0,
        wall_profile=wp0,
        inference=corrosion_scenario.inference,
    )
    r5 = simulate_pulse_echo_2d(
        corrosion_scenario.pipe,
        corrosion_scenario.fluid,
        corrosion_scenario.steel,
        corrosion_scenario.transducer,
        corrosion_scenario.timing,
        math.radians(45.0),
        z_m=0.0,
        wall_profile=wp5,
        inference=corrosion_scenario.inference,
    )
    assert r5.ground_truth_distance_m > r0.ground_truth_distance_m


def test_default_corrosion_params_are_mm_scale() -> None:
    """Default scenario should produce realistic (mm-scale) pit growth."""
    scenario = load_internal_scenario("scenarios/internal_pipe_corrosion_default.yaml")
    engine = scenario.build_corrosion_engine()
    engine.run_to(10.0)

    max_loss_mm = float(engine.cloud.total_metal_loss_m.max() * 1000.0)
    mean_loss_mm = float(engine.cloud.total_metal_loss_m.mean() * 1000.0)

    # Uniform corrosion alone contributes 0.5 mm at 10 years in default YAML.
    assert mean_loss_mm >= 0.5
    # Pitting should add variation, but remain realistic (not meter-scale blowups).
    assert max_loss_mm > mean_loss_mm
    assert max_loss_mm < 10.0
