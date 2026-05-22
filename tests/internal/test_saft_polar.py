from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from well_array_sim.internal import load_internal_scenario
from well_array_sim.internal.axial_scan import simulate_axial_scan
from well_array_sim.internal.ray_forward import simulate_pulse_echo_2d
from well_array_sim.internal.saft_polar import infer_axial_slice_saft, polar_saft_focus_image
from well_array_sim.internal.scenario import InferenceConfig


ROOT = Path(__file__).resolve().parents[2]


def _default_scenario():
    return load_internal_scenario(ROOT / "scenarios" / "internal_pipe_default.yaml")


def _matched_filter_inference(base: InferenceConfig) -> InferenceConfig:
    return replace(base, mode="matched_filter")


def _narrow_angular_inference(base: InferenceConfig) -> InferenceConfig:
    return replace(base, mode="angular_saft", angular_window_deg=1.0)


def test_angular_saft_uniform_wall() -> None:
    scenario = _default_scenario()
    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=30.0,
        inference=scenario.inference,
    )
    assert np.allclose(scan.inferred_distance_m, scan.ground_truth_distance_m, atol=0.005)


def test_narrow_angular_window_near_matched_filter() -> None:
    scenario = _default_scenario()
    mf = _matched_filter_inference(scenario.inference)
    narrow = _narrow_angular_inference(scenario.inference)

    mf_scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=30.0,
        inference=mf,
    )
    narrow_scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=30.0,
        inference=narrow,
    )
    assert np.allclose(narrow_scan.inferred_distance_m, mf_scan.inferred_distance_m, atol=0.003)


def test_axial_station_z0_matches_pulse_echo_sweep_matched_filter() -> None:
    scenario = _default_scenario()
    inference = _matched_filter_inference(scenario.inference)
    angle_step_deg = 30.0
    angles_deg = np.arange(0.0, 360.0, angle_step_deg, dtype=float)
    singles = [
        simulate_pulse_echo_2d(
            scenario.pipe,
            scenario.fluid,
            scenario.steel,
            scenario.transducer,
            scenario.timing,
            math.radians(angle_deg),
            z_m=0.0,
            inference=inference,
        )
        for angle_deg in angles_deg
    ]
    axial = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=angle_step_deg,
        inference=inference,
    )
    inferred = np.array([s.inferred_distance_m for s in singles])
    measured = np.array([s.measured_echo_us for s in singles])
    assert np.allclose(axial.inferred_distance_m[0], inferred, atol=1e-9)
    assert np.allclose(axial.measured_echo_us[0], measured, atol=1e-9)


def test_angular_saft_finds_bump_in_axial_scan() -> None:
    scenario = load_internal_scenario(ROOT / "scenarios" / "internal_pipe_wavy_wall.yaml")
    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=10.0,
        wall_profile=scenario.wall_profile,
        echo=scenario.echo,
        inference=scenario.inference,
    )
    bump_idx = int(np.argmin(np.abs(scan.angles_deg - 45.0)))
    gt = scan.ground_truth_distance_m[0, bump_idx]
    inferred = scan.inferred_distance_m[0, bump_idx]
    assert gt > scenario.pipe.inner_radius_m
    assert abs(inferred - gt) < 0.008


def test_angular_saft_reduces_noise_on_uniform_wall() -> None:
    from well_array_sim.internal.wall_profile import EchoConfig

    scenario = _default_scenario()
    echo = EchoConfig(amplitude_exponent=0.0, snr_db=18.0, noise_seed=7)
    mf = _matched_filter_inference(scenario.inference)
    angular = scenario.inference

    z_vals = np.array([0.0])
    mf_scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_vals,
        angle_step_deg=10.0,
        echo=echo,
        inference=mf,
    )
    ang_scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_vals,
        angle_step_deg=10.0,
        echo=echo,
        inference=angular,
    )

    mf_err = np.abs(mf_scan.inferred_distance_m - mf_scan.ground_truth_distance_m)
    ang_err = np.abs(ang_scan.inferred_distance_m - ang_scan.ground_truth_distance_m)
    assert float(np.mean(ang_err)) <= float(np.mean(mf_err))


def test_saft_polar_module_is_blind_to_wall_profile() -> None:
    source = (ROOT / "src" / "well_array_sim" / "internal" / "saft_polar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "WallProfile" not in names
    assert "ground_truth" not in names
    assert "inner_radius_at" not in names


def test_inference_mode_saft_alias_removed() -> None:
    import pytest

    from well_array_sim.internal.scenario import parse_inference_config

    with pytest.raises(ValueError, match='inference.mode "saft" was removed'):
        parse_inference_config({"inference": {"mode": "saft"}}, nominal_inner_radius_m=0.1)


def test_polar_focus_image_shape() -> None:
    scenario = _default_scenario()
    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=45.0,
        inference=scenario.inference,
    )
    assert scan.p_rx is not None and scan.p_tx is not None and scan.time_us is not None
    time_s = scan.time_us * 1e-6
    r_grid = np.linspace(scenario.inference.r_min_m, scenario.inference.r_max_m, 50)
    image = polar_saft_focus_image(
        scan.p_rx[0],
        scan.p_tx,
        time_s,
        scan.angles_deg,
        r_grid,
        scenario.fluid.vp,
        beam_width_deg=15.0,
        coherent_sum=True,
    )
    assert image.shape == (len(r_grid), len(scan.angles_deg))

    inferred, focus = infer_axial_slice_saft(
        scan.p_rx[0],
        scan.p_tx,
        time_s,
        scan.angles_deg,
        scenario.fluid.vp,
        scenario.inference,
    )
    assert inferred.shape == (len(scan.angles_deg),)
    assert focus.shape[1] == len(scan.angles_deg)
