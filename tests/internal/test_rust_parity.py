from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from well_array_sim.internal import load_internal_scenario
from well_array_sim.internal._rust_backend import rust_available, simulate_axial_scan_rust
from well_array_sim.internal.axial_scan import simulate_axial_scan

ROOT = Path(__file__).resolve().parents[2]


def _default_scenario():
    return load_internal_scenario(ROOT / "scenarios" / "internal_pipe_default.yaml")


def _wavy_scenario():
    return load_internal_scenario(ROOT / "scenarios" / "internal_pipe_wavy_wall.yaml")


@pytest.fixture
def enable_rust(monkeypatch: pytest.MonkeyPatch):
    if not rust_available():
        pytest.skip("well_array_sim_core extension not built")
    monkeypatch.setenv("WELL_ARRAY_SIM_USE_RUST", "1")


def _run_python(scenario, z_stations, *, angle_step_deg: float, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WELL_ARRAY_SIM_USE_RUST", raising=False)
    return simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_stations,
        angle_step_deg=angle_step_deg,
        z_step_m=scenario.z_step_m,
        wall_profile=scenario.wall_profile,
        echo=scenario.echo,
        inference=scenario.inference,
    )


def _run_rust(scenario, z_stations, *, angle_step_deg: float):
    result = simulate_axial_scan_rust(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_stations,
        angle_step_deg=angle_step_deg,
        z_step_m=scenario.z_step_m,
        wall_profile=scenario.wall_profile,
        echo=scenario.echo,
        inference=scenario.inference,
    )
    assert result is not None
    return result


def test_rust_extension_available_after_build():
    if os.environ.get("WELL_ARRAY_SIM_REQUIRE_RUST") == "1":
        assert rust_available()


def test_rust_ray_parity_uniform_wall(enable_rust, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _default_scenario()
    z_stations = np.array([0.0, 0.05, 0.10])
    angle_step_deg = 30.0

    py_scan = _run_python(scenario, z_stations, angle_step_deg=angle_step_deg, monkeypatch=monkeypatch)
    rs_scan = _run_rust(scenario, z_stations, angle_step_deg=angle_step_deg)

    assert rs_scan.engine == "rust"
    assert np.allclose(
        rs_scan.ground_truth_distance_m,
        py_scan.ground_truth_distance_m,
        atol=1e-6,
    )
    assert np.allclose(rs_scan.p_tx, py_scan.p_tx, atol=1e-6)
    assert py_scan.p_rx is not None and rs_scan.p_rx is not None
    assert np.allclose(rs_scan.p_rx, py_scan.p_rx, atol=1e-6)


def test_rust_saft_parity_uniform_wall(enable_rust, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _default_scenario()
    z_stations = np.array([0.0])
    angle_step_deg = 30.0

    py_scan = _run_python(scenario, z_stations, angle_step_deg=angle_step_deg, monkeypatch=monkeypatch)
    rs_scan = _run_rust(scenario, z_stations, angle_step_deg=angle_step_deg)

    assert np.allclose(
        rs_scan.inferred_distance_m,
        py_scan.inferred_distance_m,
        atol=0.005,
    )
    assert np.allclose(
        rs_scan.inferred_distance_m,
        rs_scan.ground_truth_distance_m,
        atol=0.005,
    )


def test_rust_saft_parity_wavy_wall(enable_rust, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _wavy_scenario()
    z_stations = np.array([0.0, 0.10, 0.20])
    angle_step_deg = 30.0

    py_scan = _run_python(scenario, z_stations, angle_step_deg=angle_step_deg, monkeypatch=monkeypatch)
    rs_scan = _run_rust(scenario, z_stations, angle_step_deg=angle_step_deg)

    assert np.allclose(
        rs_scan.inferred_distance_m,
        py_scan.inferred_distance_m,
        atol=0.008,
    )
    assert np.allclose(
        rs_scan.ground_truth_distance_m,
        py_scan.ground_truth_distance_m,
        atol=1e-6,
    )


def test_axial_scan_dispatches_to_rust(enable_rust, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELL_ARRAY_SIM_USE_RUST", "1")
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
    assert scan.engine == "rust"
