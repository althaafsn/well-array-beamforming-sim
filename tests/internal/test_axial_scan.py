from __future__ import annotations

import math

import numpy as np

from well_array_sim.internal import load_internal_scenario
from well_array_sim.internal.axial_scan import simulate_axial_scan, z_stations_m
from well_array_sim.internal.ray_forward import simulate_pulse_echo_2d


def _default_setup():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    scenario = load_internal_scenario(root / "scenarios" / "internal_pipe_default.yaml")
    return scenario


def test_z_stations_count_and_endpoints() -> None:
    stations = z_stations_m(0.4, z_step_m=0.01)
    assert len(stations) == 41
    assert np.isclose(stations[0], 0.0)
    assert np.isclose(stations[-1], 0.4)


def test_transducer_at_bore_origin() -> None:
    scenario = _default_setup()
    assert scenario.transducer.x_m == 0.0
    assert scenario.transducer.y_m == 0.0
    assert scenario.transducer.center_freq_hz == 250_000


def test_axial_scan_shapes_and_uniform_wall() -> None:
    scenario = _default_setup()
    z_vals = z_stations_m(0.04, z_step_m=0.02)
    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_vals,
        angle_step_deg=30.0,
        z_step_m=0.02,
        inference=scenario.inference,
    )
    assert scan.inferred_distance_m.shape == (len(z_vals), len(scan.angles_deg))
    assert scan.engine == "ray"
    assert scan.p_rx is not None
    assert scan.p_rx.shape == (len(z_vals), len(scan.angles_deg), len(scan.time_us))
    assert np.allclose(scan.inferred_distance_m, scan.ground_truth_distance_m, atol=0.005)


def test_axial_station_z0_matches_pulse_echo_sweep() -> None:
    scenario = _default_setup()
    angle_step_deg = 30.0
    angles_deg = np.arange(0.0, 360.0, angle_step_deg, dtype=float)
    singles = []
    for angle_deg in angles_deg:
        singles.append(
            simulate_pulse_echo_2d(
                scenario.pipe,
                scenario.fluid,
                scenario.steel,
                scenario.transducer,
                scenario.timing,
                math.radians(angle_deg),
                z_m=0.0,
                inference=scenario.inference,
            )
        )
    axial = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=angle_step_deg,
        z_step_m=0.01,
        inference=scenario.inference,
    )
    inferred = np.array([s.inferred_distance_m for s in singles])
    measured = np.array([s.measured_echo_us for s in singles])
    assert np.allclose(axial.inferred_distance_m[0], inferred, atol=1e-9)
    assert np.allclose(axial.measured_echo_us[0], measured, atol=1e-9)


def test_axial_scan_can_skip_waveform_storage() -> None:
    scenario = _default_setup()
    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        np.array([0.0]),
        angle_step_deg=90.0,
        inference=scenario.inference,
        store_waveforms=False,
    )
    assert scan.p_rx is None
    assert scan.p_tx is None
    assert scan.time_us is None
