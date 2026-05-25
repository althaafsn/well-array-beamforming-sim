import numpy as np
import pytest
from pathlib import Path

from well_array_sim.internal._rust_backend import rust_available, simulate_axial_scan_rust
from well_array_sim.internal import load_internal_scenario

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not rust_available(), reason="rust extension not built")
def test_rust_saft_inferred_unchanged_after_fft(monkeypatch):
    monkeypatch.setenv("WELL_ARRAY_SIM_USE_RUST", "1")
    scenario = load_internal_scenario(ROOT / "scenarios" / "internal_pipe_wavy_wall.yaml")
    z = np.array([0.0, 0.10])
    scan = simulate_axial_scan_rust(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z,
        angle_step_deg=30.0,
        z_step_m=scenario.z_step_m,
        wall_profile=scenario.wall_profile,
        echo=scenario.echo,
        inference=scenario.inference,
    )
    assert scan is not None
    assert np.all(np.isfinite(scan.inferred_distance_m))
    assert scan.inferred_distance_m.shape == (2, 12)
