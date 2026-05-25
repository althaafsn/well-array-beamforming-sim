import numpy as np
import pytest
from pathlib import Path

from well_array_sim.internal import load_internal_scenario
from well_array_sim.internal.axial_scan import simulate_axial_scan
from well_array_sim.internal._rust_backend import rust_available, simulate_axial_scan_rust
from well_array_sim.internal.wall_profile import EchoConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not rust_available(), reason="rust extension not built")
def test_noise_per_shot_reset_matches_python_pattern(monkeypatch):
    """Python resets RNG each shot; Rust must do the same (identical noise per angle at a z)."""
    scenario = load_internal_scenario(ROOT / "scenarios" / "internal_pipe_default.yaml")
    echo = EchoConfig(snr_db=20.0, noise_seed=123, amplitude_exponent=0.0)
    z = np.array([0.0, 0.05])
    monkeypatch.delenv("WELL_ARRAY_SIM_USE_RUST", raising=False)
    py = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z,
        angle_step_deg=45.0,
        echo=echo,
        inference=scenario.inference,
    )
    rs = simulate_axial_scan_rust(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z,
        angle_step_deg=45.0,
        echo=echo,
        inference=scenario.inference,
    )
    assert rs is not None
    assert py.p_rx is not None and rs.p_rx is not None

    # Per-shot seed reset: every angle at a given z gets the same noise draw.
    for backend_rx in (py.p_rx, rs.p_rx):
        for iz in range(len(z)):
            for it in range(1, backend_rx.shape[1]):
                assert np.allclose(backend_rx[iz, 0], backend_rx[iz, it], atol=1e-12)

    # Noise was applied (traces differ from a no-noise run).
    rs_clean = simulate_axial_scan_rust(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z,
        angle_step_deg=45.0,
        echo=EchoConfig(snr_db=None, noise_seed=None, amplitude_exponent=0.0),
        inference=scenario.inference,
    )
    assert rs_clean is not None and rs_clean.p_rx is not None
    assert not np.allclose(rs.p_rx, rs_clean.p_rx, atol=1e-9)
