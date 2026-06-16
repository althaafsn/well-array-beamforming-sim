"""Tests for the 2D FDTD wave engine (``waveform_2d`` package)."""

from __future__ import annotations

import numpy as np

from waveform_2d.main import test_quadratic as verify_quadratic_solution
from waveform_2d.acoustic_pml import build_pipe_model, infer_distance, run_pulse_echo
from waveform_2d.live_sim import LiveSimulation, SCENES
from waveform_2d.main import WaveField, apply_border_damping


def test_quadratic_verification() -> None:
    err = verify_quadratic_solution(Nx=10, Ny=12)
    assert err < 1e-11


def test_wavefield_variable_velocity_recalculates_dt() -> None:
    f = WaveField(40, 40, c=1.0)
    dt_water = f.dt
    c = np.full((40, 40), 1.0)
    c[:, 20:] = 4.0
    f.set_velocity(c, recalc_dt=True)
    assert f.dt < dt_water
    assert np.isfinite(f.dt)


def test_live_sim_scenes_construct() -> None:
    for name in SCENES:
        sim = LiveSimulation(80, 80)
        sim.set_scene(name)
        assert sim.field is not None


def test_live_sim_pipe_drip_stays_finite() -> None:
    sim = LiveSimulation(120, 120)
    sim.set_scene("pipe")
    sim.clear_sources()
    sim.add_drip(0.5, 0.5)
    for _ in range(40):
        sim.step(6)
    assert np.isfinite(sim.field.field).all()
    assert float(np.abs(sim.field.field).max()) > 1e-6


def test_live_sim_drawn_structure_drip() -> None:
    sim = LiveSimulation(100, 100)
    sim.set_scene("blank")
    sim.add_steel_ring(0.5, 0.5, 0.35, 0.22)
    sim.add_drip(0.5, 0.5)
    for _ in range(30):
        sim.step(5)
    assert np.isfinite(sim.field.field).all()


def test_pml_pipe_echo_finite() -> None:
    model = build_pipe_model(80, 100, dx=1e-4, corrosion=False)
    t, ascan, _, _, _ = run_pulse_echo(model, fc=1e6, n_steps=400)
    dist, _, _ = infer_distance(t, ascan, fc=1e6)
    assert dist is not None
    assert np.isfinite(ascan).all()


def test_live_sim_render_bytes() -> None:
    sim = LiveSimulation(60, 60)
    sim.set_scene("tank")
    sim.step(3)
    blob = sim.render()
    assert len(blob) == 60 * 60 * 4
