from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from well_array_sim.internal import BoreFluid, Pipe2D, load_internal_scenario


def test_pipe2d_wall_point() -> None:
    pipe = Pipe2D(inner_radius_m=0.1092375, wall_thickness_m=0.013)
    pt = pipe.wall_point(0.0)
    assert np.isclose(pt[0], pipe.inner_radius_m)
    assert np.isclose(pt[1], 0.0)
    assert np.isclose(pipe.outer_radius_m, pipe.inner_radius_m + pipe.wall_thickness_m)


def test_load_internal_scenario() -> None:
    root = Path(__file__).resolve().parents[2]
    scenario = load_internal_scenario(root / "scenarios" / "internal_pipe_default.yaml")
    assert scenario.pipe.inner_radius_m > 0
    assert scenario.fluid.vp == 1300
    assert scenario.transducer.center_freq_hz == 250_000
    assert scenario.angle_step_deg == 1.0
    assert scenario.z_step_m == 0.01
    assert scenario.pipe_3d.length_m == 0.4
    assert len(scenario.z_stations()) == 41


def test_bore_fluid_impedance() -> None:
    fluid = BoreFluid(rho=900, vp=1300)
    assert fluid.impedance == 900 * 1300
