#!/usr/bin/env python3
"""Manual benchmark for interactive performance optimizations."""

from __future__ import annotations

import math
import time
from pathlib import Path

from well_array_sim.internal import load_internal_scenario, simulate_pulse_echo_2d
from well_array_sim.internal.axial_scan import simulate_axial_scan
from well_array_sim.internal.geometry3d import clear_geometry_cache, ground_truth_cylinder_pointcloud
from well_array_sim.internal.visualize import figure_axial_cylinder_map, figure_packet_scene_2d


def _timed(label: str, fn) -> float:
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    print(f"{label:40s} {elapsed:7.3f} s")
    return elapsed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scenario = load_internal_scenario(root / "scenarios" / "internal_pipe_default.yaml")
    z_stations = scenario.z_stations()
    angle_step = 5.0

    print("=== Simulation benchmarks ===")
    _timed(
        "pulse echo @ 45 deg",
        lambda: simulate_pulse_echo_2d(
            scenario.pipe,
            scenario.fluid,
            scenario.steel,
            scenario.transducer,
            scenario.timing,
            math.radians(45.0),
            inference=scenario.inference,
        ),
    )
    _timed(
        "axial ray+SAFT @ 5 deg",
        lambda: simulate_axial_scan(
            scenario.pipe,
            scenario.fluid,
            scenario.steel,
            scenario.transducer,
            scenario.timing,
            z_stations,
            angle_step_deg=angle_step,
            z_step_m=0.01,
            inference=scenario.inference,
        ),
    )

    pulse = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        math.radians(45.0),
        inference=scenario.inference,
    )
    axial = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        z_stations,
        angle_step_deg=angle_step,
        z_step_m=0.01,
        inference=scenario.inference,
    )

    print("\n=== Visualization benchmarks ===")
    clear_geometry_cache()
    _timed(
        "packet scene",
        lambda: figure_packet_scene_2d(scenario, pulse, t_s=0.00012),
    )
    _timed(
        "cached GT cylinder",
        lambda: ground_truth_cylinder_pointcloud(scenario.pipe_3d, radius_m=scenario.pipe.inner_radius_m),
    )
    _timed("axial SAFT point cloud", lambda: figure_axial_cylinder_map(axial, scenario))


if __name__ == "__main__":
    main()
