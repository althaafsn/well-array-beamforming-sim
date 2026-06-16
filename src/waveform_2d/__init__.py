"""
2D finite-difference wave engine for ultrasonic NDT simulation.

Modules
-------
main
    Langtangen & Linge explicit (leapfrog) FDTD on a 2D displacement grid.
acoustic_pml
    Velocity-stress FDTD with split-field PML for pulse-echo A-scans.
webapp
    FastAPI WebSocket server and Pyodide static demo for live visualization.
"""

from .main import (
    WaveField,
    advance,
    apply_border_damping,
    inject_point,
    plot_field,
    read_probe,
    solver,
    test_quadratic,
)

__all__ = [
    "WaveField",
    "advance",
    "apply_border_damping",
    "inject_point",
    "plot_field",
    "read_probe",
    "solver",
    "test_quadratic",
]
