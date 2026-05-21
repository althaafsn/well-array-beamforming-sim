from __future__ import annotations

from matplotlib.figure import Figure

from well_array_sim.internal.axial_scan import AxialScanResult
from well_array_sim.internal.corrosion.engine import CorrosionSnapshot
from well_array_sim.internal.figure_layers import FigureLayers
from well_array_sim.internal.pulse_echo_result import PulseEchoResult
from well_array_sim.internal.scenario import InternalScenario
from well_array_sim.internal.visualize import (
    figure_axial_cylinder_map,
    figure_axial_radius_map,
    figure_corrosion_pointcloud_3d,
    figure_corrosion_thickness_map,
    figure_packet_scene_2d,
    figure_pulse_echo,
    figure_saft_profile,
)

MODE_SINGLE = "single"
MODE_AXIAL = "axial"
MODE_CORROSION = "corrosion"

VIEWS_SINGLE = ("Packet scene", "Pulse echo", "SAFT profile")
VIEWS_AXIAL = ("SAFT point cloud", "Radius map")
VIEWS_CORROSION = ("Corrosion 3D", "Thickness map")


def view_supports_overlays(view: str) -> bool:
    return view in (
        "Pulse echo",
        "SAFT profile",
        "Packet scene",
        "SAFT point cloud",
        "Radius map",
    )


def build_figure_for_view(
    *,
    mode: str,
    view: str,
    scenario: InternalScenario,
    pulse_echo_result: PulseEchoResult | None,
    axial_result: AxialScanResult | None,
    corrosion_snapshot: CorrosionSnapshot | None,
    show_inferred: bool,
    show_ground_truth: bool,
    layers: FigureLayers | None = None,
    packet_time_s: float = 0.0,
) -> Figure:
    """Build matplotlib figure for the selected GUI view."""
    if mode == MODE_SINGLE:
        if pulse_echo_result is None:
            raise ValueError("Single-angle view requires a pulse-echo result")
        if view == "Packet scene":
            return figure_packet_scene_2d(
                scenario,
                pulse_echo_result,
                t_s=packet_time_s,
                show_inferred=show_inferred,
                show_ground_truth=show_ground_truth,
            )
        if view == "Pulse echo":
            return figure_pulse_echo(
                pulse_echo_result,
                show_inferred=show_inferred,
                show_ground_truth=show_ground_truth,
                layers=layers,
            )
        if view == "SAFT profile":
            return figure_saft_profile(
                pulse_echo_result,
                show_inferred=show_inferred,
                show_ground_truth=show_ground_truth,
                layers=layers,
            )
        raise ValueError(f"Unknown single-angle view: {view}")

    if mode == MODE_AXIAL:
        if axial_result is None:
            raise ValueError("Axial view requires an axial scan result")
        if view == "SAFT point cloud":
            return figure_axial_cylinder_map(
                axial_result,
                scenario,
                show_inferred=show_inferred,
                show_ground_truth=show_ground_truth,
                layers=layers,
            )
        if view == "Radius map":
            return figure_axial_radius_map(
                axial_result,
                scenario,
                show_inferred=show_inferred,
                show_ground_truth=show_ground_truth,
                layers=layers,
            )
        raise ValueError(f"Unknown axial view: {view}")

    if mode == MODE_CORROSION:
        if corrosion_snapshot is None:
            raise ValueError("Corrosion view requires a corrosion snapshot")
        if view == "Corrosion 3D":
            return figure_corrosion_pointcloud_3d(corrosion_snapshot)
        if view == "Thickness map":
            return figure_corrosion_thickness_map(corrosion_snapshot)
        raise ValueError(f"Unknown corrosion view: {view}")

    raise ValueError(f"Unknown mode: {mode}")
