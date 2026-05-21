from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from well_array_sim.internal.axial_scan import AxialScanResult
from well_array_sim.internal.corrosion.engine import CorrosionSnapshot
from well_array_sim.internal.corrosion.point_cloud import PipeWallPointCloud
from well_array_sim.internal.figure_layers import FigureLayers
from well_array_sim.internal.geometry3d import (
    axial_scan_pointcloud_3d,
    ground_truth_axial_pointcloud_3d,
    ground_truth_circle_2d,
)
from well_array_sim.internal.pulse_echo_result import PulseEchoResult
from well_array_sim.internal.scenario import InternalScenario
from well_array_sim.internal.wave_packet import packet_positions_at_time


def _save_figure(fig: Figure, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _inner_wall_xy(scenario: InternalScenario, z_m: float) -> np.ndarray:
    profile = scenario.wall_profile
    if profile is None:
        return ground_truth_circle_2d(scenario.pipe.inner_radius_m)
    return profile.sample_xy_at_z(z_m)


def figure_axial_cylinder_map(
    scan: AxialScanResult,
    scenario: InternalScenario,
    *,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
    layers: FigureLayers | None = None,
) -> Figure:
    """3D point cloud of SAFT-inferred inner radius R(θ, z)."""
    fig = plt.figure(figsize=(8, 7))
    if layers is not None:
        layers.fig = fig
        layers.inferred_artists.clear()
        layers.ground_truth_artists.clear()

    ax3d = fig.add_subplot(1, 1, 1, projection="3d")
    pipe3d = scenario.pipe_3d
    if show_ground_truth:
        gt_pts = ground_truth_axial_pointcloud_3d(
            pipe3d,
            wall_profile=scenario.wall_profile,
            nominal_inner_radius_m=scenario.pipe.inner_radius_m,
        )
        gt_artist = ax3d.scatter(
            gt_pts[:, 0],
            gt_pts[:, 1],
            gt_pts[:, 2],
            c="crimson",
            s=1,
            alpha=0.35,
            depthshade=True,
            label="True R(θ, z)",
        )
        if layers is not None:
            layers.ground_truth_artists.append(gt_artist)
    if show_inferred:
        scan_pts = axial_scan_pointcloud_3d(
            scan.z_stations_m,
            scan.angles_deg,
            scan.inferred_distance_m,
        )
        inf_artist = ax3d.scatter(
            scan_pts[:, 0],
            scan_pts[:, 1],
            scan_pts[:, 2],
            c="green",
            s=8,
            alpha=0.9,
            depthshade=True,
            label="Inferred R(θ, z)",
        )
        if layers is not None:
            layers.inferred_artists.append(inf_artist)

    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("SAFT inferred R(θ, z) point cloud")
    if show_inferred or show_ground_truth:
        ax3d.legend(loc="upper right", fontsize=8)
    if not show_inferred and not show_ground_truth:
        ax3d.text2D(
            0.5,
            0.5,
            "Enable Show inferred or Show ground truth",
            transform=ax3d.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.4",
        )
    return fig


def _axial_map_extent(scan: AxialScanResult) -> list[float]:
    angle_end = (
        float(scan.angles_deg[-1] + scan.angle_step_deg)
        if len(scan.angles_deg) > 1
        else 360.0
    )
    z0 = float(scan.z_stations_m[0])
    z1 = float(scan.z_stations_m[-1])
    if len(scan.z_stations_m) == 1 or np.isclose(z0, z1):
        half = max(scan.z_step_m * 0.5, 0.005)
        z0 -= half
        z1 += half
    return [
        float(scan.angles_deg[0]),
        angle_end,
        z0,
        z1,
    ]


def figure_axial_radius_map(
    scan: AxialScanResult,
    scenario: InternalScenario,
    *,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
    layers: FigureLayers | None = None,
) -> Figure:
    """Flattened heatmap of inner radius R(θ, z) — θ horizontal, z vertical."""
    del scenario  # signature parity with other axial figures
    fig, ax = plt.subplots(figsize=(10, 5))
    if layers is not None:
        layers.fig = fig
        layers.inferred_artists.clear()
        layers.ground_truth_artists.clear()

    extent = _axial_map_extent(scan)
    vmin_mm = float(
        min(scan.inferred_distance_m.min(), scan.ground_truth_distance_m.min()) * 1000.0
    )
    vmax_mm = float(
        max(scan.inferred_distance_m.max(), scan.ground_truth_distance_m.max()) * 1000.0
    )

    gt_im = ax.imshow(
        scan.ground_truth_distance_m * 1000.0,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="Reds",
        vmin=vmin_mm,
        vmax=vmax_mm,
        alpha=0.95,
        label="True R(θ, z)",
    )
    gt_im.set_visible(show_ground_truth)
    if layers is not None:
        layers.ground_truth_artists.append(gt_im)

    inf_im = ax.imshow(
        scan.inferred_distance_m * 1000.0,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin_mm,
        vmax=vmax_mm,
        alpha=0.82 if show_ground_truth else 1.0,
        label="Inferred R(θ, z)",
    )
    inf_im.set_visible(show_inferred)
    if layers is not None:
        layers.inferred_artists.append(inf_im)

    ax.set_xlabel("Steer angle θ (deg)")
    ax.set_ylabel("Axial position z (m)")
    ax.set_title("Inner radius R(θ, z) [mm]")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin_mm, vmax=vmax_mm))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Radius (mm)")

    if show_inferred or show_ground_truth:
        handles = []
        labels = []
        if show_ground_truth:
            handles.append(plt.Line2D([0], [0], color="crimson", lw=4))
            labels.append("True R(θ, z)")
        if show_inferred:
            handles.append(plt.Line2D([0], [0], color=plt.cm.viridis(0.65), lw=4))
            labels.append("Inferred R(θ, z)")
        ax.legend(handles, labels, loc="upper right", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "Enable Show inferred or Show ground truth",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.4",
        )
    return fig


def figure_packet_scene_2d(
    scenario: InternalScenario,
    result: PulseEchoResult,
    *,
    t_s: float,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
) -> Figure:
    """Cross-section with wall, ray, and wave packets at simulation time t_s."""
    z_m = result.z_m
    theta = result.theta_rad
    direction = result.beam_dir_xy
    inner_pts = _inner_wall_xy(scenario, z_m)
    outer_pts = ground_truth_circle_2d(scenario.pipe.outer_radius_m)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(outer_pts[:, 0], outer_pts[:, 1], c="gray", s=1, alpha=0.4, label="Outer wall")
    ax.scatter(inner_pts[:, 0], inner_pts[:, 1], c="steelblue", s=2, alpha=0.8, label="Inner wall (GT)")

    ray_len = scenario.pipe.outer_radius_m * 1.05
    ax.plot(
        [0.0, direction[0] * ray_len],
        [0.0, direction[1] * ray_len],
        color="orange",
        linewidth=1.5,
        alpha=0.7,
        label="Ray",
    )
    ax.scatter([0.0], [0.0], c="darkorange", s=80, zorder=5, label="Transducer")

    px, py, amps = packet_positions_at_time(
        result.trajectory,
        t_s,
        fluid_vp=result.fluid_vp,
    )
    if len(px) > 0:
        ax.scatter(px, py, c=amps, cmap="Purples", s=8 + 120 * amps, alpha=0.75, label="Wave packets")

    if show_ground_truth:
        gt_xy = direction * result.ground_truth_distance_m
        ax.scatter([gt_xy[0]], [gt_xy[1]], c="crimson", s=60, marker="x", zorder=6, label="True wall hit")
    if show_inferred:
        inf_xy = direction * result.inferred_distance_m
        ax.scatter([inf_xy[0]], [inf_xy[1]], c="green", s=60, marker="+", zorder=6, label="SAFT estimate")

    lim = scenario.pipe.outer_radius_m * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Packet scene @ t={t_s * 1e6:.1f} µs | θ={np.degrees(theta):.0f}° | z={z_m * 1000:.0f} mm"
    )
    ax.legend(loc="upper right", fontsize=8)
    return fig


def figure_pulse_echo(
    result: PulseEchoResult,
    *,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
    layers: FigureLayers | None = None,
) -> Figure:
    """TX and RX waveforms with echo markers."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if layers is not None:
        layers.fig = fig
        layers.inferred_artists.clear()
        layers.ground_truth_artists.clear()

    axes[0].plot(result.time_us, result.p_tx, color="gray", linewidth=1.0)
    axes[0].set_ylabel("TX amp.")
    axes[0].set_title("Transmitted pulse")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.time_us, result.p_rx, color="navy", linewidth=1.0, label="p_rx(t)")
    gt_us = 2.0 * result.ground_truth_distance_m / result.fluid_vp * 1e6
    inf_us = 2.0 * result.inferred_distance_m / result.fluid_vp * 1e6
    if show_ground_truth:
        artist = axes[1].axvline(gt_us, color="crimson", linestyle="--", label="True echo")
        if layers is not None:
            layers.ground_truth_artists.append(artist)
    if show_inferred:
        artist = axes[1].axvline(inf_us, color="green", linestyle="-.", label="SAFT echo")
        if layers is not None:
            layers.inferred_artists.append(artist)
    axes[1].set_xlabel("Time (µs)")
    axes[1].set_ylabel("RX amp.")
    axes[1].set_title("Received waveform (coherent tone-burst echo)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    return fig


def figure_saft_profile(
    result: PulseEchoResult,
    *,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
    layers: FigureLayers | None = None,
) -> Figure:
    """Blind SAFT range profile I(r)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    if layers is not None:
        layers.fig = fig
        layers.inferred_artists.clear()
        layers.ground_truth_artists.clear()

    r_mm = result.range_profile_r_m * 1000.0
    ax.plot(r_mm, result.range_profile_I, color="darkgreen", linewidth=1.5, label="I(r)")
    if show_inferred:
        artist = ax.axvline(
            result.inferred_distance_m * 1000.0,
            color="green",
            linestyle="-.",
            label=f"SAFT r̂={result.inferred_distance_m * 1000:.1f} mm",
        )
        if layers is not None:
            layers.inferred_artists.append(artist)
    if show_ground_truth:
        artist = ax.axvline(
            result.ground_truth_distance_m * 1000.0,
            color="crimson",
            linestyle="--",
            label=f"True R={result.ground_truth_distance_m * 1000:.1f} mm",
        )
        if layers is not None:
            layers.ground_truth_artists.append(artist)
    ax.set_xlabel("Trial radius r (mm)")
    ax.set_ylabel("Matched-filter power")
    ax.set_title(f"Blind SAFT @ θ={np.degrees(result.theta_rad):.0f}°")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def save_axial_scan_npz(scan: AxialScanResult, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "export_type": "AxialScan3D",
        "engine": scan.engine,
        "z_stations_m": scan.z_stations_m,
        "angles_deg": scan.angles_deg,
        "inferred_distance_m": scan.inferred_distance_m,
        "measured_echo_us": scan.measured_echo_us,
        "peak_amplitude": scan.peak_amplitude,
        "error_mm": scan.error_mm,
        "wall_distance_m": scan.wall_distance_m,
        "ground_truth_distance_m": scan.ground_truth_distance_m,
        "angle_step_deg": scan.angle_step_deg,
        "z_step_m": scan.z_step_m,
    }
    if scan.time_us is not None:
        payload["time_us"] = scan.time_us
    if scan.p_tx is not None:
        payload["p_tx"] = scan.p_tx
    if scan.p_rx is not None:
        payload["p_rx"] = scan.p_rx
    np.savez_compressed(out_path, **payload)
    return out_path


def plot_axial_scan_exports(
    scan: AxialScanResult,
    scenario: InternalScenario,
    out: Path,
    *,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
) -> tuple[Path, Path, Path]:
    """Write axial SAFT point cloud, radius map PNGs, and NPZ."""
    out = Path(out)
    cloud_path = _save_figure(
        figure_axial_cylinder_map(
            scan,
            scenario,
            show_inferred=show_inferred,
            show_ground_truth=show_ground_truth,
        ),
        Path(f"{out}_axial_point_cloud.png"),
    )
    map_path = _save_figure(
        figure_axial_radius_map(
            scan,
            scenario,
            show_inferred=show_inferred,
            show_ground_truth=show_ground_truth,
        ),
        Path(f"{out}_axial_radius_map.png"),
    )
    npz_path = save_axial_scan_npz(scan, Path(f"{out}_axial_scan.npz"))
    return cloud_path, map_path, npz_path


def plot_packet_scene(
    scenario: InternalScenario,
    result: PulseEchoResult,
    out_path: Path,
    *,
    t_s: float,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
) -> Path:
    return _save_figure(
        figure_packet_scene_2d(
            scenario,
            result,
            t_s=t_s,
            show_inferred=show_inferred,
            show_ground_truth=show_ground_truth,
        ),
        out_path,
    )


def plot_pulse_echo(
    result: PulseEchoResult,
    out_path: Path,
    *,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
) -> Path:
    return _save_figure(
        figure_pulse_echo(
            result,
            show_inferred=show_inferred,
            show_ground_truth=show_ground_truth,
        ),
        out_path,
    )


def plot_saft_profile(
    result: PulseEchoResult,
    out_path: Path,
    *,
    show_inferred: bool = True,
    show_ground_truth: bool = True,
) -> Path:
    return _save_figure(
        figure_saft_profile(
            result,
            show_inferred=show_inferred,
            show_ground_truth=show_ground_truth,
        ),
        out_path,
    )


def save_pulse_echo_npz(result: PulseEchoResult, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        export_type="PulseEchoRay2D",
        theta_rad=result.theta_rad,
        z_m=result.z_m,
        beam_dir_xy=result.beam_dir_xy,
        wall_distance_m=result.wall_distance_m,
        ground_truth_distance_m=result.ground_truth_distance_m,
        inferred_distance_m=result.inferred_distance_m,
        reflection_coeff=result.reflection_coeff,
        time_us=result.time_us,
        p_tx=result.p_tx,
        p_rx=result.p_rx,
        range_profile_r_m=result.range_profile_r_m,
        range_profile_I=result.range_profile_I,
        fluid_vp=result.fluid_vp,
    )
    return out_path


def _corrosion_scalar_mm(cloud: PipeWallPointCloud, color_by: str) -> np.ndarray:
    if color_by == "metal_loss":
        return cloud.total_metal_loss_m * 1000.0
    return cloud.remaining_thickness_m * 1000.0


def figure_corrosion_pointcloud_3d(
    snapshot: CorrosionSnapshot,
    *,
    color_by: str = "t_remaining",
) -> Figure:
    """3D scatter of wall mesh colored by remaining thickness or metal loss."""
    cloud = snapshot.cloud
    values_mm = _corrosion_scalar_mm(cloud, color_by)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sc = ax.scatter(
        cloud.xyz[:, 0],
        cloud.xyz[:, 1],
        cloud.xyz[:, 2],
        c=values_mm,
        cmap="plasma_r" if color_by == "t_remaining" else "hot",
        s=4,
        alpha=0.85,
        depthshade=True,
    )
    label = "t_remaining (mm)" if color_by == "t_remaining" else "metal_loss (mm)"
    fig.colorbar(sc, ax=ax, label=label, shrink=0.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(
        f"Corrosion @ T={snapshot.time_yr:.1f} yr | "
        f"pits={snapshot.n_pits} | uniform ML={snapshot.uniform_loss_m*1000:.2f} mm"
    )
    return fig


def figure_corrosion_thickness_map(snapshot: CorrosionSnapshot) -> Figure:
    """Flattened heatmap of remaining wall thickness t_rem(θ, z)."""
    cloud = snapshot.cloud
    extent = _axial_map_extent_from_axes(cloud.z_axis(), cloud.theta_axis_rad())
    data_mm = cloud.remaining_thickness_grid() * 1000.0
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        data_mm,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="plasma_r",
    )
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Remaining thickness t_rem(θ, z) @ T={snapshot.time_yr:.1f} yr [mm]")
    fig.colorbar(im, ax=ax, label="t_rem (mm)")
    return fig


def _axial_map_extent_from_axes(z_vals: np.ndarray, theta_rad: np.ndarray) -> list[float]:
    angle_end = float(np.rad2deg(theta_rad[-1] + (theta_rad[1] - theta_rad[0])))
    z0 = float(z_vals[0])
    z1 = float(z_vals[-1])
    if len(z_vals) == 1 or np.isclose(z0, z1):
        half = 0.005
        z0 -= half
        z1 += half
    return [float(np.rad2deg(theta_rad[0])), angle_end, z0, z1]


def save_corrosion_npz(snapshot: CorrosionSnapshot, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = snapshot.cloud
    np.savez_compressed(
        out_path,
        export_type="CorrosionSnapshot",
        time_yr=snapshot.time_yr,
        n_pits=snapshot.n_pits,
        uniform_loss_m=snapshot.uniform_loss_m,
        xyz=cloud.xyz,
        metal_loss_m=cloud.total_metal_loss_m,
        t_remaining_m=cloud.remaining_thickness_m,
        inner_radius_m=cloud.inner_radius_m,
        t_nominal_m=cloud.t_nominal_m,
        nominal_inner_radius_m=cloud.nominal_inner_radius_m,
    )
    return out_path


def plot_corrosion_exports(
    snapshot: CorrosionSnapshot,
    out: Path,
) -> tuple[Path, Path, Path]:
    out = Path(out)
    yr = snapshot.time_yr
    cloud_path = _save_figure(
        figure_corrosion_pointcloud_3d(snapshot),
        Path(f"{out}_corrosion_{yr:g}yr_3d.png"),
    )
    map_path = _save_figure(
        figure_corrosion_thickness_map(snapshot),
        Path(f"{out}_corrosion_{yr:g}yr_map.png"),
    )
    npz_path = save_corrosion_npz(snapshot, Path(f"{out}_corrosion_{yr:g}yr.npz"))
    return cloud_path, map_path, npz_path
