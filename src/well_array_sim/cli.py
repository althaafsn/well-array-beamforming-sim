from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from well_array_sim.internal import load_internal_scenario
from well_array_sim.internal.axial_scan import simulate_axial_scan
from well_array_sim.internal.ray_forward import simulate_pulse_echo_2d
from well_array_sim.internal.scenario import DEFAULT_STEER_ANGLE_DEG
from well_array_sim.io.bc_pipelines import (
    format_bc_summary,
    format_segment_detail,
    format_segment_table,
    get_segment_by_permit_id,
    load_bc_segments,
    summarize_bc_segments,
    write_scenario_from_bc_segment,
)
from well_array_sim.io.om_activity import (
    export_csv,
    filter_records,
    format_summary,
    load_om_activity,
    summarize,
)
from well_array_sim.io.pipe_categories import format_category_table, spec_for_category
from well_array_sim.export.bundle import (
    export_partition_observation_bundle,
    export_partition_plan,
    export_partition_years,
)
from well_array_sim.export.partitions import partition_plan
from well_array_sim.export.schema import DEFAULT_AXIAL_LENGTH_M
from well_array_sim.segment_study import run_segment_study, print_study_report
from well_array_sim.internal.visualize import (
    plot_axial_scan_exports,
    plot_corrosion_exports,
    plot_packet_scene,
    plot_pulse_echo,
    plot_matched_filter_profile,
    save_pulse_echo_npz,
)


CLI_DESCRIPTION = (
    "Simulate internal pipe pulse-echo ultrasound: ray physics, blind wall ranging, "
    "optional corrosion, and BC pipeline segment export."
)

CLI_EPILOG = """
What this simulates
  A monostatic tool inside a fluid-filled pipe transmits ultrasound, receives wall
  echoes, and estimates inner radius (inference) compared to the true wall geometry.

Commands (run 'well-array-sim COMMAND --help' for details)
  sim               Quick demo from a scenario YAML (one shot or 360° pipe sweep)
  bc                British Columbia pipeline GIS: browse segments, run multi-year studies
  export-partition  Export platform observation bundles (0.4 m partitions × year)
  om                Browse CER operation & maintenance CSV (metadata only, not simulation)

Examples
  well-array-sim sim --angle-deg 45
  well-array-sim sim --axial-scan --scenario scenarios/internal_pipe_default.yaml
  well-array-sim bc list --line-type Transmission --limit 5
  well-array-sim bc run --permit-id 1987 --max-length-m 1.2 --max-partitions 3 --years 0
  well-array-sim-gui
"""

KNOWN_COMMANDS = frozenset({"sim", "bc", "om", "export-partition"})


def _parse_year_list(args: argparse.Namespace) -> list[int]:
    if args.years:
        return [int(y.strip()) for y in args.years.split(",") if y.strip()]
    return [int(args.observation_year)]


def _run_export_partition(args: argparse.Namespace) -> None:
    year_list = _parse_year_list(args)
    export_kwargs = {
        "scenario_path": args.scenario,
        "segment_id": args.segment_id,
        "out_root": args.out_root,
        "z_step_m": args.z_step_m,
        "angle_step_deg": args.angle_step_deg,
    }

    if args.all_partitions:
        scenario = load_internal_scenario(args.scenario)
        sim_length_m = float(scenario.pipe_3d.length_m)
        partitions = partition_plan(
            sim_length_m,
            partition_length_m=args.partition_length_m,
            max_partitions=args.max_partitions,
        )
        paths = export_partition_plan(
            **export_kwargs,
            partitions=partitions,
            years=year_list,
        )
        for path in paths:
            print(path)
        return

    if len(year_list) > 1:
        paths = export_partition_years(
            **export_kwargs,
            years=year_list,
            partition_index=args.partition_index,
            chainage_start_m=args.chainage_start_m,
            axial_length_m=args.axial_length_m,
            run_id=args.run_id or "",
        )
        for path in paths:
            print(path)
        return

    path = export_partition_observation_bundle(
        **export_kwargs,
        observation_year=year_list[0],
        partition_index=args.partition_index,
        chainage_start_m=args.chainage_start_m,
        axial_length_m=args.axial_length_m,
        run_id=args.run_id or "",
    )
    print(path)


def _add_export_partition_arguments(export_parser: argparse.ArgumentParser) -> None:
    export_parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("scenarios/internal_pipe_corrosion_default.yaml"),
    )
    export_parser.add_argument("--segment-id", type=str, required=True)
    export_parser.add_argument(
        "--all-partitions",
        action="store_true",
        help="Export every partition in scenario pipe.length_m (default: single partition)",
    )
    export_parser.add_argument("--partition-index", type=int, default=0)
    export_parser.add_argument("--chainage-start-m", type=float, default=0.0)
    export_parser.add_argument(
        "--axial-length-m",
        type=float,
        default=DEFAULT_AXIAL_LENGTH_M,
        help="Partition length along pipe when exporting one partition (default: 0.40 m)",
    )
    export_parser.add_argument(
        "--partition-length-m",
        type=float,
        default=DEFAULT_AXIAL_LENGTH_M,
        help="Partition size when --all-partitions is set (default: 0.40 m)",
    )
    export_parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="Cap partitions exported with --all-partitions",
    )
    export_parser.add_argument("--observation-year", type=int, default=0)
    export_parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years (overrides --observation-year), e.g. 0,5,10",
    )
    export_parser.add_argument("--out-root", type=Path, required=True)
    export_parser.add_argument(
        "--z-step-m",
        type=float,
        default=None,
        help="Axial station spacing (default: scenario scan.z_step_m)",
    )
    export_parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=None,
        help="Azimuth step for 360° sweep (default: scenario scan.angle_step_deg)",
    )
    export_parser.add_argument("--run-id", type=str, default="")


def _build_export_partition_parser() -> argparse.ArgumentParser:
    export_parser = argparse.ArgumentParser(
        description="Export pipe_partition_observation bundles for platform ingest",
    )
    _add_export_partition_arguments(export_parser)
    return export_parser


def main_export_partition(argv: list[str] | None = None) -> None:
    """Entry point for well-array-sim-export console script."""
    args = _build_export_partition_parser().parse_args(argv)
    _run_export_partition(args)


def _sim_kwargs(scenario, *, corrosion_year_yr: float | None = None) -> dict:
    return {
        "wall_profile": scenario.effective_wall_profile(corrosion_year_yr),
        "echo": scenario.echo,
        "inference": scenario.inference,
    }


def _add_corrosion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corrosion-year",
        type=float,
        default=None,
        metavar="YR",
        help="Evolve corrosion to this year and use WallProfile for acoustics (requires corrosion: in YAML)",
    )
    parser.add_argument(
        "--corrosion-snapshots",
        action="store_true",
        help="Export corrosion NPZ/PNG at each snapshot_years (no acoustic sim)",
    )


def _run_single_angle(
    scenario,
    angle_deg: float,
    out: Path,
    *,
    show_inferred: bool,
    corrosion_year_yr: float | None = None,
) -> None:
    theta_rad = math.radians(angle_deg)
    result = simulate_pulse_echo_2d(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        theta_rad,
        z_m=0.0,
        **_sim_kwargs(scenario, corrosion_year_yr=corrosion_year_yr),
    )

    one_way_us = result.ground_truth_distance_m / result.fluid_vp * 1e6
    scene_path = plot_packet_scene(
        scenario,
        result,
        Path(f"{out}_packet_scene.png"),
        t_s=one_way_us * 1e-6,
        show_inferred=show_inferred,
        show_ground_truth=True,
    )
    echo_path = plot_pulse_echo(
        result,
        Path(f"{out}_pulse_echo.png"),
        show_inferred=show_inferred,
        show_ground_truth=True,
    )
    range_path = plot_matched_filter_profile(
        result,
        Path(f"{out}_range_profile.png"),
        show_inferred=show_inferred,
        show_ground_truth=True,
    )
    npz_path = save_pulse_echo_npz(result, Path(f"{out}_pulse_echo.npz"))

    err_mm = result.error_mm
    print(f"Packet scene:  {scene_path}")
    print(f"Pulse echo:    {echo_path}")
    print(f"Range profile: {range_path}")
    print(f"NPZ:           {npz_path}")
    print(
        f"θ={angle_deg:.1f}° | inferred={result.inferred_distance_m*1000:.1f} mm | "
        f"true={result.ground_truth_distance_m*1000:.1f} mm | error={err_mm:.1f} mm"
    )


def _run_axial_scan(
    scenario,
    angle_step_deg: float | None,
    out: Path,
    *,
    show_inferred: bool,
    corrosion_year_yr: float | None = None,
) -> None:
    resolved_angle_step = scenario.angle_step_deg if angle_step_deg is None else angle_step_deg
    scan = simulate_axial_scan(
        scenario.pipe,
        scenario.fluid,
        scenario.steel,
        scenario.transducer,
        scenario.timing,
        scenario.z_stations(),
        angle_step_deg=resolved_angle_step,
        z_step_m=scenario.z_step_m,
        **_sim_kwargs(scenario, corrosion_year_yr=corrosion_year_yr),
    )
    cloud_path, map_path, npz_path = plot_axial_scan_exports(
        scan,
        scenario,
        out,
        show_inferred=show_inferred,
    )

    print(f"Axial point cloud: {cloud_path}")
    print(f"Axial radius map:  {map_path}")
    print(f"NPZ:                 {npz_path}")
    print(
        f"Stations: {len(scan.z_stations_m)} @ {scan.z_step_m * 1000:.0f} mm | "
        f"Angles: {len(scan.angles_deg)} @ {scan.angle_step_deg:g}° | "
        f"mean inferred={scan.inferred_distance_m.mean() * 1000:.1f} mm | "
        f"true={scan.wall_distance_m * 1000:.1f} mm | "
        f"max |error|={np.abs(scan.error_mm).max():.2f} mm"
    )


def _run_corrosion_snapshots(scenario, out: Path) -> None:
    if not scenario.has_corrosion():
        raise SystemExit("Scenario has no corrosion: block; cannot export snapshots")
    engine = scenario.build_corrosion_engine()
    snapshots = engine.run_snapshots()
    for snap in snapshots:
        paths = plot_corrosion_exports(snap, out)
        print(f"T={snap.time_yr:g} yr: 3D={paths[0]} map={paths[1]} npz={paths[2]}")


def _run_om_summary(args: argparse.Namespace) -> None:
    records = load_om_activity(args.csv)
    filtered = filter_records(
        records,
        integrity_only=args.integrity,
        has_od=args.has_od,
        province=args.province,
        company=args.company,
        search=args.search,
    )
    print(format_summary(summarize(filtered)))


def _run_om_export(args: argparse.Namespace) -> None:
    records = load_om_activity(args.csv)
    filtered = filter_records(
        records,
        integrity_only=args.integrity,
        has_od=args.has_od,
        province=args.province,
        company=args.company,
        search=args.search,
    )
    path = export_csv(filtered, args.out)
    print(f"Exported {len(filtered)} rows to {path}")


def _run_bc_categories(_args: argparse.Namespace) -> None:
    print(format_category_table())


def _run_bc_summary(args: argparse.Namespace) -> None:
    segments = load_bc_segments(args.geojson)
    print(format_bc_summary(summarize_bc_segments(segments)))


def _run_bc_list(args: argparse.Namespace) -> None:
    segments = load_bc_segments(args.geojson)
    print(format_segment_table(segments, limit=args.limit, line_type=args.line_type))


def _run_bc_show(args: argparse.Namespace) -> None:
    segment = get_segment_by_permit_id(args.permit_id, path=args.geojson)
    print(format_segment_detail(segment, max_length_m=args.max_length_m))


def _run_bc_run(args: argparse.Namespace) -> None:
    years = None
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    segment = get_segment_by_permit_id(args.permit_id, path=args.geojson)
    result = run_segment_study(
        args.permit_id,
        geojson_path=args.geojson,
        out_root=args.out,
        years=years,
        max_length_m=args.max_length_m,
        partition_length_m=args.partition_length_m,
        max_partitions=args.max_partitions,
        z_step_m=args.z_step_m,
        angle_step_deg=args.angle_step_deg,
        plot_waveforms=not args.no_plots,
        max_plot_partitions=0 if args.no_plots else args.max_plot_partitions,
        sample_z_m=args.sample_z_m,
        sample_theta_deg=args.sample_theta_deg,
    )
    print_study_report(result, segment, max_length_m=args.max_length_m)


def _run_bc_scenario(args: argparse.Namespace) -> None:
    segment = get_segment_by_permit_id(args.permit_id, path=args.geojson)
    path = write_scenario_from_bc_segment(
        segment,
        args.out,
        max_length_m=args.max_length_m,
    )
    scenario = load_internal_scenario(path)
    pipe_spec = spec_for_category(segment.category)
    sim_length_m = min(segment.length_m, args.max_length_m)
    print(f"Wrote scenario for BC permit {segment.permit_id} to {path}")
    print(
        f"{segment.line_type_desc} -> {pipe_spec.label} | "
        f"OD {pipe_spec.outside_diameter_mm:.0f} mm | "
        f"BC length {segment.length_m / 1000:.2f} km | sim window {sim_length_m:.1f} m"
    )
    print(
        f"Scan grid:   z_step_m={scenario.z_step_m:g} m, "
        f"angle_step_deg={scenario.angle_step_deg:g}°"
    )


def _add_bc_length_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-length-m",
        type=float,
        default=40.0,
        help="Simulate at most this many metres along the BC segment (default: 40)",
    )


def _add_bc_parser(subparsers: argparse._SubParsersAction) -> None:
    bc_parser = subparsers.add_parser(
        "bc",
        help="British Columbia pipeline GIS: browse segments, simulate UT, export bundles",
        description=(
            "Work with bundled BC pipeline GeoJSON (5,940 segments). "
            "Map a real segment to pipe geometry, simulate rotating UT scans along "
            "a short window, and export platform-ready observation bundles."
        ),
    )
    bc_parser.add_argument(
        "--geojson",
        type=Path,
        default=Path("data/raw/bc_pipeline_segments.geojson"),
        help="Path to BC pipeline GeoJSON (default: data/raw/...)",
    )
    bc_sub = bc_parser.add_subparsers(dest="bc_command", required=True)

    categories = bc_sub.add_parser("categories", help="Print assumed pipe specs by category")
    categories.set_defaults(func=_run_bc_categories)

    summary = bc_sub.add_parser("summary", help="Summarize BC segments by line type / category")
    summary.set_defaults(func=_run_bc_summary)

    listing = bc_sub.add_parser("list", help="List BC segments (sorted by length)")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument(
        "--line-type",
        type=str,
        default=None,
        help="Filter by LINE_TYPE_DESC, e.g. Transmission",
    )
    listing.set_defaults(func=_run_bc_list)

    show = bc_sub.add_parser("show", help="Show one BC segment and derived pipe category")
    show.add_argument(
        "--permit-id",
        type=int,
        required=True,
        help="BC segment ID (OG_PIPELINE_SEGMENT_PERMIT_ID)",
    )
    _add_bc_length_args(show)
    show.set_defaults(func=_run_bc_show)

    run = bc_sub.add_parser(
        "run",
        help="Simulate UT along a pipeline window; export bundles (partition × year) + optional PNGs",
    )
    run.add_argument(
        "--permit-id",
        type=int,
        required=True,
        help="BC segment ID (OG_PIPELINE_SEGMENT_PERMIT_ID; use 'bc list' to browse)",
    )
    run.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated observation years (default: corrosion snapshot_years in scenario)",
    )
    run.add_argument("--out", type=Path, default=Path("outputs/segment_study"))
    _add_bc_length_args(run)
    run.add_argument(
        "--partition-length-m",
        type=float,
        default=0.40,
        help="Axial partition size along the sim window (default: 0.40 m)",
    )
    run.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="Cap number of partitions exported (default: all partitions in sim window)",
    )
    run.add_argument(
        "--z-step-m",
        type=float,
        default=None,
        help="Axial station spacing (default: scenario scan.z_step_m)",
    )
    run.add_argument(
        "--angle-step-deg",
        type=float,
        default=None,
        help="Azimuth step for 360° sweep (default: scenario scan.angle_step_deg)",
    )
    run.add_argument("--no-plots", action="store_true", help="Skip PNG previews (bundles still include waveforms)")
    run.add_argument(
        "--max-plot-partitions",
        type=int,
        default=1,
        help="Generate PNG previews for the first N partitions per year (default: 1)",
    )
    run.add_argument("--sample-z-m", type=float, default=0.0)
    run.add_argument("--sample-theta-deg", type=float, default=45.0)
    run.set_defaults(func=_run_bc_run)

    scenario = bc_sub.add_parser("scenario", help="Build internal scenario YAML from one BC segment")
    scenario.add_argument(
        "--permit-id",
        type=int,
        required=True,
        help="BC segment ID (OG_PIPELINE_SEGMENT_PERMIT_ID)",
    )
    scenario.add_argument("--out", type=Path, default=Path("scenarios/bc_selected.yaml"))
    _add_bc_length_args(scenario)
    scenario.set_defaults(func=_run_bc_scenario)


def _add_om_parser(subparsers: argparse._SubParsersAction) -> None:
    om_parser = subparsers.add_parser("om", help="Analyze CER Operation & Maintenance CSV")
    om_parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/operation-and-maintenance-activity.csv"),
        help="Path to CER O&M CSV (default: data/raw/...)",
    )
    om_sub = om_parser.add_subparsers(dest="om_command", required=True)

    summary = om_sub.add_parser("summary", help="Print summary statistics")
    summary.add_argument("--integrity", action="store_true", help="Integrity dig = Yes only")
    summary.add_argument("--has-od", action="store_true", help="Rows with outside diameter only")
    summary.add_argument("--province", type=str, default=None)
    summary.add_argument("--company", type=str, default=None)
    summary.add_argument("--search", type=str, default=None)
    summary.set_defaults(func=_run_om_summary)

    export = om_sub.add_parser("export", help="Export filtered rows to CSV")
    export.add_argument("--integrity", action="store_true")
    export.add_argument("--has-od", action="store_true")
    export.add_argument("--province", type=str, default=None)
    export.add_argument("--company", type=str, default=None)
    export.add_argument("--search", type=str, default=None)
    export.add_argument("--out", type=Path, default=Path("outputs/om_filtered.csv"))
    export.set_defaults(func=_run_om_export)


def _add_sim_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("scenarios/internal_pipe_default.yaml"),
        help="Scenario YAML (pipe, fluid, scan grid, corrosion, inference)",
    )
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=None,
        help=f"Azimuth for a single shot in degrees (default: {DEFAULT_STEER_ANGLE_DEG:g})",
    )
    parser.add_argument(
        "--axial-scan",
        action="store_true",
        help="Sweep 360° at each axial station along the pipe (tool rotation + SAFT inference)",
    )
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=None,
        help="Azimuth step for --axial-scan (default: scenario scan.angle_step_deg)",
    )
    inferred_group = parser.add_mutually_exclusive_group()
    inferred_group.add_argument(
        "--show-inferred",
        dest="show_inferred",
        action="store_true",
        help="Show blind inference overlays on PNG plots (default)",
    )
    inferred_group.add_argument(
        "--hide-inferred",
        dest="show_inferred",
        action="store_false",
        help="Hide inference overlays on PNG plots (ground truth only)",
    )
    parser.set_defaults(show_inferred=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/radial_demo"),
        help="Output path prefix for PNG/NPZ files",
    )


def _build_sim_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a quick pulse-echo simulation from a scenario YAML: "
            "one ultrasound shot at one angle, or a full rotating tool sweep."
        ),
    )
    _add_sim_arguments(parser)
    return parser


def _run_sim_cli(argv: list[str] | None = None) -> None:
    args = _build_sim_parser().parse_args(argv)
    _run_sim(args)


def _normalize_cli_argv(argv: list[str]) -> list[str]:
    """Map legacy bare flags (well-array-sim --angle-deg 45) to the sim subcommand."""
    if not argv or argv[0] in KNOWN_COMMANDS:
        return argv
    if argv[0].startswith("-"):
        return ["sim", *argv]
    return argv


def _build_root_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="well-array-sim",
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )


def _configure_root_subparsers(subparsers: argparse._SubParsersAction) -> None:
    sim_parser = subparsers.add_parser(
        "sim",
        help="Quick pulse-echo demo from a scenario YAML",
        description=(
            "Simulate one ultrasound shot at one azimuth, or rotate through 360° at "
            "each axial station along the pipe. Writes PNG/NPZ previews."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_sim_arguments(sim_parser)
    _add_corrosion_args(sim_parser)
    sim_parser.set_defaults(func=_run_sim)

    export_parser = subparsers.add_parser(
        "export-partition",
        help="Export platform observation bundles (partition × year)",
        description=(
            "Run the full UT simulation for one or more 0.4 m pipe partitions and "
            "write manifest.json + parquets for acoustic-ndt-platform ingest."
        ),
    )
    _add_export_partition_arguments(export_parser)
    export_parser.set_defaults(func=_run_export_partition)

    _add_bc_parser(subparsers)
    _add_om_parser(subparsers)


def main() -> None:
    raw_argv = sys.argv[1:]

    if raw_argv and raw_argv[0] == "export-partition":
        main_export_partition(raw_argv[1:])
        return

    if not raw_argv or (len(raw_argv) == 1 and raw_argv[0] in ("-h", "--help")):
        _build_root_parser().print_help()
        return

    argv = _normalize_cli_argv(raw_argv)
    parser = _build_root_parser()
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _configure_root_subparsers(subparsers)

    args = parser.parse_args(argv)
    if args.command in {"om", "bc"}:
        if not hasattr(args, "func"):
            parser.error(f"{args.command} requires a subcommand")
    args.func(args)


def _run_sim(args: argparse.Namespace) -> None:
    scenario = load_internal_scenario(args.scenario)

    if getattr(args, "corrosion_snapshots", False):
        _run_corrosion_snapshots(scenario, args.out)
        return

    corrosion_year = getattr(args, "corrosion_year", None)
    if corrosion_year is not None and not scenario.has_corrosion():
        raise SystemExit("--corrosion-year requires a corrosion: block in the scenario YAML")

    if args.axial_scan:
        _run_axial_scan(
            scenario,
            args.angle_step_deg,
            args.out,
            show_inferred=args.show_inferred,
            corrosion_year_yr=corrosion_year,
        )
        return

    angle_deg = args.angle_deg
    if angle_deg is None:
        angle_deg = DEFAULT_STEER_ANGLE_DEG
    _run_single_angle(
        scenario,
        angle_deg,
        args.out,
        show_inferred=args.show_inferred,
        corrosion_year_yr=corrosion_year,
    )


if __name__ == "__main__":
    main()
