"""Load BC pipeline segments and build simulator scenarios from line type."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from well_array_sim.io.pipe_categories import (
    REPO_ROOT,
    PipeCategory,
    apply_category_spec,
    category_from_bc_line_type,
    scenario_yaml_from_category,
    spec_for_category,
)

DEFAULT_BC_GEOJSON = REPO_ROOT / "data" / "raw" / "bc_pipeline_segments.geojson"
DEFAULT_CORROSION_TEMPLATE = REPO_ROOT / "scenarios" / "internal_pipe_corrosion_default.yaml"


@dataclass(frozen=True)
class BcPipelineSegment:
    permit_id: int
    object_id: int
    line_type_desc: str
    line_type: str
    feature_length_m: float | None
    physical_pipe_length_m: float | None
    proponent: str
    status: str
    project_number: str
    segment_number: int
    category: PipeCategory

    @property
    def length_m(self) -> float:
        for candidate in (self.feature_length_m, self.physical_pipe_length_m):
            if candidate is not None and candidate > 0:
                return float(candidate)
        return 0.4

    @classmethod
    def from_feature(cls, feature: dict[str, Any]) -> BcPipelineSegment:
        props = feature["properties"]
        line_type_desc = str(props.get("LINE_TYPE_DESC") or "").strip()
        return cls(
            permit_id=int(props["OG_PIPELINE_SEGMENT_PERMIT_ID"]),
            object_id=int(props["OBJECTID"]),
            line_type_desc=line_type_desc,
            line_type=str(props.get("LINE_TYPE") or ""),
            feature_length_m=_optional_float(props.get("FEATURE_LENGTH_M")),
            physical_pipe_length_m=_optional_float(props.get("PHYSICAL_PIPE_LENGTH")),
            proponent=str(props.get("PROPONENT") or ""),
            status=str(props.get("STATUS") or ""),
            project_number=str(props.get("PROJECT_NUMBER") or ""),
            segment_number=int(props.get("SEGMENT_NUMBER") or 0),
            category=category_from_bc_line_type(line_type_desc),
        )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_bc_segments(path: Path | str | None = None) -> list[BcPipelineSegment]:
    geojson_path = DEFAULT_BC_GEOJSON if path is None else Path(path)
    with geojson_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [BcPipelineSegment.from_feature(feature) for feature in payload["features"]]


def get_segment_by_permit_id(
    permit_id: int,
    *,
    segments: Iterable[BcPipelineSegment] | None = None,
    path: Path | str | None = None,
) -> BcPipelineSegment:
    items = list(segments) if segments is not None else load_bc_segments(path)
    matches = [segment for segment in items if segment.permit_id == permit_id]
    if not matches:
        raise KeyError(f"No BC segment with OG_PIPELINE_SEGMENT_PERMIT_ID={permit_id}")
    if len(matches) > 1:
        raise KeyError(f"Multiple BC segments share OG_PIPELINE_SEGMENT_PERMIT_ID={permit_id}")
    return matches[0]


def resolved_sim_length_m(segment: BcPipelineSegment, *, max_length_m: float | None = None) -> float:
    """Cap BC centre-line length to a sim-friendly window."""
    cap = 40.0 if max_length_m is None else float(max_length_m)
    return round(min(segment.length_m, max(cap, 0.4)), 3)


def scenario_yaml_from_bc_segment(
    segment: BcPipelineSegment,
    *,
    template_path: Path | None = None,
    max_length_m: float | None = None,
) -> dict[str, Any]:
    template = DEFAULT_CORROSION_TEMPLATE if template_path is None else Path(template_path)
    with template.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    spec = spec_for_category(segment.category)
    sim_length_m = resolved_sim_length_m(segment, max_length_m=max_length_m)
    data = apply_category_spec(data, spec, length_m=sim_length_m)
    inner_r = float(data["pipe"]["inner_radius_m"])
    data["inference"] = {
        "mode": "angular_saft",
        "r_min_m": round(max(0.01, inner_r * 0.65), 6),
        "r_max_m": round(inner_r * 1.35, 6),
        "r_step_m": 0.0005,
        "angular_window_deg": 15.0,
        "coherent_sum": True,
    }
    vp = float(data["medium"]["bore_fluid"]["vp"])
    t_end_us = max(float(data["timing"].get("t_end_us", 600.0)), 2.0 * inner_r * 1.35 / vp * 1e6 + 50.0)
    data["timing"]["t_end_us"] = round(t_end_us, 1)
    data["bc_source"] = {
        "permit_id": segment.permit_id,
        "object_id": segment.object_id,
        "line_type_desc": segment.line_type_desc,
        "line_type": segment.line_type,
        "feature_length_m": segment.feature_length_m,
        "physical_pipe_length_m": segment.physical_pipe_length_m,
        "proponent": segment.proponent,
        "status": segment.status,
        "project_number": segment.project_number,
        "segment_number": segment.segment_number,
        "pipe_category": segment.category.value,
        "bc_length_m": segment.length_m,
        "sim_length_m": sim_length_m,
    }
    return data


def format_segment_detail(segment: BcPipelineSegment, *, max_length_m: float | None = None) -> str:
    spec = spec_for_category(segment.category)
    sim_length_m = resolved_sim_length_m(segment, max_length_m=max_length_m)
    lines = [
        f"BC segment permit_id={segment.permit_id} (OBJECTID={segment.object_id})",
        f"  Line type:   {segment.line_type_desc} ({segment.line_type})",
        f"  Proponent:   {segment.proponent}",
        f"  Status:      {segment.status}",
        f"  BC length:   {segment.length_m / 1000:.3f} km ({segment.length_m:.1f} m)",
        f"  Category:    {spec.label} ({spec.category.value})",
        f"  Pipe OD:     {spec.outside_diameter_mm:.0f} mm ({spec.nps_label})",
        f"  Wall:        {spec.wall_thickness_mm:.1f} mm | SMYS {spec.smys_mpa:.0f} MPa | MOP {spec.mop_mpa:.0f} MPa",
        f"  Sim window:  {sim_length_m:.1f} m (first {sim_length_m:.1f} m of segment; adjust with --max-length-m)",
    ]
    return "\n".join(lines)


def format_segment_table(
    segments: Iterable[BcPipelineSegment],
    *,
    limit: int = 20,
    line_type: str | None = None,
) -> str:
    items = list(segments)
    if line_type:
        key = line_type.strip()
        items = [segment for segment in items if segment.line_type_desc == key]
    items.sort(key=lambda segment: segment.length_m, reverse=True)
    items = items[: max(1, int(limit))]
    lines = [
        " permit_id  line_type        length_km  category",
        " ---------- ---------------- ---------- ------------------",
    ]
    for segment in items:
        lines.append(
            f" {segment.permit_id:>10}  {segment.line_type_desc:<16} "
            f"{segment.length_m / 1000:>9.2f}  {segment.category.value}"
        )
    return "\n".join(lines)


def write_scenario_from_bc_segment(
    segment: BcPipelineSegment,
    path: Path,
    *,
    template_path: Path | None = None,
    max_length_m: float | None = None,
) -> Path:
    data = scenario_yaml_from_bc_segment(
        segment,
        template_path=template_path,
        max_length_m=max_length_m,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return path


def summarize_bc_segments(segments: Iterable[BcPipelineSegment]) -> dict[str, Any]:
    items = list(segments)
    by_line_type = Counter(segment.line_type_desc for segment in items)
    by_category = Counter(segment.category.value for segment in items)
    lengths = [segment.length_m for segment in items]
    return {
        "count": len(items),
        "by_line_type": dict(by_line_type.most_common()),
        "by_category": dict(by_category.most_common()),
        "length_m_total": sum(lengths),
        "length_m_median": sorted(lengths)[len(lengths) // 2] if lengths else 0.0,
    }


def format_bc_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"BC pipeline segments: {summary['count']:,}",
        f"Total centre-line length: {summary['length_m_total'] / 1000:.1f} km",
        f"Median segment length: {summary['length_m_median']:.1f} m",
        "",
        "By category:",
    ]
    for category, count in summary.get("by_category", {}).items():
        spec = spec_for_category(category)
        lines.append(
            f"  {spec.label}: {count:,} "
            f"(OD {spec.outside_diameter_mm:.0f} mm, wall {spec.wall_thickness_mm:.1f} mm, "
            f"SMYS {spec.smys_mpa:.0f} MPa, MOP {spec.mop_mpa:.0f} MPa)"
        )
    lines.append("")
    lines.append("By BC line type:")
    for line_type, count in summary.get("by_line_type", {}).items():
        lines.append(f"  {line_type}: {count:,}")
    return "\n".join(lines)
