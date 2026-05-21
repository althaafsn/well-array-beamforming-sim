"""Pipe category defaults for BC line types and scenario geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIO_TEMPLATE = REPO_ROOT / "scenarios" / "internal_pipe_default.yaml"


class PipeCategory(str, Enum):
    GATHERING_FLOWLINE = "gathering_flowline"
    MID_SIZE_FEEDER = "mid_size_feeder"
    MAJOR_TRANSMISSION = "major_transmission"


@dataclass(frozen=True)
class PipeCategorySpec:
    category: PipeCategory
    label: str
    outside_diameter_mm: float
    wall_thickness_mm: float
    smys_mpa: float
    mop_mpa: float
    bc_line_types: tuple[str, ...]

    @property
    def wall_thickness_m(self) -> float:
        return self.wall_thickness_mm / 1000.0

    @property
    def inner_radius_m(self) -> float:
        return inner_radius_from_od_mm(
            self.outside_diameter_mm,
            wall_thickness_m=self.wall_thickness_m,
        )

    @property
    def nps_label(self) -> str:
        if self.outside_diameter_mm == 168.0:
            return 'NPS 6"'
        if self.outside_diameter_mm == 324.0:
            return 'NPS 12"'
        if self.outside_diameter_mm == 762.0:
            return 'NPS 30"'
        nps = self.outside_diameter_mm / 25.4
        return f'~NPS {nps:.1f}"'


PIPE_CATEGORY_SPECS: dict[PipeCategory, PipeCategorySpec] = {
    PipeCategory.GATHERING_FLOWLINE: PipeCategorySpec(
        category=PipeCategory.GATHERING_FLOWLINE,
        label="Gathering / Flowline",
        outside_diameter_mm=168.0,
        wall_thickness_mm=4.8,
        smys_mpa=359.0,
        mop_mpa=3.0,
        bc_line_types=("Gathering", "Flow"),
    ),
    PipeCategory.MID_SIZE_FEEDER: PipeCategorySpec(
        category=PipeCategory.MID_SIZE_FEEDER,
        label="Mid-Size Feeder",
        outside_diameter_mm=324.0,
        wall_thickness_mm=7.9,
        smys_mpa=414.0,
        mop_mpa=6.0,
        bc_line_types=("Intermediate", "Fuel Gas", "Distribution", "Injection", "Instrumentation"),
    ),
    PipeCategory.MAJOR_TRANSMISSION: PipeCategorySpec(
        category=PipeCategory.MAJOR_TRANSMISSION,
        label="Major Transmission",
        outside_diameter_mm=762.0,
        wall_thickness_mm=13.0,
        smys_mpa=483.0,
        mop_mpa=8.0,
        bc_line_types=("Transmission",),
    ),
}

BC_LINE_TYPE_TO_CATEGORY: dict[str, PipeCategory] = {
    line_type: spec.category
    for spec in PIPE_CATEGORY_SPECS.values()
    for line_type in spec.bc_line_types
}


def inner_radius_from_od_mm(od_mm: float, *, wall_thickness_m: float) -> float:
    inner_radius_m = od_mm / 2000.0 - wall_thickness_m
    if inner_radius_m <= 0:
        raise ValueError(f"OD {od_mm} mm too small for wall thickness {wall_thickness_m * 1000:.1f} mm")
    return inner_radius_m


def category_from_bc_line_type(line_type_desc: str) -> PipeCategory:
    key = (line_type_desc or "").strip()
    try:
        return BC_LINE_TYPE_TO_CATEGORY[key]
    except KeyError as exc:
        known = ", ".join(sorted(BC_LINE_TYPE_TO_CATEGORY))
        raise ValueError(f"Unknown BC line type {line_type_desc!r}; expected one of: {known}") from exc


def spec_for_category(category: PipeCategory | str) -> PipeCategorySpec:
    if isinstance(category, str):
        category = PipeCategory(category)
    return PIPE_CATEGORY_SPECS[category]


def apply_category_spec(
    data: dict[str, Any],
    spec: PipeCategorySpec,
    *,
    length_m: float | None = None,
) -> dict[str, Any]:
    """Fill pipe geometry and operating/material defaults from a category spec."""
    pipe = data.setdefault("pipe", {})
    materials = data.setdefault("materials", {})
    steel = materials.setdefault("steel", {"rho": 7850, "vp": 5778})
    operations = data.setdefault("operations", {})

    resolved_length = length_m if length_m is not None else pipe.get("length_m", 0.4)
    resolved_length = max(float(resolved_length), 0.4)

    pipe["category"] = spec.category.value
    pipe["outside_diameter_mm"] = spec.outside_diameter_mm
    pipe["inner_radius_m"] = round(spec.inner_radius_m, 6)
    pipe["wall_thickness_m"] = spec.wall_thickness_m
    pipe["length_m"] = round(resolved_length, 3)

    steel["smys_mpa"] = spec.smys_mpa
    operations["mop_mpa"] = spec.mop_mpa

    data["pipe_category"] = {
        "category": spec.category.value,
        "label": spec.label,
        "outside_diameter_mm": spec.outside_diameter_mm,
        "wall_thickness_mm": spec.wall_thickness_mm,
        "smys_mpa": spec.smys_mpa,
        "mop_mpa": spec.mop_mpa,
        "nps_label": spec.nps_label,
        "bc_line_types": list(spec.bc_line_types),
    }
    return data


def scenario_yaml_from_category(
    category: PipeCategory | str,
    *,
    length_m: float | None = None,
    template_path: Path | None = None,
) -> dict[str, Any]:
    template = DEFAULT_SCENARIO_TEMPLATE if template_path is None else Path(template_path)
    with template.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    spec = spec_for_category(category)
    return apply_category_spec(data, spec, length_m=length_m)


def format_category_table() -> str:
    lines = [
        "Pipe category defaults (OD, wall, SMYS, MOP assumed from category):",
        "",
    ]
    for spec in PIPE_CATEGORY_SPECS.values():
        lines.append(f"- {spec.label} ({spec.category.value})")
        lines.append(
            f"  OD {spec.outside_diameter_mm:.0f} mm ({spec.nps_label}), "
            f"wall {spec.wall_thickness_mm:.1f} mm, "
            f"SMYS {spec.smys_mpa:.0f} MPa, MOP {spec.mop_mpa:.0f} MPa"
        )
        lines.append(f"  BC line types: {', '.join(spec.bc_line_types)}")
    return "\n".join(lines)
