from __future__ import annotations

import pytest

from well_array_sim.internal.scenario import load_internal_scenario
from well_array_sim.io.bc_pipelines import (
    get_segment_by_permit_id,
    load_bc_segments,
    scenario_yaml_from_bc_segment,
    summarize_bc_segments,
)
from well_array_sim.io.pipe_categories import (
    PipeCategory,
    category_from_bc_line_type,
    inner_radius_from_od_mm,
    scenario_yaml_from_category,
    spec_for_category,
)


def test_category_specs_match_user_table() -> None:
    gathering = spec_for_category(PipeCategory.GATHERING_FLOWLINE)
    assert gathering.outside_diameter_mm == 168.0
    assert gathering.wall_thickness_mm == pytest.approx(4.8)
    assert gathering.smys_mpa == 359.0
    assert gathering.mop_mpa == 3.0

    feeder = spec_for_category(PipeCategory.MID_SIZE_FEEDER)
    assert feeder.outside_diameter_mm == 324.0
    assert feeder.wall_thickness_mm == pytest.approx(7.9)
    assert feeder.smys_mpa == 414.0
    assert feeder.mop_mpa == 6.0

    transmission = spec_for_category(PipeCategory.MAJOR_TRANSMISSION)
    assert transmission.outside_diameter_mm == 762.0
    assert transmission.wall_thickness_mm == pytest.approx(13.0)
    assert transmission.smys_mpa == 483.0
    assert transmission.mop_mpa == 8.0


@pytest.mark.parametrize(
    ("line_type", "category"),
    [
        ("Gathering", PipeCategory.GATHERING_FLOWLINE),
        ("Flow", PipeCategory.GATHERING_FLOWLINE),
        ("Intermediate", PipeCategory.MID_SIZE_FEEDER),
        ("Fuel Gas", PipeCategory.MID_SIZE_FEEDER),
        ("Distribution", PipeCategory.MID_SIZE_FEEDER),
        ("Injection", PipeCategory.MID_SIZE_FEEDER),
        ("Instrumentation", PipeCategory.MID_SIZE_FEEDER),
        ("Transmission", PipeCategory.MAJOR_TRANSMISSION),
    ],
)
def test_bc_line_type_mapping(line_type: str, category: PipeCategory) -> None:
    assert category_from_bc_line_type(line_type) is category


def test_inner_radius_from_od_mm() -> None:
    inner = inner_radius_from_od_mm(762.0, wall_thickness_m=0.013)
    assert inner == pytest.approx(0.368, rel=1e-3)


def test_scenario_yaml_from_category_includes_ops_and_materials() -> None:
    data = scenario_yaml_from_category(PipeCategory.MAJOR_TRANSMISSION, length_m=12.5)
    assert data["pipe"]["category"] == "major_transmission"
    assert data["pipe"]["outside_diameter_mm"] == 762.0
    assert data["pipe"]["length_m"] == pytest.approx(12.5)
    assert data["operations"]["mop_mpa"] == 8.0
    assert data["materials"]["steel"]["smys_mpa"] == 483.0


def test_bc_segments_all_map_to_categories() -> None:
    segments = load_bc_segments()
    summary = summarize_bc_segments(segments)
    assert summary["count"] == 5940
    assert set(summary["by_line_type"]) == {
        "Gathering",
        "Flow",
        "Intermediate",
        "Fuel Gas",
        "Distribution",
        "Injection",
        "Instrumentation",
        "Transmission",
    }
    assert summary["by_category"]["gathering_flowline"] == 3439
    assert summary["by_category"]["major_transmission"] == 1392
    assert summary["by_category"]["mid_size_feeder"] == 1109


def test_scenario_from_bc_transmission_segment(tmp_path) -> None:
    from well_array_sim.io.bc_pipelines import write_scenario_from_bc_segment

    segment = get_segment_by_permit_id(13)
    assert segment.line_type_desc == "Transmission"
    data = scenario_yaml_from_bc_segment(segment)
    assert data["pipe"]["outside_diameter_mm"] == 762.0
    assert data["bc_source"]["permit_id"] == 13

    scenario_path = tmp_path / "bc.yaml"
    write_scenario_from_bc_segment(segment, scenario_path)
    loaded = load_internal_scenario(scenario_path)
    assert loaded.pipe.inner_radius_m == pytest.approx(0.368, rel=1e-3)
    assert loaded.pipe.wall_thickness_m == pytest.approx(0.013)
