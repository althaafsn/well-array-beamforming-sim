from __future__ import annotations

import pytest

from well_array_sim.io.om_activity import (
    filter_records,
    load_om_activity,
    od_mm_to_nps_label,
    summarize,
)


@pytest.fixture(scope="module")
def all_records():
    return load_om_activity()


def test_load_om_activity_count(all_records) -> None:
    assert len(all_records) == 9469


def test_filter_integrity_and_od(all_records) -> None:
    integrity = filter_records(all_records, integrity_only=True)
    assert len(integrity) == 7134
    with_od = filter_records(integrity, has_od=True)
    assert len(with_od) == 386
    assert all(r.pipeline_outside_diameter_mm is not None for r in with_od)


def test_summarize_has_od_stats(all_records) -> None:
    subset = filter_records(all_records, has_od=True)
    summary = summarize(subset)
    assert summary["count"] == 634
    assert summary["od_mm_median"] == pytest.approx(609.6, rel=1e-3)


def test_od_mm_to_nps_label() -> None:
    assert "24" in od_mm_to_nps_label(609.6)
    assert "30" in od_mm_to_nps_label(762.0)
