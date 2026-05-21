"""Tests for axial partition planning."""

from __future__ import annotations

import pytest

from well_array_sim.export.partitions import partition_plan


def test_partition_plan_single_chunk() -> None:
    slices = partition_plan(0.4, partition_length_m=0.4)
    assert len(slices) == 1
    assert slices[0].partition_index == 0
    assert slices[0].chainage_start_m == 0.0
    assert slices[0].axial_length_m == 0.4


def test_partition_plan_one_km() -> None:
    slices = partition_plan(1000.0, partition_length_m=0.4)
    assert len(slices) == 2500
    assert slices[-1].chainage_start_m == pytest.approx(999.6)
    assert slices[-1].axial_length_m == pytest.approx(0.4)


def test_partition_plan_respects_max_partitions() -> None:
    slices = partition_plan(10.0, partition_length_m=0.4, max_partitions=3)
    assert len(slices) == 3
    assert slices[-1].chainage_start_m + slices[-1].axial_length_m == pytest.approx(1.2)
