"""Axial partition planning for long pipeline segments."""

from __future__ import annotations

import math
from dataclasses import dataclass

from well_array_sim.export.schema import DEFAULT_AXIAL_LENGTH_M


@dataclass(frozen=True)
class PartitionSlice:
    partition_index: int
    chainage_start_m: float
    axial_length_m: float


def partition_plan(
    total_length_m: float,
    *,
    partition_length_m: float = DEFAULT_AXIAL_LENGTH_M,
    max_partitions: int | None = None,
) -> list[PartitionSlice]:
    """Split a sim window into fixed-length axial partitions."""
    total = max(float(total_length_m), float(partition_length_m))
    chunk = max(float(partition_length_m), 1e-6)
    n_full = int(math.ceil(total / chunk))
    n_partitions = n_full if max_partitions is None else min(n_full, max(1, int(max_partitions)))

    slices: list[PartitionSlice] = []
    start = 0.0
    for index in range(n_partitions):
        remaining = total - start
        length = min(chunk, remaining)
        slices.append(
            PartitionSlice(
                partition_index=index,
                chainage_start_m=start,
                axial_length_m=length,
            )
        )
        start += length
    return slices
