from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matplotlib.figure import Figure


@dataclass
class FigureLayers:
    """Track overlay artists so the GUI can toggle visibility without rebuilding."""

    fig: Figure
    inferred_artists: list[Any] = field(default_factory=list)
    ground_truth_artists: list[Any] = field(default_factory=list)

    def apply_overlays(self, *, show_inferred: bool, show_ground_truth: bool) -> None:
        for artist in self.inferred_artists:
            artist.set_visible(show_inferred)
        for artist in self.ground_truth_artists:
            artist.set_visible(show_ground_truth)

    @property
    def supports_overlay_toggle(self) -> bool:
        return bool(self.inferred_artists or self.ground_truth_artists)
