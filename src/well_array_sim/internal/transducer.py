from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from well_array_sim.core.pulse import gaussian_tone_burst, make_time_axis


@dataclass(frozen=True)
class PointTransducer:
    """Monostatic point transducer at the bore-axis origin in a 2D slice."""

    center_freq_hz: float
    bandwidth: float
    position_xy: tuple[float, float] = (0.0, 0.0)

    @property
    def x_m(self) -> float:
        return self.position_xy[0]

    @property
    def y_m(self) -> float:
        return self.position_xy[1]

    def transmit_pulse(self, timing: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Return (time_s, p_tx) for the configured tone burst."""
        time_s = make_time_axis(timing["t_end_us"], timing["dt_us"])
        pulse = gaussian_tone_burst(
            time_s,
            f0_hz=self.center_freq_hz,
            bandwidth=self.bandwidth,
        )
        return time_s, pulse
