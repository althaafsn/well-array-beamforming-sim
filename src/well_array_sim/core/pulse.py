from __future__ import annotations

import numpy as np


def gaussian_envelope(
    time_s: np.ndarray,
    f0_hz: float,
    bandwidth: float,
    tp0_s: float = 8e-6,
) -> np.ndarray:
    """Gaussian envelope matching dataset pulse definition (unipolar, peak = 1)."""
    alpha = (np.pi * bandwidth * f0_hz) ** 2 / (4.0 * abs(np.log(0.5)))
    exponent = np.clip(-alpha * (time_s - tp0_s) ** 2, -700.0, 0.0)
    return np.exp(exponent)


def gaussian_tone_burst(
    time_s: np.ndarray,
    f0_hz: float,
    bandwidth: float,
    tp0_s: float = 8e-6,
) -> np.ndarray:
    """Gaussian-modulated sine pulse matching dataset pulse definition."""
    envelope = gaussian_envelope(time_s, f0_hz, bandwidth, tp0_s)
    carrier = np.sin(2 * np.pi * f0_hz * (time_s - tp0_s))
    pulse = envelope * carrier
    peak = np.max(np.abs(pulse))
    if peak > 0:
        pulse = pulse / peak
    return pulse


def make_time_axis(t_end_us: float, dt_us: float) -> np.ndarray:
    count = int(round(t_end_us / dt_us)) + 1
    return np.arange(count) * dt_us * 1e-6
