"""
Blind range imaging (SAFT / matched-filter migration) for monostatic pulse-echo.

This module must not use ground-truth wall geometry — only received waveforms,
TX template, fluid speed, and a blind range grid from scenario config.
"""

from __future__ import annotations

import numpy as np


def inference_range_grid(
    *,
    r_min_m: float,
    r_max_m: float,
    r_step_m: float,
) -> np.ndarray:
    if r_step_m <= 0:
        raise ValueError("r_step_m must be positive")
    if r_max_m <= r_min_m:
        raise ValueError("r_max_m must exceed r_min_m")
    return np.arange(r_min_m, r_max_m + 0.5 * r_step_m, r_step_m, dtype=float)


def _parabolic_peak_offset(y0: float, y1: float, y2: float) -> float:
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) <= 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))


def saft_range_profile(
    p_rx: np.ndarray,
    p_template: np.ndarray,
    time_s: np.ndarray,
    r_grid_m: np.ndarray,
    fluid_vp: float,
) -> tuple[np.ndarray, float]:
    """
    Blind range migration and estimate.

    Cross-correlates RX with the TX tone burst, builds I(r) on the blind grid by
    interpolating |corr|² vs range, and estimates radius from the correlation
    peak lag (with parabolic sub-sample refinement).
    """
    p_rx = np.asarray(p_rx, dtype=float)
    p_template = np.asarray(p_template, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    r_grid_m = np.asarray(r_grid_m, dtype=float)

    if len(time_s) < 2:
        dt_s = 0.5e-6
    else:
        dt_s = float(time_s[1] - time_s[0])

    corr = np.correlate(p_rx, p_template, mode="full")
    template_offset = len(p_template) - 1

    lag_min = int(round(2.0 * float(r_grid_m[0]) / fluid_vp / dt_s))
    lag_max = int(round(2.0 * float(r_grid_m[-1]) / fluid_vp / dt_s))
    lag_max = min(lag_max, len(p_rx) - 1)
    lag_min = max(0, lag_min)
    if lag_max <= lag_min:
        return np.zeros(len(r_grid_m), dtype=float), float(r_grid_m[0])

    lags = np.arange(lag_min, lag_max + 1, dtype=int)
    corr_seg = corr[template_offset + lags]
    power = corr_seg * corr_seg
    peak_rel = int(np.argmax(power))
    peak_lag = float(lags[peak_rel])

    if 0 < peak_rel < len(power) - 1:
        peak_lag += _parabolic_peak_offset(
            float(power[peak_rel - 1]),
            float(power[peak_rel]),
            float(power[peak_rel + 1]),
        )

    r_hat = fluid_vp * peak_lag * dt_s / 2.0

    r_from_lag = lags.astype(float) * dt_s * fluid_vp / 2.0
    image = np.interp(r_grid_m, r_from_lag, power, left=0.0, right=0.0)

    return image, float(r_hat)
