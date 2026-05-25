"""
Angular synthetic-aperture SAFT for monostatic bore-center pulse-echo.

Combines waveforms from a full 360° steer sweep at one axial station into a
polar focus image I(r, φ), then extracts blind radius estimates per direction.

Must not use ground-truth wall geometry — only waveforms, TX template, fluid
speed, and blind range / angular windows from scenario config.
"""

from __future__ import annotations

import numpy as np

from well_array_sim.internal.imaging import inference_range_grid, parabolic_peak_offset
from well_array_sim.internal.scenario import InferenceConfig


def angular_apodization_weights(
    image_angles_deg: np.ndarray,
    shot_angles_deg: np.ndarray,
    *,
    beam_width_deg: float,
) -> np.ndarray:
    """
    Weight matrix W[image_phi, shot_theta] from Gaussian apodization in angle.

    ``beam_width_deg`` is treated as the full width at half maximum (FWHM).
    """
    if beam_width_deg <= 0.0:
        raise ValueError("beam_width_deg must be positive")
    sigma = float(beam_width_deg) / 2.355
    image = np.asarray(image_angles_deg, dtype=float)
    shots = np.asarray(shot_angles_deg, dtype=float)
    delta = image[:, None] - shots[None, :]
    delta = (delta + 180.0) % 360.0 - 180.0
    return np.exp(-0.5 * (delta / sigma) ** 2)


def _sample_dt_s(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return 0.5e-6
    return float(time_s[1] - time_s[0])


def _correlation_profiles(
    p_rx: np.ndarray,
    p_template: np.ndarray,
) -> np.ndarray:
    """Return full cross-correlation trace for each shot: shape (n_theta, n_corr)."""
    p_rx = np.asarray(p_rx, dtype=float)
    p_template = np.asarray(p_template, dtype=float)
    n_theta = p_rx.shape[0]
    n_corr = p_rx.shape[1] + len(p_template) - 1
    out = np.zeros((n_theta, n_corr), dtype=float)
    for it in range(n_theta):
        out[it] = np.correlate(p_rx[it], p_template, mode="full")
    return out


def _lag_window(
    r_grid_m: np.ndarray,
    fluid_vp: float,
    dt_s: float,
    *,
    n_rx: int,
) -> np.ndarray:
    lag_min = int(round(2.0 * float(r_grid_m[0]) / fluid_vp / dt_s))
    lag_max = int(round(2.0 * float(r_grid_m[-1]) / fluid_vp / dt_s))
    lag_max = min(lag_max, n_rx - 1)
    lag_min = max(0, lag_min)
    if lag_max <= lag_min:
        return np.array([lag_min], dtype=int)
    return np.arange(lag_min, lag_max + 1, dtype=int)


def _combined_power_vs_lag(
    corrs: np.ndarray,
    template_offset: int,
    lags: np.ndarray,
    weights: np.ndarray,
    *,
    coherent_sum: bool,
) -> np.ndarray:
    """Return focus power vs lag for each image angle: shape (n_phi, n_lags)."""
    corr_seg = corrs[:, template_offset + lags]
    if coherent_sum:
        combined = weights @ corr_seg
        return combined * combined
    return weights @ (corr_seg * corr_seg)


def _radius_from_lag_peak(
    lags: np.ndarray,
    power: np.ndarray,
    *,
    fluid_vp: float,
    dt_s: float,
) -> float:
    peak_rel = int(np.argmax(power))
    peak_lag = float(lags[peak_rel])
    if 0 < peak_rel < len(power) - 1:
        peak_lag += parabolic_peak_offset(
            float(power[peak_rel - 1]),
            float(power[peak_rel]),
            float(power[peak_rel + 1]),
        )
    return float(fluid_vp * peak_lag * dt_s / 2.0)


def polar_saft_focus_image(
    p_rx: np.ndarray,
    p_template: np.ndarray,
    time_s: np.ndarray,
    angles_deg: np.ndarray,
    r_grid_m: np.ndarray,
    fluid_vp: float,
    *,
    beam_width_deg: float,
    coherent_sum: bool = True,
    corrs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build polar focus image I(r, φ) by angular SAFT migration.

    Power is accumulated on the lag axis per image angle, then interpolated onto
    the blind radius grid for visualization.
    """
    p_rx = np.asarray(p_rx, dtype=float)
    p_template = np.asarray(p_template, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    r_grid_m = np.asarray(r_grid_m, dtype=float)

    if p_rx.ndim != 2:
        raise ValueError("p_rx must have shape (n_theta, n_samples)")
    n_theta, n_rx = p_rx.shape
    if len(angles_deg) != n_theta:
        raise ValueError("angles_deg length must match p_rx n_theta")

    dt_s = _sample_dt_s(time_s)
    template_offset = len(p_template) - 1
    if corrs is None:
        corrs = _correlation_profiles(p_rx, p_template)
    lags = _lag_window(r_grid_m, fluid_vp, dt_s, n_rx=n_rx)
    weights = angular_apodization_weights(
        angles_deg,
        angles_deg,
        beam_width_deg=beam_width_deg,
    )
    power_vs_lag = _combined_power_vs_lag(
        corrs,
        template_offset,
        lags,
        weights,
        coherent_sum=coherent_sum,
    )

    r_from_lag = lags.astype(float) * dt_s * fluid_vp / 2.0
    n_phi = len(angles_deg)
    n_r = len(r_grid_m)
    image = np.zeros((n_r, n_phi), dtype=float)
    for iphi in range(n_phi):
        image[:, iphi] = np.interp(
            r_grid_m,
            r_from_lag,
            power_vs_lag[iphi],
            left=0.0,
            right=0.0,
        )
    return image


def infer_radius_from_focus(
    focus_image: np.ndarray,
    r_grid_m: np.ndarray,
) -> np.ndarray:
    """Extract blind radius estimate per image column with parabolic refinement."""
    image = np.asarray(focus_image, dtype=float)
    r_grid_m = np.asarray(r_grid_m, dtype=float)
    if image.ndim != 2:
        raise ValueError("focus_image must be 2D (n_r, n_phi)")
    if image.shape[0] != len(r_grid_m):
        raise ValueError("focus_image n_r must match r_grid_m length")

    n_r, n_phi = image.shape
    inferred = np.zeros(n_phi, dtype=float)
    for iphi in range(n_phi):
        metric = image[:, iphi]
        peak_rel = int(np.argmax(metric))
        if n_r <= 1:
            inferred[iphi] = float(r_grid_m[peak_rel])
            continue

        r_hat = float(r_grid_m[peak_rel])
        if 0 < peak_rel < n_r - 1:
            offset = parabolic_peak_offset(
                float(metric[peak_rel - 1]),
                float(metric[peak_rel]),
                float(metric[peak_rel + 1]),
            )
            step = float(r_grid_m[1] - r_grid_m[0]) if n_r > 1 else 0.0
            r_hat = float(r_grid_m[peak_rel] + offset * step)
        inferred[iphi] = r_hat
    return inferred


def infer_axial_slice_saft(
    p_rx_slice: np.ndarray,
    p_tx: np.ndarray,
    time_s: np.ndarray,
    angles_deg: np.ndarray,
    fluid_vp: float,
    inference: InferenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Angular SAFT inference for one axial station.

    Peaks are found on the migrated lag axis (same principle as matched-filter
    SAFT), then converted to radius. Returns (inferred_r[n_theta], focus_image).
    """
    p_rx_slice = np.asarray(p_rx_slice, dtype=float)
    p_tx = np.asarray(p_tx, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)

    r_grid = inference_range_grid(
        r_min_m=inference.r_min_m,
        r_max_m=inference.r_max_m,
        r_step_m=inference.r_step_m,
    )
    dt_s = _sample_dt_s(time_s)
    template_offset = len(p_tx) - 1
    corrs = _correlation_profiles(p_rx_slice, p_tx)
    lags = _lag_window(r_grid, fluid_vp, dt_s, n_rx=p_rx_slice.shape[1])
    weights = angular_apodization_weights(
        angles_deg,
        angles_deg,
        beam_width_deg=inference.angular_window_deg,
    )
    power_vs_lag = _combined_power_vs_lag(
        corrs,
        template_offset,
        lags,
        weights,
        coherent_sum=inference.coherent_sum,
    )

    inferred = np.array(
        [
            _radius_from_lag_peak(
                lags,
                power_vs_lag[iphi],
                fluid_vp=fluid_vp,
                dt_s=dt_s,
            )
            for iphi in range(len(angles_deg))
        ],
        dtype=float,
    )
    focus = polar_saft_focus_image(
        p_rx_slice,
        p_tx,
        time_s,
        angles_deg,
        r_grid,
        fluid_vp,
        beam_width_deg=inference.angular_window_deg,
        coherent_sum=inference.coherent_sum,
        corrs=corrs,
    )
    return inferred, focus
