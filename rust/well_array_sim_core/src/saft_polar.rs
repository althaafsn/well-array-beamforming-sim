use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub fn parabolic_peak_offset(y0: f64, y1: f64, y2: f64) -> f64 {
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() <= 1e-12 {
        return 0.0;
    }
    (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
}

/// Reference implementation kept for parity debugging (FFT path is production).
#[allow(dead_code)]
pub fn correlate_full_naive(a: &[f64], b: &[f64]) -> Vec<f64> {
    let b_rev: Vec<f64> = b.iter().copied().rev().collect();
    let n = a.len() + b.len().saturating_sub(1);
    let mut out = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b_rev.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Full linear cross-correlation: correlate(a, b) == convolve(a, reverse(b)).
pub fn correlate_full_fft(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len().saturating_sub(1);
    if n == 0 {
        return Vec::new();
    }
    let fft_len = n.next_power_of_two().max(2);
    let mut buf_a: Vec<Complex<f64>> = a
        .iter()
        .map(|&v| Complex { re: v, im: 0.0 })
        .collect();
    buf_a.resize(fft_len, Complex { re: 0.0, im: 0.0 });
    let b_rev: Vec<f64> = b.iter().copied().rev().collect();
    let mut buf_b: Vec<Complex<f64>> = b_rev
        .iter()
        .map(|&v| Complex { re: v, im: 0.0 })
        .collect();
    buf_b.resize(fft_len, Complex { re: 0.0, im: 0.0 });

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);
    fft.process(&mut buf_a);
    fft.process(&mut buf_b);
    for i in 0..fft_len {
        buf_a[i] = buf_a[i] * buf_b[i];
    }
    ifft.process(&mut buf_a);
    let scale = 1.0 / fft_len as f64;
    buf_a.iter().take(n).map(|c| c.re * scale).collect()
}

pub fn correlate_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    correlate_full_fft(a, b)
}

pub fn inference_range_grid(r_min_m: f64, r_max_m: f64, r_step_m: f64) -> Vec<f64> {
    let mut grid = Vec::new();
    let mut r = r_min_m;
    while r <= r_max_m + 0.5 * r_step_m {
        grid.push(r);
        r += r_step_m;
    }
    grid
}

fn sample_dt_s(time_s: &[f64]) -> f64 {
    if time_s.len() < 2 {
        0.5e-6
    } else {
        time_s[1] - time_s[0]
    }
}

pub fn angular_apodization_weights(
    image_angles_deg: &[f64],
    shot_angles_deg: &[f64],
    beam_width_deg: f64,
) -> Vec<f64> {
    let sigma = beam_width_deg / 2.355;
    let n_phi = image_angles_deg.len();
    let n_theta = shot_angles_deg.len();
    let mut weights = vec![0.0; n_phi * n_theta];
    for (iphi, &image) in image_angles_deg.iter().enumerate() {
        for (it, &shot) in shot_angles_deg.iter().enumerate() {
            let mut delta = image - shot;
            delta = ((delta + 180.0) % 360.0) - 180.0;
            let w = (-0.5 * (delta / sigma).powi(2)).exp();
            weights[iphi * n_theta + it] = w;
        }
    }
    weights
}

pub fn correlation_profiles(
    p_rx: &[f64],
    n_theta: usize,
    n_samples: usize,
    p_template: &[f64],
) -> Vec<f64> {
    let n_corr = n_samples + p_template.len() - 1;
    let mut out = vec![0.0; n_theta * n_corr];
    for it in 0..n_theta {
        let row = &p_rx[it * n_samples..(it + 1) * n_samples];
        let corr = correlate_full(row, p_template);
        out[it * n_corr..(it + 1) * n_corr].copy_from_slice(&corr);
    }
    out
}

pub fn lag_window(r_grid_m: &[f64], fluid_vp: f64, dt_s: f64, n_rx: usize) -> Vec<usize> {
    let lag_min = (2.0 * r_grid_m[0] / fluid_vp / dt_s).round() as isize;
    let lag_max = (2.0 * r_grid_m[r_grid_m.len() - 1] / fluid_vp / dt_s).round() as isize;
    let lag_max = lag_max.min((n_rx - 1) as isize) as usize;
    let lag_min = lag_min.max(0) as usize;
    if lag_max <= lag_min {
        return vec![lag_min];
    }
    (lag_min..=lag_max).collect()
}

pub fn combined_power_vs_lag(
    corrs: &[f64],
    n_theta: usize,
    n_corr: usize,
    template_offset: usize,
    lags: &[usize],
    weights: &[f64],
    n_phi: usize,
    coherent_sum: bool,
) -> Vec<f64> {
    let n_lags = lags.len();
    let mut power = vec![0.0; n_phi * n_lags];
    for iphi in 0..n_phi {
        for (ilag, &lag) in lags.iter().enumerate() {
            let idx = template_offset + lag;
            let mut sum = 0.0;
            for it in 0..n_theta {
                let corr_val = corrs[it * n_corr + idx];
                let w = weights[iphi * n_theta + it];
                if coherent_sum {
                    sum += w * corr_val;
                } else {
                    sum += w * corr_val * corr_val;
                }
            }
            power[iphi * n_lags + ilag] = if coherent_sum { sum * sum } else { sum };
        }
    }
    power
}

pub fn radius_from_lag_peak(lags: &[usize], power: &[f64], fluid_vp: f64, dt_s: f64) -> f64 {
    let mut peak_rel = 0usize;
    let mut peak_val = f64::NEG_INFINITY;
    for (i, &p) in power.iter().enumerate() {
        if p > peak_val {
            peak_val = p;
            peak_rel = i;
        }
    }
    let mut peak_lag = lags[peak_rel] as f64;
    if 0 < peak_rel && peak_rel < power.len() - 1 {
        peak_lag += parabolic_peak_offset(
            power[peak_rel - 1],
            power[peak_rel],
            power[peak_rel + 1],
        );
    }
    fluid_vp * peak_lag * dt_s / 2.0
}

pub struct SaftSliceContext {
    pub dt_s: f64,
    pub template_offset: usize,
    pub n_corr: usize,
    pub lags: Vec<usize>,
    pub weights: Vec<f64>,
    pub n_phi: usize,
}

pub fn build_saft_slice_context(
    p_tx: &[f64],
    time_s: &[f64],
    angles_deg: &[f64],
    fluid_vp: f64,
    r_min_m: f64,
    r_max_m: f64,
    r_step_m: f64,
    angular_window_deg: f64,
    n_samples: usize,
) -> SaftSliceContext {
    let r_grid = inference_range_grid(r_min_m, r_max_m, r_step_m);
    let dt_s = sample_dt_s(time_s);
    let template_offset = p_tx.len() - 1;
    let n_corr = n_samples + p_tx.len() - 1;
    let lags = lag_window(&r_grid, fluid_vp, dt_s, n_samples);
    let n_phi = angles_deg.len();
    let weights = angular_apodization_weights(angles_deg, angles_deg, angular_window_deg);
    SaftSliceContext {
        dt_s,
        template_offset,
        n_corr,
        lags,
        weights,
        n_phi,
    }
}

pub fn infer_axial_slice_saft_with_context(
    p_rx_slice: &[f64],
    n_theta: usize,
    n_samples: usize,
    p_tx: &[f64],
    ctx: &SaftSliceContext,
    fluid_vp: f64,
    coherent_sum: bool,
) -> Vec<f64> {
    let corrs = correlation_profiles(p_rx_slice, n_theta, n_samples, p_tx);
    let n_lags = ctx.lags.len();
    let power_vs_lag = combined_power_vs_lag(
        &corrs,
        n_theta,
        ctx.n_corr,
        ctx.template_offset,
        &ctx.lags,
        &ctx.weights,
        ctx.n_phi,
        coherent_sum,
    );

    let mut inferred = vec![0.0; ctx.n_phi];
    for iphi in 0..ctx.n_phi {
        let row = &power_vs_lag[iphi * n_lags..(iphi + 1) * n_lags];
        inferred[iphi] = radius_from_lag_peak(&ctx.lags, row, fluid_vp, ctx.dt_s);
    }
    inferred
}

#[allow(dead_code)]
pub fn infer_axial_slice_saft(
    p_rx_slice: &[f64],
    n_theta: usize,
    n_samples: usize,
    p_tx: &[f64],
    time_s: &[f64],
    angles_deg: &[f64],
    fluid_vp: f64,
    r_min_m: f64,
    r_max_m: f64,
    r_step_m: f64,
    angular_window_deg: f64,
    coherent_sum: bool,
) -> Vec<f64> {
    let ctx = build_saft_slice_context(
        p_tx,
        time_s,
        angles_deg,
        fluid_vp,
        r_min_m,
        r_max_m,
        r_step_m,
        angular_window_deg,
        n_samples,
    );
    infer_axial_slice_saft_with_context(
        p_rx_slice,
        n_theta,
        n_samples,
        p_tx,
        &ctx,
        fluid_vp,
        coherent_sum,
    )
}

pub fn matched_filter_range(
    p_rx: &[f64],
    p_tx: &[f64],
    time_s: &[f64],
    r_grid_m: &[f64],
    fluid_vp: f64,
) -> f64 {
    let dt_s = sample_dt_s(time_s);
    let corr = correlate_full(p_rx, p_tx);
    let template_offset = p_tx.len() - 1;
    let lags = lag_window(r_grid_m, fluid_vp, dt_s, p_rx.len());
    if lags.is_empty() {
        return r_grid_m[0];
    }
    let power: Vec<f64> = lags
        .iter()
        .map(|&lag| {
            let v = corr[template_offset + lag];
            v * v
        })
        .collect();
    radius_from_lag_peak(&lags, &power, fluid_vp, dt_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlate_matches_numpy_style_peak() {
        let mut signal = vec![0.0; 8];
        signal[3] = 1.0;
        let template = vec![0.0, 1.0, 0.0];
        let corr = correlate_full(&signal, &template);
        let peak = corr
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak, 4);
    }

    #[test]
    fn fft_correlate_matches_naive_full_mode() {
        let a = vec![0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let naive = correlate_full_naive(&a, &b);
        let fft = correlate_full_fft(&a, &b);
        assert_eq!(naive.len(), fft.len());
        for (n, f) in naive.iter().zip(fft.iter()) {
            assert!((n - f).abs() < 1e-10, "delta={}", n - f);
        }
    }

    #[test]
    fn fft_correlate_peak_index_matches_naive() {
        let mut signal = vec![0.0; 512];
        signal[200] = 1.0;
        let template: Vec<f64> = (0..32)
            .map(|i| (-0.5 * ((i as f64 - 16.0) / 4.0).powi(2)).exp())
            .collect();
        let naive = correlate_full_naive(&signal, &template);
        let fft = correlate_full_fft(&signal, &template);
        let peak_naive = naive
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0;
        let peak_fft = fft
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_naive, peak_fft);
    }
}
