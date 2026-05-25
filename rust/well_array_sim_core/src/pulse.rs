pub fn gaussian_envelope(time_s: f64, f0_hz: f64, bandwidth: f64, tp0_s: f64) -> f64 {
    let alpha = (std::f64::consts::PI * bandwidth * f0_hz).powi(2) / (4.0 * 0.6931471805599453);
    let exponent = (-alpha * (time_s - tp0_s).powi(2)).clamp(-700.0, 0.0);
    exponent.exp()
}

pub fn gaussian_tone_burst(
    time_axis: &[f64],
    f0_hz: f64,
    bandwidth: f64,
    tp0_s: f64,
) -> Vec<f64> {
    let mut pulse: Vec<f64> = time_axis
        .iter()
        .map(|&t| {
            let envelope = gaussian_envelope(t, f0_hz, bandwidth, tp0_s);
            let carrier = (2.0 * std::f64::consts::PI * f0_hz * (t - tp0_s)).sin();
            envelope * carrier
        })
        .collect();
    let peak = pulse.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if peak > 0.0 {
        for sample in &mut pulse {
            *sample /= peak;
        }
    }
    pulse
}

pub fn make_time_axis(t_end_us: f64, dt_us: f64) -> Vec<f64> {
    let count = (t_end_us / dt_us).round() as usize + 1;
    (0..count).map(|i| i as f64 * dt_us * 1e-6).collect()
}
