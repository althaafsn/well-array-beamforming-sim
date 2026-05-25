use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64;

use crate::pulse::gaussian_tone_burst;
use crate::wall_lookup::{echo_amplitude_scale, WallProfileFlat};

pub fn reflection_coeff(fluid_rho: f64, fluid_vp: f64, steel_rho: f64, steel_vp: f64) -> f64 {
    let z_fluid = fluid_rho * fluid_vp;
    let z_steel = steel_rho * steel_vp;
    (z_steel - z_fluid) / (z_steel + z_fluid)
}

fn arrival_index(time_s: &[f64], arrival_time_s: f64) -> isize {
    let dt_s = if time_s.len() > 1 {
        time_s[1] - time_s[0]
    } else {
        1e-6
    };
    let t0_s = time_s.first().copied().unwrap_or(0.0);
    ((arrival_time_s - t0_s) / dt_s).round() as isize
}

pub fn synthesize_received_trace(
    time_s: &[f64],
    p_tx: &[f64],
    wall_distance_m: f64,
    fluid_vp: f64,
    reflection_coeff: f64,
    amplitude_scale: f64,
) -> Vec<f64> {
    let mut p_rx = vec![0.0; time_s.len()];
    let one_way_s = wall_distance_m / fluid_vp;
    let round_trip_s = 2.0 * one_way_s;
    let scale = reflection_coeff * amplitude_scale;

    for (i, (&t, &amp)) in time_s.iter().zip(p_tx.iter()).enumerate() {
        let idx = arrival_index(time_s, t + round_trip_s);
        if 0 <= idx && (idx as usize) < p_rx.len() {
            p_rx[idx as usize] += scale * amp;
            let _ = i;
        }
    }
    p_rx
}

pub fn add_trace_noise(
    trace: &[f64],
    snr_db: Option<f64>,
    rng: Option<&mut Pcg64>,
    reference_peak: f64,
) -> Vec<f64> {
    let (Some(snr_db), Some(rng)) = (snr_db, rng) else {
        return trace.to_vec();
    };
    if reference_peak <= 0.0 {
        return trace.to_vec();
    }
    let noise_rms = reference_peak / 10_f64.powf(snr_db / 20.0);
    let normal = Normal::new(0.0, noise_rms).expect("valid normal");
    trace
        .iter()
        .map(|&v| v + normal.sample(rng))
        .collect()
}

pub fn simulate_shot(
    time_s: &[f64],
    p_tx: &[f64],
    theta_rad: f64,
    z_m: f64,
    nominal_inner_radius_m: f64,
    fluid_vp: f64,
    reflection_coeff: f64,
    profile: &WallProfileFlat,
    amplitude_exponent: f64,
    snr_db: Option<f64>,
    noise_seed: Option<u64>,
) -> (Vec<f64>, f64) {
    let ground_truth_m = profile.inner_radius_at(z_m, theta_rad, nominal_inner_radius_m);
    let amplitude_scale = echo_amplitude_scale(
        profile,
        amplitude_exponent,
        z_m,
        theta_rad,
        nominal_inner_radius_m,
    );

    let p_rx_nominal = synthesize_received_trace(
        time_s,
        p_tx,
        nominal_inner_radius_m,
        fluid_vp,
        reflection_coeff,
        1.0,
    );
    let reference_peak = p_rx_nominal
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    let mut p_rx = synthesize_received_trace(
        time_s,
        p_tx,
        ground_truth_m,
        fluid_vp,
        reflection_coeff,
        amplitude_scale,
    );
    let mut noise_rng = noise_seed.map(Pcg64::seed_from_u64);
    p_rx = add_trace_noise(&p_rx, snr_db, noise_rng.as_mut(), reference_peak);
    (p_rx, ground_truth_m)
}

pub fn build_tx_pulse(
    t_end_us: f64,
    dt_us: f64,
    center_freq_hz: f64,
    bandwidth: f64,
    tp0_s: f64,
) -> (Vec<f64>, Vec<f64>) {
    let time_s = crate::pulse::make_time_axis(t_end_us, dt_us);
    let p_tx = gaussian_tone_burst(&time_s, center_freq_hz, bandwidth, tp0_s);
    (time_s, p_tx)
}
