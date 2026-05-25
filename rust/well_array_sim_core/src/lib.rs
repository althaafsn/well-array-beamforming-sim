mod pulse;
mod ray_forward;
mod saft_polar;
mod wall_lookup;

use std::sync::Arc;

use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use ray_forward::{build_tx_pulse, reflection_coeff, simulate_shot};
use saft_polar::{
    build_saft_slice_context, infer_axial_slice_saft_with_context, inference_range_grid,
    matched_filter_range, SaftSliceContext,
};
use wall_lookup::WallProfileFlat;

struct ZStationOutput {
    iz: usize,
    ground_truth: Vec<f64>,
    peaks: Vec<f64>,
    inferred: Vec<f64>,
    measured_us: Vec<f64>,
    p_rx_slice: Vec<f64>,
}

fn parallel_z_enabled() -> bool {
    std::env::var("WELL_ARRAY_SIM_RUST_PARALLEL")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(true)
}

#[allow(clippy::too_many_arguments)]
fn process_z_station(
    iz: usize,
    z_m: f64,
    n_theta: usize,
    n_tx: usize,
    time_s: &[f64],
    p_tx: &[f64],
    angles: &[f64],
    nominal_inner_radius_m: f64,
    fluid_vp: f64,
    refl: f64,
    profile: &WallProfileFlat,
    amplitude_exponent: f64,
    snr_db: Option<f64>,
    noise_seed: Option<u64>,
    per_shot_inference: bool,
    use_angular_saft: bool,
    store_waveforms: bool,
    coherent_sum: bool,
    r_grid: &[f64],
    saft_ctx: Option<&SaftSliceContext>,
) -> ZStationOutput {
    let mut ground_truth = vec![0.0; n_theta];
    let mut peaks = vec![0.0; n_theta];
    let mut inferred = vec![0.0; n_theta];
    let mut measured_us = vec![0.0; n_theta];
    let mut p_rx_slice = if store_waveforms {
        vec![0.0; n_theta * n_tx]
    } else {
        Vec::new()
    };

    for (it, &angle_deg) in angles.iter().enumerate() {
        let theta_rad = angle_deg.to_radians();
        let (p_rx, gt_m) = simulate_shot(
            time_s,
            p_tx,
            theta_rad,
            z_m,
            nominal_inner_radius_m,
            fluid_vp,
            refl,
            profile,
            amplitude_exponent,
            snr_db,
            noise_seed,
        );
        ground_truth[it] = gt_m;
        peaks[it] = p_rx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        if store_waveforms {
            let base = it * n_tx;
            p_rx_slice[base..base + n_tx].copy_from_slice(&p_rx);
        }

        if per_shot_inference {
            let inferred_m = matched_filter_range(&p_rx, p_tx, time_s, r_grid, fluid_vp);
            inferred[it] = inferred_m;
            measured_us[it] = 2.0 * inferred_m / fluid_vp * 1e6;
        }
    }

    if use_angular_saft {
        let ctx = saft_ctx.expect("saft context required for angular_saft");
        let slice_inferred = infer_axial_slice_saft_with_context(
            &p_rx_slice,
            n_theta,
            n_tx,
            p_tx,
            ctx,
            fluid_vp,
            coherent_sum,
        );
        for (it, &r) in slice_inferred.iter().enumerate() {
            inferred[it] = r;
            measured_us[it] = 2.0 * r / fluid_vp * 1e6;
        }
    }

    ZStationOutput {
        iz,
        ground_truth,
        peaks,
        inferred,
        measured_us,
        p_rx_slice,
    }
}

#[pyfunction]
#[pyo3(signature = (
    nominal_inner_radius_m,
    fluid_rho,
    fluid_vp,
    steel_rho,
    steel_vp,
    center_freq_hz,
    bandwidth,
    t_end_us,
    dt_us,
    tp0_s,
    z_stations,
    angles_deg,
    wall_z_m=None,
    wall_theta_rad=None,
    wall_inner_radius_m=None,
    wall_amplitude_multiplier=None,
    amplitude_exponent=0.0,
    snr_db=None,
    noise_seed=None,
    inference_mode="angular_saft",
    r_min_m=0.07,
    r_max_m=0.14,
    r_step_m=0.0005,
    angular_window_deg=15.0,
    coherent_sum=true,
    store_waveforms=true,
))]
#[allow(clippy::too_many_arguments)]
fn simulate_axial_scan_rust<'py>(
    py: Python<'py>,
    nominal_inner_radius_m: f64,
    fluid_rho: f64,
    fluid_vp: f64,
    steel_rho: f64,
    steel_vp: f64,
    center_freq_hz: f64,
    bandwidth: f64,
    t_end_us: f64,
    dt_us: f64,
    tp0_s: f64,
    z_stations: PyReadonlyArray1<'py, f64>,
    angles_deg: PyReadonlyArray1<'py, f64>,
    wall_z_m: Option<PyReadonlyArray1<'py, f64>>,
    wall_theta_rad: Option<PyReadonlyArray1<'py, f64>>,
    wall_inner_radius_m: Option<PyReadonlyArray2<'py, f64>>,
    wall_amplitude_multiplier: Option<PyReadonlyArray2<'py, f64>>,
    amplitude_exponent: f64,
    snr_db: Option<f64>,
    noise_seed: Option<u64>,
    inference_mode: &str,
    r_min_m: f64,
    r_max_m: f64,
    r_step_m: f64,
    angular_window_deg: f64,
    coherent_sum: bool,
    store_waveforms: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let z_list: Vec<f64> = z_stations.as_slice()?.to_vec();
    let angles: Vec<f64> = angles_deg.as_slice()?.to_vec();
    let n_z = z_list.len();
    let n_theta = angles.len();
    if n_z == 0 || n_theta == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "z_stations and angles_deg must be non-empty",
        ));
    }

    let profile = Arc::new(build_wall_profile(
        wall_z_m,
        wall_theta_rad,
        wall_inner_radius_m,
        wall_amplitude_multiplier,
        nominal_inner_radius_m,
    )?);

    let (time_s, p_tx) = build_tx_pulse(t_end_us, dt_us, center_freq_hz, bandwidth, tp0_s);
    let n_tx = time_s.len();
    let refl = reflection_coeff(fluid_rho, fluid_vp, steel_rho, steel_vp);

    let use_angular_saft = inference_mode == "angular_saft";
    let per_shot_inference = inference_mode == "matched_filter";
    if use_angular_saft && !store_waveforms {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "angular_saft inference requires store_waveforms=True",
        ));
    }

    let r_grid = inference_range_grid(r_min_m, r_max_m, r_step_m);
    let saft_ctx = if use_angular_saft {
        Some(build_saft_slice_context(
            &p_tx,
            &time_s,
            &angles,
            fluid_vp,
            r_min_m,
            r_max_m,
            r_step_m,
            angular_window_deg,
            n_tx,
        ))
    } else {
        None
    };
    let saft_ctx = saft_ctx.as_ref();

    let mut inferred = vec![0.0; n_z * n_theta];
    let mut measured_us = vec![0.0; n_z * n_theta];
    let mut peaks = vec![0.0; n_z * n_theta];
    let mut ground_truth = vec![0.0; n_z * n_theta];
    let mut p_rx_flat = if store_waveforms {
        vec![0.0; n_z * n_theta * n_tx]
    } else {
        Vec::new()
    };

    let process = |iz: usize, z_m: f64| {
        process_z_station(
            iz,
            z_m,
            n_theta,
            n_tx,
            &time_s,
            &p_tx,
            &angles,
            nominal_inner_radius_m,
            fluid_vp,
            refl,
            profile.as_ref(),
            amplitude_exponent,
            snr_db,
            noise_seed,
            per_shot_inference,
            use_angular_saft,
            store_waveforms,
            coherent_sum,
            &r_grid,
            saft_ctx,
        )
    };

    let outputs: Vec<ZStationOutput> = if parallel_z_enabled() && n_z > 1 {
        z_list
            .par_iter()
            .enumerate()
            .map(|(iz, &z_m)| process(iz, z_m))
            .collect()
    } else {
        z_list
            .iter()
            .enumerate()
            .map(|(iz, &z_m)| process(iz, z_m))
            .collect()
    };

    for out in outputs {
        let row_base = out.iz * n_theta;
        ground_truth[row_base..row_base + n_theta].copy_from_slice(&out.ground_truth);
        peaks[row_base..row_base + n_theta].copy_from_slice(&out.peaks);
        inferred[row_base..row_base + n_theta].copy_from_slice(&out.inferred);
        measured_us[row_base..row_base + n_theta].copy_from_slice(&out.measured_us);
        if store_waveforms {
            let slice_base = out.iz * n_theta * n_tx;
            p_rx_flat[slice_base..slice_base + n_theta * n_tx].copy_from_slice(&out.p_rx_slice);
        }
    }

    let time_us: Vec<f64> = time_s.iter().map(|&t| t * 1e6).collect();
    let angle_step = if n_theta > 1 {
        angles[1] - angles[0]
    } else {
        1.0
    };
    let z_step = if n_z > 1 { z_list[1] - z_list[0] } else { 0.01 };

    let inferred_arr = Array2::from_shape_vec((n_z, n_theta), inferred).expect("inferred shape");
    let measured_arr =
        Array2::from_shape_vec((n_z, n_theta), measured_us).expect("measured shape");
    let peaks_arr = Array2::from_shape_vec((n_z, n_theta), peaks).expect("peaks shape");
    let ground_truth_arr =
        Array2::from_shape_vec((n_z, n_theta), ground_truth).expect("ground truth shape");

    let dict = PyDict::new(py);
    dict.set_item("z_stations_m", PyArray1::from_vec(py, z_list))?;
    dict.set_item("angles_deg", PyArray1::from_vec(py, angles))?;
    dict.set_item(
        "inferred_distance_m",
        PyArray2::from_owned_array(py, inferred_arr),
    )?;
    dict.set_item(
        "measured_echo_us",
        PyArray2::from_owned_array(py, measured_arr),
    )?;
    dict.set_item("peak_amplitude", PyArray2::from_owned_array(py, peaks_arr))?;
    dict.set_item(
        "ground_truth_distance_m",
        PyArray2::from_owned_array(py, ground_truth_arr),
    )?;
    dict.set_item("time_us", PyArray1::from_vec(py, time_us))?;
    dict.set_item("p_tx", PyArray1::from_vec(py, p_tx))?;
    if store_waveforms {
        let p_rx_arr = Array3::from_shape_vec((n_z, n_theta, n_tx), p_rx_flat).expect("p_rx shape");
        dict.set_item("p_rx", PyArray3::from_owned_array(py, p_rx_arr))?;
    } else {
        dict.set_item("p_rx", py.None())?;
    }
    dict.set_item("wall_distance_m", nominal_inner_radius_m)?;
    dict.set_item("angle_step_deg", angle_step)?;
    dict.set_item("z_step_m", z_step)?;
    dict.set_item("engine", "rust")?;
    Ok(dict)
}

fn build_wall_profile(
    wall_z_m: Option<PyReadonlyArray1<'_, f64>>,
    wall_theta_rad: Option<PyReadonlyArray1<'_, f64>>,
    wall_inner_radius_m: Option<PyReadonlyArray2<'_, f64>>,
    wall_amplitude_multiplier: Option<PyReadonlyArray2<'_, f64>>,
    nominal_inner_radius_m: f64,
) -> PyResult<WallProfileFlat> {
    let (Some(z), Some(theta), Some(radii)) = (wall_z_m, wall_theta_rad, wall_inner_radius_m)
    else {
        return Ok(WallProfileFlat {
            z_m: Vec::new(),
            theta_rad: Vec::new(),
            inner_radius_m: Vec::new(),
            amplitude_multiplier: None,
            nominal_inner_radius_m,
        });
    };
    let z_m = z.as_slice()?.to_vec();
    let theta_rad = theta.as_slice()?.to_vec();
    let shape = radii.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "wall_inner_radius_m must be 2D",
        ));
    }
    let flat = radii.as_slice()?.to_vec();
    let amp = if let Some(a) = wall_amplitude_multiplier {
        Some(a.as_slice()?.to_vec())
    } else {
        None
    };
    Ok(WallProfileFlat {
        z_m,
        theta_rad,
        inner_radius_m: flat,
        amplitude_multiplier: amp,
        nominal_inner_radius_m,
    })
}

#[pyfunction]
fn extension_available() -> bool {
    true
}

#[pymodule]
fn well_array_sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_axial_scan_rust, m)?)?;
    m.add_function(wrap_pyfunction!(extension_available, m)?)?;
    Ok(())
}
