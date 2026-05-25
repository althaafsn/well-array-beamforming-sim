pub struct WallProfileFlat {
    pub z_m: Vec<f64>,
    pub theta_rad: Vec<f64>,
    pub inner_radius_m: Vec<f64>,
    pub amplitude_multiplier: Option<Vec<f64>>,
    #[allow(dead_code)]
    pub nominal_inner_radius_m: f64,
}

impl WallProfileFlat {
    pub fn is_present(&self) -> bool {
        !self.z_m.is_empty() && !self.theta_rad.is_empty() && !self.inner_radius_m.is_empty()
    }

    fn index(&self, z_m: f64, theta_rad: f64) -> (usize, usize) {
        let n_z = self.z_m.len();
        let n_theta = self.theta_rad.len();
        let mut z_idx = 0usize;
        let mut z_best = f64::INFINITY;
        for (i, &z) in self.z_m.iter().enumerate() {
            let d = (z - z_m).abs();
            if d < z_best {
                z_best = d;
                z_idx = i;
            }
        }
        let mut theta_idx = 0usize;
        let mut theta_best = f64::INFINITY;
        for (i, &theta) in self.theta_rad.iter().enumerate() {
            let dtheta = (theta - theta_rad).sin().atan2((theta - theta_rad).cos()).abs();
            if dtheta < theta_best {
                theta_best = dtheta;
                theta_idx = i;
            }
        }
        let _ = (n_z, n_theta);
        (z_idx, theta_idx)
    }

    pub fn inner_radius_at(&self, z_m: f64, theta_rad: f64, nominal: f64) -> f64 {
        if !self.is_present() {
            return nominal;
        }
        let (z_idx, theta_idx) = self.index(z_m, theta_rad);
        let n_theta = self.theta_rad.len();
        self.inner_radius_m[z_idx * n_theta + theta_idx]
    }

    pub fn amplitude_multiplier_at(&self, z_m: f64, theta_rad: f64) -> f64 {
        let Some(ref amp) = self.amplitude_multiplier else {
            return 1.0;
        };
        if !self.is_present() {
            return 1.0;
        }
        let (z_idx, theta_idx) = self.index(z_m, theta_rad);
        let n_theta = self.theta_rad.len();
        amp[z_idx * n_theta + theta_idx]
    }
}

pub fn echo_amplitude_scale(
    profile: &WallProfileFlat,
    amplitude_exponent: f64,
    z_m: f64,
    theta_rad: f64,
    nominal_inner_radius_m: f64,
) -> f64 {
    let local_r = profile.inner_radius_at(z_m, theta_rad, nominal_inner_radius_m);
    let mut scale = profile.amplitude_multiplier_at(z_m, theta_rad);
    if amplitude_exponent != 0.0 && local_r > 0.0 {
        scale *= (nominal_inner_radius_m / local_r).powf(amplitude_exponent);
    }
    scale
}
