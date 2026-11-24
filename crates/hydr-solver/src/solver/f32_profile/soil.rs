//! Vectorized `DType::F32` Modified Green-Ampt soil infiltration for `SurfaceSolverF32`.
//!
//! Executes natively on `CpuRuntime`, `CudaRuntime`, and `WgpuRuntime`.

use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Host-side Modified Green-Ampt soil parameters for `SurfaceSolverF32`.
///
/// Cells with `ks_m_per_s == 0.0` are treated as impervious.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceF32Soil {
    /// Saturated hydraulic conductivity `K_s` (`m/s`, `>= 0`).
    pub ks_m_per_s: Vec<f32>,
    /// Wetting-front capillary suction head `psi` (`m`, `>= 0`).
    pub psi_m: Vec<f32>,
    /// Initial moisture deficit `theta_d` (dimensionless, `>= 0`).
    pub theta_d: Vec<f32>,
    /// Initial cumulative infiltrated depth `F_0` (`m`, `>= 0`).
    pub initial_cumulative_m: Vec<f32>,
}

impl SurfaceF32Soil {
    /// Uniform Green-Ampt soil over `cell_count` cells.
    pub fn uniform(cell_count: usize, ks_m_per_s: f32, psi_m: f32, theta_d: f32) -> Self {
        Self {
            ks_m_per_s: vec![ks_m_per_s; cell_count],
            psi_m: vec![psi_m; cell_count],
            theta_d: vec![theta_d; cell_count],
            initial_cumulative_m: vec![0.0; cell_count],
        }
    }

    pub(crate) fn validate(&self, cell_count: usize) -> Result<(), EngineError> {
        let reject = |msg: String| EngineError::rejected("surface solver f32 soil", msg);
        for (name, slice) in [
            ("ks_m_per_s", &self.ks_m_per_s),
            ("psi_m", &self.psi_m),
            ("theta_d", &self.theta_d),
            ("initial_cumulative_m", &self.initial_cumulative_m),
        ] {
            if slice.len() != cell_count {
                return Err(reject(format!(
                    "{name} length ({}) != cell_count ({cell_count})",
                    slice.len()
                )));
            }
            if slice.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(reject(format!("{name} must be finite and >= 0")));
            }
        }
        Ok(())
    }
}

/// Device-resident `DType::F32` Modified Green-Ampt state.
pub(super) struct SoilStateF32<R: Runtime<DType = DType>> {
    ks: Tensor<R>,
    psi_theta: Tensor<R>,
    cum_f: Tensor<R>,
}

impl<R: Runtime<DType = DType>> SoilStateF32<R>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    pub(super) fn new(
        client: &R::Client,
        soil: &SurfaceF32Soil,
        ny: usize,
        nx: usize,
    ) -> Result<Self, EngineError> {
        let count = ny * nx;
        soil.validate(count)?;
        let shape = [ny, nx];
        let device = client.device();
        let psi_theta_vec: Vec<f32> = soil
            .psi_m
            .iter()
            .zip(&soil.theta_d)
            .map(|(&p, &t)| p * t)
            .collect();
        Ok(Self {
            ks: Tensor::<R>::from_slice(&soil.ks_m_per_s, &shape, device)?,
            psi_theta: Tensor::<R>::from_slice(&psi_theta_vec, &shape, device)?,
            cum_f: Tensor::<R>::from_slice(&soil.initial_cumulative_m, &shape, device)?,
        })
    }

    pub(super) fn cumulative_to_vec(&self) -> Result<Vec<f32>, EngineError> {
        Ok(self.cum_f.try_to_vec::<f32>()?)
    }

    pub(super) fn set_cumulative(&mut self, cum_f: Tensor<R>) {
        self.cum_f = cum_f;
    }

    pub(super) fn restore_cumulative(
        &mut self,
        client: &R::Client,
        values: &[f32],
        ny: usize,
        nx: usize,
    ) -> Result<(), EngineError> {
        if values.len() != ny * nx || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(EngineError::rejected(
                "surface solver f32 soil snapshot",
                "cumulative_infiltration_m length or finiteness invalid",
            ));
        }
        self.cum_f = Tensor::<R>::from_slice(values, &[ny, nx], client.device())?;
        Ok(())
    }

    /// Computes one Green-Ampt infiltration step in `DType::F32` without mutating `self`.
    ///
    /// Returns `(h_after, infiltrated_depth_m, next_cum_f)`.
    #[allow(clippy::type_complexity)]
    pub(super) fn step(
        &self,
        client: &R::Client,
        h: &Tensor<R>,
        active: &Tensor<R>,
        dt_f32: f32,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>), EngineError> {
        let shape = h.shape();
        let device = h.device();
        let zero = Tensor::<R>::zeros(shape, DType::F32, device)?;
        let tiny = Tensor::<R>::full_scalar(shape, DType::F32, 1e-12, device)?;
        let ks_dt = client.mul_scalar(&self.ks, f64::from(dt_f32))?;
        let pervious = client.gt(&ks_dt, &zero)?;

        // Initial guess: F1 = F0 + Ks * dt + sqrt(2 * Ks * dt * psi * theta_d).
        let f0 = &self.cum_f;
        let pt = &self.psi_theta;
        let boost = client.sqrt(&client.mul_scalar(&client.mul(&ks_dt, pt)?, 2.0)?)?;
        let mut f1 = client.add(&client.add(f0, &ks_dt)?, &boost)?;
        let f0_pt = client.maximum(&client.add(f0, pt)?, &tiny)?;

        // 6 safeguarded Newton iterations for F1 - F0 - pt * ln((F1 + pt)/(F0 + pt)) = Ks * dt.
        for _ in 0..6 {
            let f1_pt = client.maximum(&client.add(&f1, pt)?, &tiny)?;
            let ratio = client.div(&f1_pt, &f0_pt)?;
            let g = client.sub(
                &client.sub(
                    &client.sub(&f1, f0)?,
                    &client.mul(pt, &client.log(&ratio)?)?,
                )?,
                &ks_dt,
            )?;
            // g'(F1) = F1 / (F1 + pt) >= eps
            let gp = client.maximum(&client.div(&f1, &f1_pt)?, &tiny)?;
            let step = client.div(&g, &gp)?;
            f1 = client.maximum(&client.sub(&f1, &step)?, f0)?;
        }

        let potential = client.where_cond(&pervious, &client.sub(&f1, f0)?, &zero)?;
        let capped = client.mul(&client.minimum(h, &potential)?, active)?;
        let h_after = client.sub(h, &capped)?;
        let next_cum = client.add(f0, &capped)?;
        Ok((h_after, capped, next_cum))
    }
}
