//! Open boundary conditions and coastal/wind momentum forcing for `SurfaceSolverF32`.

use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Exterior edge boundary condition for one side of the 2D `F32` grid.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SurfaceF32SideBoundary {
    /// Closed impermeable reflecting wall (zero mass flux, hydrostatic wall pressure).
    #[default]
    Wall,
    /// Zero-gradient transmissive open outflow (`h_g = h_in`, `q_g = q_in`).
    Transmissive,
    /// Prescribed exterior water-surface elevation in meters (`h_g = max(0, stage - bed)`).
    Stage(f32),
    /// Prescribed normal unit discharge into the domain (`m²/s`, positive inward).
    Discharge(f32),
}

impl SurfaceF32SideBoundary {
    pub(crate) fn validate(self, side: &str) -> Result<(), EngineError> {
        match self {
            Self::Wall | Self::Transmissive => Ok(()),
            Self::Stage(eta) if eta.is_finite() => Ok(()),
            Self::Discharge(q) if q.is_finite() => Ok(()),
            other => Err(EngineError::rejected(
                "surface solver f32 boundary",
                format!("non-finite boundary parameter on {side}: {other:?}"),
            )),
        }
    }
}

/// Boundary conditions on all four exterior sides of the 2D `F32` grid.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SurfaceF32Boundaries {
    /// West edge (`col = 0`, outward normal `-x`).
    pub west: SurfaceF32SideBoundary,
    /// East edge (`col = nx - 1`, outward normal `+x`).
    pub east: SurfaceF32SideBoundary,
    /// South edge (`row = 0`, outward normal `-y`).
    pub south: SurfaceF32SideBoundary,
    /// North edge (`row = ny - 1`, outward normal `+y`).
    pub north: SurfaceF32SideBoundary,
}

impl SurfaceF32Boundaries {
    /// All four edges closed impermeable walls.
    pub const fn impermeable_walls() -> Self {
        Self {
            west: SurfaceF32SideBoundary::Wall,
            east: SurfaceF32SideBoundary::Wall,
            south: SurfaceF32SideBoundary::Wall,
            north: SurfaceF32SideBoundary::Wall,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        self.west.validate("west")?;
        self.east.validate("east")?;
        self.south.validate("south")?;
        self.north.validate("north")?;
        Ok(())
    }
}

/// Opt-in coastal & meteorological momentum forcing for `SurfaceSolverF32`.
///
/// Default has all forcing terms zero (no overhead).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceF32CoastalForcing {
    /// Uniform 10 m wind x-component (`m/s`).
    pub wind_x_m_per_s: f32,
    /// Uniform 10 m wind y-component (`m/s`).
    pub wind_y_m_per_s: f32,
    /// Dimensionless wind drag coefficient `C_d` (default `1.5e-3`).
    pub wind_drag_coeff: f32,
    /// Coriolis frequency `f = 2 * Omega * sin(lat)` (`1/s`, default `0.0`).
    pub coriolis_f_per_s: f32,
    /// Atmospheric pressure gradient `dP/dx` (`Pa/m`, default `0.0`).
    pub dp_dx_pa_per_m: f32,
    /// Atmospheric pressure gradient `dP/dy` (`Pa/m`, default `0.0`).
    pub dp_dy_pa_per_m: f32,
    /// Water density (`kg/m³`, default `1000.0`).
    pub water_density_kg_per_m3: f32,
    /// Air density (`kg/m³`, default `1.225`).
    pub air_density_kg_per_m3: f32,
}

impl Default for SurfaceF32CoastalForcing {
    fn default() -> Self {
        Self {
            wind_x_m_per_s: 0.0,
            wind_y_m_per_s: 0.0,
            wind_drag_coeff: 1.5e-3,
            coriolis_f_per_s: 0.0,
            dp_dx_pa_per_m: 0.0,
            dp_dy_pa_per_m: 0.0,
            water_density_kg_per_m3: 1000.0,
            air_density_kg_per_m3: 1.225,
        }
    }
}

impl SurfaceF32CoastalForcing {
    /// Returns true if any coastal forcing term is nonzero.
    pub fn is_active(&self) -> bool {
        self.wind_x_m_per_s != 0.0
            || self.wind_y_m_per_s != 0.0
            || self.coriolis_f_per_s != 0.0
            || self.dp_dx_pa_per_m != 0.0
            || self.dp_dy_pa_per_m != 0.0
    }

    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        let reject = |msg: String| EngineError::rejected("surface solver f32 coastal forcing", msg);
        for (name, v) in [
            ("wind_x_m_per_s", self.wind_x_m_per_s),
            ("wind_y_m_per_s", self.wind_y_m_per_s),
            ("coriolis_f_per_s", self.coriolis_f_per_s),
            ("dp_dx_pa_per_m", self.dp_dx_pa_per_m),
            ("dp_dy_pa_per_m", self.dp_dy_pa_per_m),
        ] {
            if !v.is_finite() {
                return Err(reject(format!("{name} must be finite, got {v}")));
            }
        }
        if !self.wind_drag_coeff.is_finite() || self.wind_drag_coeff < 0.0 {
            return Err(reject(format!(
                "wind_drag_coeff must be finite and >= 0, got {}",
                self.wind_drag_coeff
            )));
        }
        if !self.water_density_kg_per_m3.is_finite() || self.water_density_kg_per_m3 <= 0.0 {
            return Err(reject(format!(
                "water_density_kg_per_m3 must be positive, got {}",
                self.water_density_kg_per_m3
            )));
        }
        if !self.air_density_kg_per_m3.is_finite() || self.air_density_kg_per_m3 < 0.0 {
            return Err(reject(format!(
                "air_density_kg_per_m3 must be >= 0, got {}",
                self.air_density_kg_per_m3
            )));
        }
        Ok(())
    }
}

/// Builds ghost `(h_g, qn_g, qt_g, z_g)` for one exterior edge slice.
///
/// `inward_sign` is `+1.0` for west/south edges (where inward flow is `+x`/`+y`) and `-1.0` for
/// east/north edges (where inward flow is `-x`/`-y`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn build_ghost_edge<R: Runtime<DType = DType>>(
    client: &R::Client,
    side: SurfaceF32SideBoundary,
    h_in: &Tensor<R>,
    qn_in: &Tensor<R>,
    qt_in: &Tensor<R>,
    z_in: &Tensor<R>,
    inward_sign: f64,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>), EngineError>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let shape = h_in.shape();
    let device = h_in.device();
    let zero = Tensor::<R>::zeros(shape, DType::F32, device)?;
    match side {
        SurfaceF32SideBoundary::Wall => Ok((
            h_in.clone(),
            client.neg(qn_in)?,
            qt_in.clone(),
            z_in.clone(),
        )),
        SurfaceF32SideBoundary::Transmissive => {
            Ok((h_in.clone(), qn_in.clone(), qt_in.clone(), z_in.clone()))
        }
        SurfaceF32SideBoundary::Stage(eta) => {
            let eta_t = Tensor::<R>::full_scalar(shape, DType::F32, f64::from(eta), device)?;
            let h_g = client.maximum(&client.sub(&eta_t, z_in)?, &zero)?;
            Ok((h_g, qn_in.clone(), zero, z_in.clone()))
        }
        SurfaceF32SideBoundary::Discharge(q_inward) => {
            let q_signed = f64::from(q_inward) * inward_sign;
            let qn_g = Tensor::<R>::full_scalar(shape, DType::F32, q_signed, device)?;
            Ok((h_in.clone(), qn_g, zero, z_in.clone()))
        }
    }
}

/// Applies wind stress, atmospheric pressure gradient, and Crank-Nicolson Coriolis rotation in `F32`.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_coastal_forcing_f32<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    active: &Tensor<R>,
    coastal: SurfaceF32CoastalForcing,
    dt: f32,
    dry_depth: f32,
) -> Result<(Tensor<R>, Tensor<R>), EngineError>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    if !coastal.is_active() {
        return Ok((qx.clone(), qy.clone()));
    }
    let shape = h.shape();
    let device = h.device();
    let zero = Tensor::<R>::zeros(shape, DType::F32, device)?;
    let eps = Tensor::<R>::full_scalar(shape, DType::F32, f64::from(dry_depth), device)?;
    let wet = client.gt(h, &eps)?;

    let w_speed = f64::from(coastal.wind_x_m_per_s).hypot(f64::from(coastal.wind_y_m_per_s));
    let drag_ratio = f64::from(coastal.wind_drag_coeff)
        * (f64::from(coastal.air_density_kg_per_m3) / f64::from(coastal.water_density_kg_per_m3));
    let dt_f = f64::from(dt);
    let tau_x_dt = drag_ratio * w_speed * f64::from(coastal.wind_x_m_per_s) * dt_f;
    let tau_y_dt = drag_ratio * w_speed * f64::from(coastal.wind_y_m_per_s) * dt_f;
    let inv_rho = 1.0 / f64::from(coastal.water_density_kg_per_m3);
    let pg_x = -f64::from(coastal.dp_dx_pa_per_m) * inv_rho * dt_f;
    let pg_y = -f64::from(coastal.dp_dy_pa_per_m) * inv_rho * dt_f;

    let mut qx_new = client.add(
        &client.add_scalar(qx, tau_x_dt)?,
        &client.mul_scalar(h, pg_x)?,
    )?;
    let mut qy_new = client.add(
        &client.add_scalar(qy, tau_y_dt)?,
        &client.mul_scalar(h, pg_y)?,
    )?;

    if coastal.coriolis_f_per_s != 0.0 {
        let alpha = f64::from(coastal.coriolis_f_per_s) * dt_f;
        let inv_det = 1.0 / (1.0 + alpha * alpha);
        let rx = client.mul_scalar(
            &client.add(&qx_new, &client.mul_scalar(&qy_new, alpha)?)?,
            inv_det,
        )?;
        let ry = client.mul_scalar(
            &client.sub(&qy_new, &client.mul_scalar(&qx_new, alpha)?)?,
            inv_det,
        )?;
        qx_new = rx;
        qy_new = ry;
    }

    let qx_out = client.mul(&client.where_cond(&wet, &qx_new, &zero)?, active)?;
    let qy_out = client.mul(&client.where_cond(&wet, &qy_new, &zero)?, active)?;
    Ok((qx_out, qy_out))
}
