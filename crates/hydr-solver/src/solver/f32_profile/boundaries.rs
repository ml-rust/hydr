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

/// Optional per-face open masks for the four exterior sides of the 2D `F32` grid.
///
/// When a side's mask is `Some(mask)`, faces with `true` use the configured
/// [`SurfaceF32SideBoundary`] while faces with `false` act as impermeable reflecting walls.
/// `west` and `east` masks have length `ny`; `south` and `north` masks have length `nx`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SurfaceF32OpenMasks {
    /// West edge mask (`col = 0`, length `ny`). `true` = open, `false` = wall.
    pub west: Option<Vec<bool>>,
    /// East edge mask (`col = nx - 1`, length `ny`). `true` = open, `false` = wall.
    pub east: Option<Vec<bool>>,
    /// South edge mask (`row = 0`, length `nx`). `true` = open, `false` = wall.
    pub south: Option<Vec<bool>>,
    /// North edge mask (`row = ny - 1`, length `nx`). `true` = open, `false` = wall.
    pub north: Option<Vec<bool>>,
}

impl SurfaceF32OpenMasks {
    pub(crate) fn validate(&self, nx: usize, ny: usize) -> Result<(), EngineError> {
        for (name, mask, expected) in [
            ("west", &self.west, ny),
            ("east", &self.east, ny),
            ("south", &self.south, nx),
            ("north", &self.north, nx),
        ] {
            if let Some(m) = mask
                && m.len() != expected
            {
                return Err(EngineError::rejected(
                    "surface solver f32 open masks",
                    format!("{name} mask length ({}) != expected ({expected})", m.len()),
                ));
            }
        }
        Ok(())
    }
}

/// Device-resident per-face open masks (`1.0` = open, `0.0` = wall).
pub(super) struct OpenMaskTensors<R: Runtime<DType = DType>> {
    pub(super) west: Option<Tensor<R>>,
    pub(super) east: Option<Tensor<R>>,
    pub(super) south: Option<Tensor<R>>,
    pub(super) north: Option<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> Default for OpenMaskTensors<R> {
    fn default() -> Self {
        Self {
            west: None,
            east: None,
            south: None,
            north: None,
        }
    }
}

impl<R: Runtime<DType = DType>> OpenMaskTensors<R> {
    pub(super) fn from_host(
        masks: &SurfaceF32OpenMasks,
        nx: usize,
        ny: usize,
        device: &R::Device,
    ) -> Result<Self, EngineError> {
        masks.validate(nx, ny)?;
        let upload =
            |opt: &Option<Vec<bool>>, shape: &[usize]| -> Result<Option<Tensor<R>>, EngineError> {
                match opt {
                    Some(m) => {
                        let vals: Vec<f32> = m
                            .iter()
                            .map(|&b| if b { 1.0_f32 } else { 0.0_f32 })
                            .collect();
                        Ok(Some(Tensor::<R>::from_slice(&vals, shape, device)?))
                    }
                    None => Ok(None),
                }
            };
        Ok(Self {
            west: upload(&masks.west, &[ny, 1])?,
            east: upload(&masks.east, &[ny, 1])?,
            south: upload(&masks.south, &[1, nx])?,
            north: upload(&masks.north, &[1, nx])?,
        })
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
    open_mask: Option<&Tensor<R>>,
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
    let (h_open, qn_open, qt_open, z_open) = match side {
        SurfaceF32SideBoundary::Wall => {
            return Ok((
                h_in.clone(),
                client.neg(qn_in)?,
                qt_in.clone(),
                z_in.clone(),
            ));
        }
        SurfaceF32SideBoundary::Transmissive => {
            (h_in.clone(), qn_in.clone(), qt_in.clone(), z_in.clone())
        }
        SurfaceF32SideBoundary::Stage(eta) => {
            let eta_t = Tensor::<R>::full_scalar(shape, DType::F32, f64::from(eta), device)?;
            let h_g = client.maximum(&client.sub(&eta_t, z_in)?, &zero)?;
            (h_g, qn_in.clone(), zero, z_in.clone())
        }
        SurfaceF32SideBoundary::Discharge(q_inward) => {
            let q_signed = f64::from(q_inward) * inward_sign;
            let qn_g = Tensor::<R>::full_scalar(shape, DType::F32, q_signed, device)?;
            (h_in.clone(), qn_g, zero, z_in.clone())
        }
    };
    if let Some(mask) = open_mask {
        let half = Tensor::<R>::full_scalar(shape, DType::F32, 0.5, device)?;
        let is_open = client.gt(mask, &half)?;
        let qn_wall = client.neg(qn_in)?;
        let h_g = client.where_cond(&is_open, &h_open, h_in)?;
        let qn_g = client.where_cond(&is_open, &qn_open, &qn_wall)?;
        let qt_g = client.where_cond(&is_open, &qt_open, qt_in)?;
        let z_g = client.where_cond(&is_open, &z_open, z_in)?;
        return Ok((h_g, qn_g, qt_g, z_g));
    }
    Ok((h_open, qn_open, qt_open, z_open))
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

/// Applies semi-implicit Manning bed friction in `DType::F32`.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_manning_f32<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    manning_sq: &Tensor<R>,
    dt: f32,
    gravity: f32,
    dry_depth: f32,
) -> Result<(Tensor<R>, Tensor<R>), EngineError>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let shape = h.shape();
    let device = h.device();
    let zero = Tensor::<R>::zeros(shape, DType::F32, device)?;
    let eps = Tensor::<R>::full_scalar(shape, DType::F32, f64::from(dry_depth), device)?;
    let wet = client.gt(h, &eps)?;
    let safe_h = client.where_cond(&wet, h, &eps)?;

    let ux = client.where_cond(&wet, &client.div(qx, &safe_h)?, &zero)?;
    let uy = client.where_cond(&wet, &client.div(qy, &safe_h)?, &zero)?;
    let speed = client.sqrt(&client.add(&client.mul(&ux, &ux)?, &client.mul(&uy, &uy)?)?)?;

    let h_4_3 = client.pow_scalar(&safe_h, 4.0 / 3.0)?;
    let coeff = f64::from(dt) * f64::from(gravity);
    let drag = client.div(
        &client.mul_scalar(&client.mul(manning_sq, &speed)?, coeff)?,
        &h_4_3,
    )?;
    let denom = client.add_scalar(&drag, 1.0)?;

    let new_qx = client.where_cond(&wet, &client.div(qx, &denom)?, &zero)?;
    let new_qy = client.where_cond(&wet, &client.div(qy, &denom)?, &zero)?;
    Ok((new_qx, new_qy))
}
