use super::{
    NativeBoundaryTraces,
    algebra::{Bounds, Interior},
    discharge::Depth,
    kernel::Lane,
};
use crate::channel::flow::{ChannelError, context::Context, triangular::envelope::upper};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    pub(super) fn boundary_evaluate(
        &self,
        endpoint: usize,
        interior: &Interior<R>,
        coefficient: &Tensor<R>,
        depth: Depth<R>,
        target: Option<(Tensor<R>, Tensor<R>)>,
    ) -> Result<Lane<R>, ChannelError> {
        let c = &self.client;
        let h = depth.value;
        let hb = depth.bounds;
        let hi = &interior.depth;
        let ui = &interior.velocity;
        let zero = self.constant(&h, 0.0)?;
        self.finite(&h, "scalar boundary trace depth")?;
        let area = if self.boundary_triangle() {
            c.mul(&c.mul(coefficient, &h)?, &h)?
        } else {
            c.mul(coefficient, &h)?
        };
        self.boundary_require(&c.gt(&area, &zero)?, "trace area is unresolved")?;
        let ab = self.boundary_area(&hb, coefficient)?;
        let curve = self.boundary_velocity(&hb, interior)?;
        let curve_nominal = c.sub(ui, &self.boundary_curve_nominal(&h, hi)?)?;
        let prescribed = target.is_some();
        let (q, longitudinal) = if let Some((outward, longitudinal)) = target {
            (outward, longitudinal)
        } else {
            let rate = c.mul(&area, &curve_nominal)?;
            let longitudinal = c.mul_scalar(&rate, if endpoint == 0 { -1.0 } else { 1.0 })?;
            (rate, longitudinal)
        };
        let u = c.div(&q, &area)?;
        let ub = self.bdiv(&Bounds::exact(&q), &ab)?;
        let sound = self.boundary_subcritical(&hb, &ub)?;
        self.boundary_subcritical(&hb, &curve)?;
        let defect = self.babs(&self.bsub(&ub, &curve)?)?;
        let defect_b = Bounds::exact(&defect);
        let water_defect = self.bmul(&ab, &defect_b)?.hi;
        let momentum_defect = self
            .bmul(
                &self.bmul(&ab, &defect_b)?,
                &self.badd(
                    &self.bscale(&Bounds::exact(&self.babs(&ub)?), 2.0)?,
                    &defect_b,
                )?,
            )?
            .hi;
        let config = &self.definition.config.boundary;
        let interior_sound = self.boundary_sound(&interior.depth_bounds)?;
        let scale = c.maximum(
            &c.maximum(&self.babs(&interior.velocity_bounds)?, &self.babs(&ub)?)?,
            &c.maximum(&interior_sound.hi, &sound.hi)?,
        )?;
        self.boundary_require(
            &c.le(&defect, &c.mul_scalar(&scale, config.velocity_relative)?)?,
            "velocity compatibility exceeds its dedicated control",
        )?;
        let mut curve_q = self.bmul(&ab, &curve)?;
        let q_scale = c.maximum(&c.abs(&q)?, &self.babs(&curve_q)?)?;
        let q_tolerance = c.add_scalar(
            &c.mul_scalar(&q_scale, config.discharge_relative)?,
            config.discharge_absolute_m3_per_s,
        )?;
        self.boundary_require(
            &c.le(&water_defect, &q_tolerance)?,
            "water compatibility exceeds its dedicated controls",
        )?;
        let pressure_b = self.boundary_pressure(&hb, coefficient)?;
        let mut physical_momentum = self.badd(&self.bmul(&Bounds::exact(&q), &ub)?, &pressure_b)?;
        let compatible_momentum = self.badd(&self.bmul(&curve_q, &curve)?, &pressure_b)?;
        let m_scale = c.maximum(
            &self.babs(&physical_momentum)?,
            &self.babs(&compatible_momentum)?,
        )?;
        let m_tolerance = c.add_scalar(
            &c.mul_scalar(&m_scale, config.momentum_relative)?,
            config.momentum_absolute_m4_per_s2,
        )?;
        self.boundary_require(
            &c.le(&momentum_defect, &m_tolerance)?,
            "momentum compatibility exceeds its dedicated controls",
        )?;
        let signal = self.boundary_wave(&hb, interior, &curve, &ub)?;
        let (pressure, pressure_bound) = {
            let gravity = self.constant(&h, self.definition.config.gravity)?;
            let divisor = if self.boundary_triangle() { 3.0 } else { 2.0 };
            let factor = c.div(&gravity, &self.constant(&h, divisor)?)?;
            let factor_bound = c.where_cond(
                &c.eq(&factor, &zero)?,
                &self.constant(&h, f64::from_bits(1))?,
                &zero,
            )?;
            let ga = c.mul(&factor, &area)?;
            let ga_bound = self.product_bound(
                &factor,
                &factor_bound,
                &area,
                &ga,
                "scalar pressure coefficient-area product",
            )?;
            let pressure = c.mul(&ga, &h)?;
            let bound = self.product_bound(
                &ga,
                &ga_bound,
                &h,
                &pressure,
                "scalar full-section pressure",
            )?;
            (pressure, bound)
        };
        let advective = c.mul(&q, &u)?;
        let advective_bound = self.product_bound(
            &q,
            &zero,
            &u,
            &advective,
            "scalar boundary advective momentum",
        )?;
        let momentum = c.add(&advective, &pressure)?;
        let momentum_bound = self.bound_add(&advective_bound, &pressure_bound)?;
        let water_bound = if prescribed {
            zero.clone()
        } else {
            self.product_bound(&area, &zero, &curve_nominal, &q, "scalar stage discharge")?
        };
        curve_q.lo = c.minimum(&curve_q.lo, &q)?;
        curve_q.hi = c.maximum(&curve_q.hi, &q)?;
        physical_momentum.lo = c.minimum(&physical_momentum.lo, &momentum)?;
        physical_momentum.hi = c.maximum(&physical_momentum.hi, &momentum)?;
        let water_width = if prescribed {
            zero.clone()
        } else {
            upper::<R>(c, &c.sub(&curve_q.hi, &curve_q.lo)?)?
        };
        let momentum_width = upper::<R>(c, &c.sub(&physical_momentum.hi, &physical_momentum.lo)?)?;
        let flux = &self.definition.config;
        self.boundary_require(
            &c.le(
                &water_width,
                &c.add_scalar(
                    &c.mul_scalar(&c.abs(&q)?, flux.riemann_flux_relative)?,
                    flux.riemann_water_flux_absolute_m3_per_s,
                )?,
            )?,
            "physical water flux width exceeds its controls",
        )?;
        self.boundary_require(
            &c.le(
                &momentum_width,
                &c.add_scalar(
                    &c.mul_scalar(&c.abs(&momentum)?, flux.riemann_flux_relative)?,
                    flux.riemann_momentum_flux_absolute_m4_per_s2,
                )?,
            )?,
            "physical momentum flux width exceeds its controls",
        )?;
        for value in [
            &area,
            &q,
            &u,
            &momentum,
            &defect,
            &water_defect,
            &momentum_defect,
            &water_width,
            &momentum_width,
        ] {
            self.finite(value, "scalar boundary physical trace")?;
        }
        Ok(Lane {
            longitudinal,
            water_bound,
            momentum_bound,
            trace: NativeBoundaryTraces {
                depth_m: h,
                area_m2: area,
                outward_velocity_m_per_s: u,
                outward_discharge_m3_per_s: q,
                momentum_flux_m4_per_s2: momentum,
                signal_speed_bound_m_per_s: signal,
                velocity_compatibility_abs_bound_m_per_s: defect,
                water_compatibility_abs_bound_m3_per_s: water_defect,
                momentum_compatibility_abs_bound_m4_per_s2: momentum_defect,
                water_flux_width_m3_per_s: water_width,
                momentum_flux_width_m4_per_s2: momentum_width,
                cap_compatibility_approximation: self
                    .constant(&zero, if depth.approximate { 1.0 } else { 0.0 })?,
                endpoints: vec![endpoint],
                root_iterations: vec![depth.iterations],
            },
        })
    }
}
