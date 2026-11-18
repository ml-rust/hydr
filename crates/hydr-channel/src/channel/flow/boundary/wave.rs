use super::algebra::{Bounds, Interior};
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
    fn boundary_shock_magnitude(
        &self,
        h: &Bounds<R>,
        hi: &Bounds<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        let c = &self.client;
        let high = Bounds {
            lo: c.maximum(&h.lo, &hi.lo)?,
            hi: c.maximum(&h.hi, &hi.hi)?,
        };
        let ratio = self.bdiv(hi, &high)?;
        let one = Bounds::exact(&self.constant(&h.lo, 1.0)?);
        if self.boundary_triangle() {
            // Cancel the common depth difference before enclosing the Rankine-Hugoniot quotient.
            let polynomial = self.badd(&self.badd(&one, &ratio)?, &self.bmul(&ratio, &ratio)?)?;
            let shape = self.bdiv(&polynomial, &self.badd(&one, &ratio)?)?;
            let scale = self.bdiv(
                &self.bmul(
                    &self.bsqrt(&Bounds::exact(
                        &self.constant(&h.lo, self.definition.config.gravity)?,
                    ))?,
                    &self.bsqrt(&high)?,
                )?,
                &self.bsqrt(&Bounds::exact(&self.constant(&h.lo, 3.0)?))?,
            )?;
            self.bmul(
                &self.bmul(&scale, &self.bdiv(&high, hi)?)?,
                &self.bsqrt(&shape)?,
            )
        } else {
            let factor = self.bsqrt(&self.bscale(&self.badd(&one, &ratio)?, 0.5)?)?;
            self.bmul(
                &self.bmul(
                    &self.bsqrt(&Bounds::exact(
                        &self.constant(&h.lo, self.definition.config.gravity)?,
                    ))?,
                    &self.bdiv(&high, &self.bsqrt(hi)?)?,
                )?,
                &factor,
            )
        }
    }
    pub(super) fn boundary_wave(
        &self,
        h: &Bounds<R>,
        interior: &Interior<R>,
        compatible: &Bounds<R>,
        delivered: &Bounds<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let sound = self.boundary_sound(h)?;
        let interior_sound = self.boundary_sound(&interior.depth_bounds)?;
        let lambda_i = self.bsub(&interior.velocity_bounds, &interior_sound)?;
        let lambda_b = self.bsub(compatible, &sound)?;
        let zero = self.constant(&h.lo, 0.0)?;
        self.boundary_require(
            &c.lt(&lambda_i.hi, &zero)?,
            "compatible wave head lacks a negative directed speed",
        )?;
        self.boundary_require(
            &c.lt(&lambda_b.hi, &zero)?,
            "compatible wave tail lacks a negative directed speed",
        )?;
        let mut speed = c.maximum(
            &self.babs(&self.badd(
                &Bounds::exact(&self.babs(&interior.velocity_bounds)?),
                &interior_sound,
            )?)?,
            &self.babs(&self.badd(&Bounds::exact(&self.babs(delivered)?), &sound)?)?,
        )?;
        if !self.all(&c.le(&h.hi, &interior.depth_bounds.lo)?)? {
            let mut magnitude = self.boundary_shock_magnitude(h, &interior.depth_bounds)?;
            magnitude.lo = c.maximum(&magnitude.lo, &interior_sound.lo)?;
            let shock = self.bsub(&interior.velocity_bounds, &magnitude)?;
            self.boundary_require(
                &c.lt(&shock.hi, &zero)?,
                "compatible compression wave lacks a negative directed speed",
            )?;
            // Pressure powers 2 and 1.5 are strictly convex in area.
            // D is the shock magnitude and f = ui - ub is the compatible velocity jump.
            // For positive compression, the secant slope gives D > ci and D - f < cb.
            // Thus ub - cb < ui - D < ui - ci proves both strict Lax inequalities.
            // Independent rounded speed enclosures need no strict separation for weak compression.
            speed = c.maximum(&speed, &self.babs(&shock)?)?;
        }
        {
            // Every retained compatibility defect requires the delivered pair's full invariant-region bound.
            let factor = if self.boundary_triangle() { 4.0 } else { 2.0 };
            let low = c.minimum(
                &self
                    .bsub(
                        &interior.velocity_bounds,
                        &self.bscale(&interior_sound, factor)?,
                    )?
                    .lo,
                &self.bsub(delivered, &self.bscale(&sound, factor)?)?.lo,
            )?;
            let high = c.maximum(
                &self
                    .badd(
                        &interior.velocity_bounds,
                        &self.bscale(&interior_sound, factor)?,
                    )?
                    .hi,
                &self.badd(delivered, &self.bscale(&sound, factor)?)?.hi,
            )?;
            speed = c.maximum(&speed, &c.maximum(&c.abs(&low)?, &c.abs(&high)?)?)?;
        }
        let speed = upper::<R>(c, &speed)?;
        self.finite(&speed, "scalar boundary signal speed")?;
        Ok(speed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::{
        ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
        RectangularChannelCell, TriangularChannelCell,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn context(triangle: bool) -> Result<Context<CpuRuntime>, ChannelError> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let geometry = if triangle {
            ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v: 1.0,
                right_bank_slope_h_per_v: 1.0,
                bed_elevation_m: 0.0,
                cells: vec![TriangularChannelCell {
                    id: "cell".into(),
                    length_m: 128.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                }],
            }
        } else {
            ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 128.0,
                    width_m: 1.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                }],
            }
        };
        Context::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "resolved weak compression wave fixture".into(),
                geometry,
                lateral_bindings: vec![],
                config: ChannelConfig {
                    gravity: 1.0,
                    ..ChannelConfig::default()
                },
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![128.0],
                longitudinal_momentum_m4_per_s: vec![16.0],
            },
        )
    }

    fn check_weak_compression(triangle: bool) -> TestResult {
        let ctx = context(triangle)?;
        let c = &ctx.client;
        let depth = ctx.tensor(&[1.0])?;
        let velocity = ctx.tensor(&[0.125])?;
        let interior = Interior {
            depth: depth.clone(),
            velocity: velocity.clone(),
            area: depth.clone(),
            discharge: velocity.clone(),
            depth_bounds: Bounds::exact(&depth),
            velocity_bounds: Bounds::exact(&velocity),
        };
        let trace = Bounds::exact(&ctx.tensor(&[1.0 + 4.0 * f64::EPSILON])?);
        assert!(ctx.all(&c.gt(&trace.lo, &interior.depth_bounds.hi)?)?);
        let compatible = ctx.boundary_velocity(&trace, &interior)?;
        let sound = ctx.boundary_sound(&trace)?;
        let interior_sound = ctx.boundary_sound(&interior.depth_bounds)?;
        let lambda_i = ctx.bsub(&interior.velocity_bounds, &interior_sound)?;
        let lambda_b = ctx.bsub(&compatible, &sound)?;
        let mut magnitude = ctx.boundary_shock_magnitude(&trace, &interior.depth_bounds)?;
        magnitude.lo = c.maximum(&magnitude.lo, &interior_sound.lo)?;
        let shock = ctx.bsub(&interior.velocity_bounds, &magnitude)?;
        for bounds in [&lambda_i, &lambda_b, &shock] {
            assert!(bounds.hi.item::<f64>()? < 0.0);
        }
        // Independent directed enclosures overlap despite strictly positive compression.
        for characteristic in [&lambda_i, &lambda_b] {
            assert!(shock.lo.item::<f64>()? <= characteristic.hi.item::<f64>()?);
            assert!(characteristic.lo.item::<f64>()? <= shock.hi.item::<f64>()?);
        }
        let speed = ctx.boundary_wave(&trace, &interior, &compatible, &compatible)?;
        let speed = speed.item::<f64>()?;
        assert!(speed.is_finite() && speed > 0.0);
        assert!(speed >= shock.lo.item::<f64>()?.abs());
        Ok(())
    }

    #[test]
    fn rectangular_weak_compression_admits_overlapping_directed_lax_enclosures() -> TestResult {
        check_weak_compression(false)
    }

    #[test]
    fn triangular_weak_compression_admits_overlapping_directed_lax_enclosures() -> TestResult {
        check_weak_compression(true)
    }
}
