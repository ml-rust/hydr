use super::algebra::{Bounds, Interior};
use crate::channel::flow::{
    ChannelError,
    context::Context,
    triangular::envelope::{lower, upper},
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

pub(super) struct ReservoirHeads<R: Runtime<DType = DType>> {
    pub velocity_head: Bounds<R>,
    pub entrance_loss_head: Bounds<R>,
    pub dissipation_head: Bounds<R>,
}

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
    pub(super) fn reservoir_add_head(
        &self,
        a: &Bounds<R>,
        b: &Bounds<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        let c = &self.client;
        let sum = self.badd(a, b)?;
        let zero = self.constant(&a.lo, 0.0)?;
        let exact_zero = c.mul(&c.eq(&b.lo, &zero)?, &c.eq(&b.hi, &zero)?)?;
        Ok(Bounds {
            lo: c.where_cond(&exact_zero, &a.lo, &sum.lo)?,
            hi: c.where_cond(&exact_zero, &a.hi, &sum.hi)?,
        })
    }
    fn reservoir_nonnegative_head(
        &self,
        speed: &Bounds<R>,
        k: Option<&Tensor<R>>,
    ) -> Result<Bounds<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(&speed.lo, 0.0)?;
        let active = if let Some(k) = k {
            c.mul(&c.gt(&speed.hi, &zero)?, &c.gt(k, &zero)?)?
        } else {
            c.gt(&speed.hi, &zero)?
        };
        let safe_speed = Bounds {
            lo: c.where_cond(&active, &speed.lo, &zero)?,
            hi: c.where_cond(&active, &speed.hi, &zero)?,
        };
        let root_g = self.bsqrt(&Bounds::exact(
            &self.constant(&zero, self.definition.config.gravity)?,
        ))?;
        let mut r = self.bdiv(&safe_speed, &root_g)?;
        if let Some(k) = k {
            let safe_k = c.where_cond(&active, k, &zero)?;
            r = self.bmul(&r, &self.bsqrt(&Bounds::exact(&safe_k))?)?;
        }
        // Nonnegative operands preserve monotonicity through the scaled square.
        let half_lo = c.mul_scalar(&c.maximum(&r.lo, &zero)?, 0.5)?;
        let half_hi = c.mul_scalar(&r.hi, 0.5)?;
        let lo = c.maximum(
            &lower::<R>(
                c,
                &c.mul(
                    &c.maximum(&lower::<R>(c, &half_lo)?, &zero)?,
                    &c.maximum(&r.lo, &zero)?,
                )?,
            )?,
            &zero,
        )?;
        let hi = upper::<R>(c, &c.mul(&upper::<R>(c, &half_hi)?, &r.hi)?)?;
        Ok(Bounds {
            lo: c.where_cond(&active, &lo, &zero)?,
            hi: c.where_cond(&active, &hi, &zero)?,
        })
    }
    pub(super) fn reservoir_heads(
        &self,
        u: &Bounds<R>,
        k: &Tensor<R>,
    ) -> Result<ReservoirHeads<R>, ChannelError> {
        let c = &self.client;
        let z = self.constant(&u.lo, 0.0)?;
        let incoming = Bounds {
            lo: c.maximum(&c.neg(&u.hi)?, &z)?,
            hi: c.maximum(&c.neg(&u.lo)?, &z)?,
        };
        let outgoing = Bounds {
            lo: c.maximum(&u.lo, &z)?,
            hi: c.maximum(&u.hi, &z)?,
        };
        let velocity = self.reservoir_nonnegative_head(&incoming, None)?;
        let loss = self.reservoir_nonnegative_head(&incoming, Some(k))?;
        let exit = self.reservoir_nonnegative_head(&outgoing, None)?;
        let dissipation = self.reservoir_add_head(&exit, &loss)?;
        Ok(ReservoirHeads {
            velocity_head: velocity,
            entrance_loss_head: loss,
            dissipation_head: Bounds {
                lo: c.maximum(&dissipation.lo, &z)?,
                hi: dissipation.hi,
            },
        })
    }
    pub(super) fn reservoir_residual(
        &self,
        h: &Tensor<R>,
        interior: &Interior<R>,
        head: &Bounds<R>,
        k: &Tensor<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        let hb = Bounds::exact(h);
        let ReservoirHeads {
            velocity_head: velocity,
            entrance_loss_head: loss,
            ..
        } = self.reservoir_heads(&self.boundary_velocity(&hb, interior)?, k)?;
        let mut residual = self.bsub(
            &self.reservoir_add_head(&self.reservoir_add_head(&hb, &velocity)?, &loss)?,
            head,
        )?;
        // Nonnegative velocity and loss heads also give G(h) >= h - head.upper.
        let depth_residual = self.bsub(&hb, head)?;
        residual.lo = self.client.maximum(&residual.lo, &depth_residual.lo)?;
        Ok(residual)
    }
    pub(super) fn reservoir_nominal(
        &self,
        h: &Tensor<R>,
        interior: &Interior<R>,
        head: &Tensor<R>,
        k: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let z = self.constant(h, 0.0)?;
        let u = c.sub(
            &interior.velocity,
            &self.boundary_curve_nominal(h, &interior.depth)?,
        )?;
        let v = c.maximum(&c.neg(&u)?, &z)?;
        let r = c.div(
            &v,
            &c.sqrt(&self.constant(h, self.definition.config.gravity)?)?,
        )?;
        let velocity = c.mul(&c.mul_scalar(&r, 0.5)?, &r)?;
        let active = c.mul(&c.gt(&v, &z)?, &c.gt(k, &z)?)?;
        let rk = c.mul(
            &c.where_cond(&active, &r, &z)?,
            &c.sqrt(&c.where_cond(&active, k, &z)?)?,
        )?;
        let loss = c.mul(&c.mul_scalar(&rk, 0.5)?, &rk)?;
        Ok(c.sub(&c.add(&c.add(h, &velocity)?, &loss)?, head)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::{
        ChannelConfig, ChannelInitialState, ChannelReachDefinition, RectangularChannelCell,
        geometry::ChannelGeometry,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn context(gravity: f64) -> Result<Context<CpuRuntime>, ChannelError> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        Context::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "test datum".into(),
                provenance: "independent reservoir head arithmetic references".into(),
                geometry: ChannelGeometry::Rectangular {
                    cells: vec![RectangularChannelCell {
                        id: "cell".into(),
                        length_m: 1.0,
                        width_m: 1.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 1.0,
                        manning_n: 0.0,
                    }],
                },
                lateral_bindings: vec![],
                config: ChannelConfig {
                    gravity,
                    ..ChannelConfig::default()
                },
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![0.5],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )
    }

    fn check_head(bounds: &Bounds<CpuRuntime>, floor_bits: u64, ceiling_bits: u64) -> TestResult {
        let lo = bounds.lo.item::<f64>()?;
        let hi = bounds.hi.item::<f64>()?;
        let floor = f64::from_bits(floor_bits);
        let ceiling = f64::from_bits(ceiling_bits);
        assert!(lo.is_finite() && hi.is_finite());
        assert!(
            lo >= 0.0 && lo <= floor,
            "lower={lo:e}, reference floor={floor:e}"
        );
        assert!(hi >= ceiling, "upper={hi:e}, reference ceiling={ceiling:e}");
        if floor > 0.0 {
            assert!(
                (hi - lo) / ceiling <= 1e-12,
                "head interval [{lo:e}, {hi:e}]"
            );
        }
        if ceiling_bits == 0 {
            assert_eq!(lo.to_bits(), 0);
            assert_eq!(hi.to_bits(), 0);
        }
        Ok(())
    }

    #[test]
    fn independently_weighted_heads_enclose_extreme_exact_input_references() -> TestResult {
        // Independent 150-digit references bracket each exact binary64 input expression.
        let fixtures = [
            (
                -1e-170,
                1.0,
                1e308,
                (0, 1),
                (4123597567603287858, 4123597567603287859),
            ),
            (
                -1e200,
                1e200,
                1.0,
                (7594448965539947098, 7594448965539947098),
                (7594448965539947098, 7594448965539947098),
            ),
            (
                -1e200,
                1e308,
                0.0,
                (5978682040737353251, 5978682040737353252),
                (0, 0),
            ),
        ];
        for (u, gravity, k, velocity_bits, loss_bits) in fixtures {
            let ctx = context(gravity)?;
            let velocity = ctx.tensor(&[u])?;
            let coefficient = ctx.tensor(&[k])?;
            let ReservoirHeads {
                velocity_head: head,
                entrance_loss_head: loss,
                ..
            } = ctx.reservoir_heads(&Bounds::exact(&velocity), &coefficient)?;
            check_head(&head, velocity_bits.0, velocity_bits.1)?;
            check_head(&loss, loss_bits.0, loss_bits.1)?;
        }
        Ok(())
    }

    #[test]
    fn uncertain_velocity_sign_retains_possible_amplified_incoming_loss() -> TestResult {
        let ctx = context(1.0)?;
        let velocity = Bounds {
            lo: ctx.tensor(&[-1e-170])?,
            hi: ctx.tensor(&[1e-170])?,
        };
        let coefficient = ctx.tensor(&[1e308])?;
        let ReservoirHeads {
            entrance_loss_head: loss,
            ..
        } = ctx.reservoir_heads(&velocity, &coefficient)?;
        let lo = loss.lo.item::<f64>()?;
        let hi = loss.hi.item::<f64>()?;
        let reference_ceiling = f64::from_bits(4123597567603287859);
        assert_eq!(lo.to_bits(), 0);
        assert!(hi.is_finite() && hi >= reference_ceiling);
        assert!(hi / reference_ceiling <= 1.0 + 1e-12);
        Ok(())
    }

    #[test]
    fn certified_outflow_selects_exact_zero_incoming_heads_before_large_loss_work() -> TestResult {
        let ctx = context(1e200)?;
        let velocity = ctx.tensor(&[1e200])?;
        let coefficient = ctx.tensor(&[f64::MAX])?;
        let ReservoirHeads {
            velocity_head: head,
            entrance_loss_head: loss,
            dissipation_head: exit,
        } = ctx.reservoir_heads(&Bounds::exact(&velocity), &coefficient)?;
        check_head(&head, 0, 0)?;
        check_head(&loss, 0, 0)?;
        check_head(&exit, 7594448965539947098, 7594448965539947098)?;
        Ok(())
    }
}
