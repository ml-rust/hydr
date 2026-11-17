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
#[path = "reservoir_head_tests.rs"]
mod tests;
