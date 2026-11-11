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
            if self.all(&c.gt(&h.lo, &interior.depth_bounds.hi)?)? {
                self.boundary_require(
                    &c.lt(&lambda_b.hi, &shock.lo)?,
                    "compatible compression lacks the trace Lax inequality",
                )?;
                self.boundary_require(
                    &c.lt(&shock.hi, &lambda_i.lo)?,
                    "compatible compression lacks the interior Lax inequality",
                )?;
            }
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
