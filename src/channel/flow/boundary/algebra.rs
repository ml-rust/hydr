use crate::channel::flow::{
    ChannelError,
    context::Context,
    triangular::envelope::{lower, upper},
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

#[derive(Clone)]
pub(super) struct Bounds<R: Runtime<DType = DType>> {
    pub lo: Tensor<R>,
    pub hi: Tensor<R>,
}

#[derive(Clone)]
pub(super) struct Interior<R: Runtime<DType = DType>> {
    pub depth: Tensor<R>,
    pub velocity: Tensor<R>,
    pub depth_bounds: Bounds<R>,
    pub velocity_bounds: Bounds<R>,
    pub area: Tensor<R>,
    pub discharge: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Bounds<R> {
    pub fn exact(x: &Tensor<R>) -> Self {
        Self {
            lo: x.clone(),
            hi: x.clone(),
        }
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    pub(super) fn badd(&self, a: &Bounds<R>, b: &Bounds<R>) -> Result<Bounds<R>, ChannelError> {
        Ok(Bounds {
            lo: lower::<R>(&self.client, &self.client.add(&a.lo, &b.lo)?)?,
            hi: upper::<R>(&self.client, &self.client.add(&a.hi, &b.hi)?)?,
        })
    }
    pub(super) fn bsub(&self, a: &Bounds<R>, b: &Bounds<R>) -> Result<Bounds<R>, ChannelError> {
        Ok(Bounds {
            lo: lower::<R>(&self.client, &self.client.sub(&a.lo, &b.hi)?)?,
            hi: upper::<R>(&self.client, &self.client.sub(&a.hi, &b.lo)?)?,
        })
    }
    pub(super) fn bmul(&self, a: &Bounds<R>, b: &Bounds<R>) -> Result<Bounds<R>, ChannelError> {
        let c = &self.client;
        let v = [
            c.mul(&a.lo, &b.lo)?,
            c.mul(&a.lo, &b.hi)?,
            c.mul(&a.hi, &b.lo)?,
            c.mul(&a.hi, &b.hi)?,
        ];
        Ok(Bounds {
            lo: lower::<R>(
                c,
                &c.minimum(&c.minimum(&v[0], &v[1])?, &c.minimum(&v[2], &v[3])?)?,
            )?,
            hi: upper::<R>(
                c,
                &c.maximum(&c.maximum(&v[0], &v[1])?, &c.maximum(&v[2], &v[3])?)?,
            )?,
        })
    }
    pub(super) fn bdiv(&self, a: &Bounds<R>, b: &Bounds<R>) -> Result<Bounds<R>, ChannelError> {
        let c = &self.client;
        if !self.all(&c.gt(&b.lo, &self.constant(&b.lo, 0.0)?)?)? {
            let lower = c.min(&b.lo, &[0], false)?.item::<f64>()?;
            let upper = c.max(&b.hi, &[0], false)?.item::<f64>()?;
            return Err(ChannelError::numerical(format!(
                "scalar boundary positive interval denominator, lower={lower}, upper={upper}: supply a resolved positive denominator or rescale the conserved state"
            )));
        }
        let values = [
            c.div(&a.lo, &b.lo)?,
            c.div(&a.lo, &b.hi)?,
            c.div(&a.hi, &b.lo)?,
            c.div(&a.hi, &b.hi)?,
        ];
        Ok(Bounds {
            lo: lower::<R>(
                c,
                &c.minimum(
                    &c.minimum(&values[0], &values[1])?,
                    &c.minimum(&values[2], &values[3])?,
                )?,
            )?,
            hi: upper::<R>(
                c,
                &c.maximum(
                    &c.maximum(&values[0], &values[1])?,
                    &c.maximum(&values[2], &values[3])?,
                )?,
            )?,
        })
    }
    pub(super) fn bsqrt(&self, a: &Bounds<R>) -> Result<Bounds<R>, ChannelError> {
        let c = &self.client;
        let z = self.constant(&a.lo, 0.0)?;
        Ok(Bounds {
            lo: c.maximum(&lower::<R>(c, &c.sqrt(&c.maximum(&a.lo, &z)?)?)?, &z)?,
            hi: upper::<R>(c, &c.sqrt(&c.maximum(&a.hi, &z)?)?)?,
        })
    }
    pub(super) fn bscale(&self, a: &Bounds<R>, s: f64) -> Result<Bounds<R>, ChannelError> {
        self.bmul(a, &Bounds::exact(&self.constant(&a.lo, s)?))
    }
    pub(super) fn babs(&self, a: &Bounds<R>) -> Result<Tensor<R>, ChannelError> {
        Ok(self
            .client
            .maximum(&self.client.abs(&a.lo)?, &self.client.abs(&a.hi)?)?)
    }
    pub(super) fn boundary_require(
        &self,
        mask: &Tensor<R>,
        message: &str,
    ) -> Result<(), ChannelError> {
        if !self.all(mask)? {
            return Err(ChannelError::numerical(format!(
                "scalar channel boundary: {message}. Supply a resolved wet subcritical target or rescale inputs"
            )));
        }
        Ok(())
    }
}
