//! Pool wet flags, endpoint heads, midpoint heads, and the raised-cutoff crossing check.

use super::{LinkError, LinkOps};
use crate::numerics::water_transfers::{finite, reject};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, IndexingOps, MidpointOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Endpoint heads and wet flags of every link.
#[derive(Clone, Debug)]
pub(crate) struct Heads<R: Runtime<DType = DType>> {
    /// Stage at endpoint A, `[L]`.
    pub a: Tensor<R>,
    /// Stage at endpoint B, `[L]`.
    pub b: Tensor<R>,
    /// One where endpoint A's pool is wet and its stage is above the attachment bed, `[L]`.
    pub wet_a: Tensor<R>,
    /// One where endpoint B's pool is wet and its stage is above the attachment bed, `[L]`.
    pub wet_b: Tensor<R>,
}

/// Pool attachments of every link.
pub(crate) struct PoolAttachments<'a, R: Runtime<DType = DType>> {
    /// Pool of endpoint A per link, I64 `[L]`.
    pub a: &'a Tensor<R>,
    /// Pool of endpoint B per link, I64 `[L]`.
    pub b: &'a Tensor<R>,
    /// Attachment bed at endpoint A, `[L]`.
    pub bed_a: &'a Tensor<R>,
    /// Attachment bed at endpoint B, `[L]`.
    pub bed_b: &'a Tensor<R>,
    /// Law cutoff at endpoint A, `[L]`.
    pub cutoff_a: &'a Tensor<R>,
    /// Law cutoff at endpoint B, `[L]`.
    pub cutoff_b: &'a Tensor<R>,
    /// Zero-volume stage per pool, `[P]`.
    pub origin: &'a Tensor<R>,
    /// Scalar zero.
    pub zero: &'a Tensor<R>,
}

impl<R: Runtime<DType = DType>> PoolAttachments<'_, R>
where
    R::Client: LinkOps<R>,
{
    /// One where a pool holds positive water.
    pub fn wet(&self, c: &R::Client, volume: &Tensor<R>) -> Result<Tensor<R>, LinkError> {
        Ok(c.gt(volume, self.zero)?)
    }

    /// Endpoint heads at pool `stages`. An endpoint is wet when its pool is wet and its stage is
    /// above the attachment bed.
    pub fn heads(
        &self,
        c: &R::Client,
        stages: &Tensor<R>,
        wet: &Tensor<R>,
    ) -> Result<Heads<R>, LinkError> {
        let a = c.index_select(stages, 0, self.a)?;
        let b = c.index_select(stages, 0, self.b)?;
        let wa = c.index_select(wet, 0, self.a)?;
        let wb = c.index_select(wet, 0, self.b)?;
        Ok(Heads {
            wet_a: c.mul(&wa, &c.gt(&a, self.bed_a)?)?,
            wet_b: c.mul(&wb, &c.gt(&b, self.bed_b)?)?,
            a,
            b,
        })
    }

    /// Endpoint heads at the temporal midpoint stage. A pool wet at either end is wet.
    ///
    /// Rejects a wet pool whose midpoint stage is at or below its origin.
    pub fn midpoint_heads(
        &self,
        c: &R::Client,
        start: &Tensor<R>,
        end: &Tensor<R>,
        start_wet: &Tensor<R>,
        end_wet: &Tensor<R>,
    ) -> Result<Heads<R>, LinkError>
    where
        R::Client: MidpointOps<R>,
    {
        let stage = c.midpoint(start, end)?;
        finite(c, &stage, "temporal midpoint stage")?;
        let wet = c.maximum(start_wet, end_wet)?;
        reject(
            c,
            &c.mul(&wet, &c.le(&stage, self.origin)?)?,
            "positive pool water has no represented midpoint depth. Rescale the vertical reference",
        )?;
        self.heads(c, &stage, &wet)
    }

    /// Rejects an active `signed` transfer whose donor end head sits at or below a cutoff raised
    /// above its pool origin.
    pub fn check_raised_crossing(
        &self,
        c: &R::Client,
        heads: &Heads<R>,
        signed: &Tensor<R>,
    ) -> Result<(), LinkError> {
        let crossing_a = c.mul(&c.gt(signed, self.zero)?, &c.le(&heads.a, self.cutoff_a)?)?;
        let crossing_b = c.mul(&c.lt(signed, self.zero)?, &c.le(&heads.b, self.cutoff_b)?)?;
        let origin_a = c.index_select(self.origin, 0, self.a)?;
        let origin_b = c.index_select(self.origin, 0, self.b)?;
        let raised_a = c.gt(self.cutoff_a, &origin_a)?;
        let raised_b = c.gt(self.cutoff_b, &origin_b)?;
        reject(
            c,
            &c.maximum(
                &c.mul(&crossing_a, &raised_a)?,
                &c.mul(&crossing_b, &raised_b)?,
            )?,
            "active transfer crosses a raised local bed or law cutoff. Shorten the interval",
        )?;
        Ok(())
    }
}
