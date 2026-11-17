//! Donor-limited link volumes: one shared factor per donor, outward rounded.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use crate::numerics::outward::upward_mul;
use crate::numerics::water_transfers::{self, AggregationKind};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Limited link volumes with their donor totals.
pub(super) struct Exchange<R: Runtime<DType = DType>> {
    /// Weir volume per link, positive surface-to-channel, `[L]`.
    pub proposed: Tensor<R>,
    /// Accepted volume per link, positive surface-to-channel, `[L]`.
    pub signed: Tensor<R>,
    /// Donor factor applied to each link, `[L]`.
    pub factor: Tensor<R>,
    /// Water each donor group receives, `[S + C]`.
    pub incoming: Tensor<R>,
    /// Water each donor group gives, `[S + C]`.
    pub outgoing: Tensor<R>,
    /// Largest aggregation rounding over every group, times the group count, in m³.
    pub aggregation_bound: f64,
}

/// Donor-group sums and their absolute rounding bounds.
struct Sums<R: Runtime<DType = DType>> {
    value: Tensor<R>,
    bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Limits `proposed` so no donor gives more than `available`.
    ///
    /// Each donor's factor is `(available - margin) / (demand + bound)` when demand exceeds it,
    /// rounded down once more. The margin grows with the donor's link count.
    pub fn exchange(
        &self,
        proposed: Tensor<R>,
        available: &Tensor<R>,
    ) -> Result<Exchange<R>, SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let one = self.scalar(1.0)?;
        let tiny = f64::from_bits(1);
        water_transfers::finite(c, &proposed, "weir link volume").map_err(transfer_error)?;
        let (positive, negative) = split(c, &proposed, &zero)?;
        let demands = self.aggregate(&c.cat(&[&positive, &negative], 0)?)?;
        let upper = water_transfers::upward_add(c, &demands.value, &demands.bound)
            .map_err(transfer_error)?;
        let limited = c.gt(&upper, available)?;
        let margin = c.add_scalar(&c.mul(available, &self.donors.degree_margin)?, tiny)?;
        let lower = c.sub(available, &margin)?;
        let has_water = c.gt(available, &zero)?;
        let constrained = c.mul(&limited, &has_water)?;
        self.reject(
            &c.mul(&constrained, &c.le(&lower, &zero)?)?,
            "positive donor water cannot hold a conservative shared allocation: rescale storage",
        )?;
        let numerator = c.where_cond(&constrained, &lower, &zero)?;
        let denominator = c.where_cond(&limited, &upper, &one)?;
        let ratio = c.div(&numerator, &denominator)?;
        let rounding = c.add_scalar(&c.mul_scalar(&c.abs(&ratio)?, f64::EPSILON)?, tiny)?;
        let conservative = c.sub(&ratio, &rounding)?;
        self.reject(
            &c.mul(&constrained, &c.le(&conservative, &zero)?)?,
            "positive donor allocation factor underflows: rescale the competing link demands",
        )?;
        let limited_factor = c.where_cond(&has_water, &conservative, &zero)?;
        let group_factor = c.where_cond(&limited, &limited_factor, &one)?;
        let factor_a = c.index_select(&group_factor, 0, &self.link_a)?;
        let factor_b = c.index_select(&group_factor, 0, &self.link_b)?;
        let factor = c.where_cond(&c.lt(&proposed, &zero)?, &factor_b, &factor_a)?;
        let signed = c.mul(&proposed, &factor)?;
        water_transfers::finite(c, &signed, "accepted link volume").map_err(transfer_error)?;
        self.reject(
            &c.mul(
                &c.mul(&c.gt(&c.abs(&proposed)?, &zero)?, &c.gt(&factor, &zero)?)?,
                &c.eq(&signed, &zero)?,
            )?,
            "positive accepted link volume underflows: rescale the exchange",
        )?;
        let (positive, negative) = split(c, &signed, &zero)?;
        let outgoing = self.aggregate(&c.cat(&[&positive, &negative], 0)?)?;
        let incoming = self.aggregate(&c.cat(&[&negative, &positive], 0)?)?;
        let protected = water_transfers::upward_add(c, &outgoing.value, &outgoing.bound)
            .map_err(transfer_error)?;
        self.reject(
            &c.gt(&protected, available)?,
            "donor totals exceed reserved donor water",
        )?;
        let combined = water_transfers::upward_add(c, &incoming.bound, &outgoing.bound)
            .map_err(transfer_error)?;
        let largest = c.max(&combined, &[0], false)?.item::<f64>()?;
        let groups = (self.surface_count + self.slot_count) as f64;
        let aggregation_bound = if largest == 0.0 {
            0.0
        } else {
            upward_mul(largest, groups)
                .map_err(|error| SurfaceChannelError::Numerical(error.to_string()))?
        };
        Ok(Exchange {
            proposed,
            signed,
            factor,
            incoming: incoming.value,
            outgoing: outgoing.value,
            aggregation_bound,
        })
    }

    /// Per-donor sums of `values` with their rounding bounds.
    fn aggregate(&self, values: &Tensor<R>) -> Result<Sums<R>, SurfaceChannelError> {
        let aggregate = self
            .donors
            .aggregate(&self.client, values, AggregationKind::Water)
            .map_err(transfer_error)?;
        Ok(Sums {
            value: aggregate.value,
            bound: aggregate.bound,
        })
    }

    /// Returns a retryable rejection when any element of `mask` is nonzero.
    pub fn reject(&self, mask: &Tensor<R>, message: &str) -> Result<(), SurfaceChannelError> {
        water_transfers::reject(&self.client, mask, message).map_err(transfer_error)
    }
}

/// Positive parts and negated negative parts of `signed`.
fn split<R: Runtime<DType = DType>>(
    c: &R::Client,
    signed: &Tensor<R>,
    zero: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), SurfaceChannelError>
where
    R::Client: Ops<R>,
{
    Ok((
        c.where_cond(&c.gt(signed, zero)?, signed, zero)?,
        c.where_cond(&c.lt(signed, zero)?, &c.neg(signed)?, zero)?,
    ))
}
