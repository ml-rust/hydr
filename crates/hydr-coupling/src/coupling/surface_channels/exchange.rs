//! Donor-limited link volumes: one shared factor per donor, outward rounded.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use crate::coupling::links::LinkError;
use crate::coupling::links::donors::{self, DonorTexts, Donors};
use crate::numerics::water_transfers::{self, AggregationKind};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Limited link volumes with their donor totals.
pub(super) struct Exchange<R: Runtime<DType = DType>> {
    /// Weir volume per link, positive surface-to-channel, `[L]`.
    pub proposed: Tensor<R>,
    /// Levelling factor applied to each weir volume before the donor limit, `[L]`.
    pub level: Tensor<R>,
    /// Part of `level` that other links impose through a shared node, `[L]`. One where the
    /// link's own cap terms set its factor.
    pub imposed: Tensor<R>,
    /// Accepted volume per link, positive surface-to-channel, `[L]`.
    pub signed: Tensor<R>,
    /// Donor factor applied to each link, `[L]`.
    pub factor: Tensor<R>,
    /// Water each donor group receives, `[S + C + P]`.
    pub incoming: Tensor<R>,
    /// Water each donor group gives, `[S + C + P]`.
    pub outgoing: Tensor<R>,
    /// Largest aggregation rounding over every group, times the group count, in m³.
    pub aggregation_bound: f64,
}

impl<R: Runtime<DType = DType>> Exchange<R> {
    /// The exchange of a coordinator without links. Every tensor is `empty`.
    pub fn empty(empty: &Tensor<R>) -> Self {
        Self {
            proposed: empty.clone(),
            level: empty.clone(),
            imposed: empty.clone(),
            signed: empty.clone(),
            factor: empty.clone(),
            incoming: empty.clone(),
            outgoing: empty.clone(),
            aggregation_bound: 0.0,
        }
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Limits the levelled `weir` volumes so no donor gives more than `available`.
    ///
    /// `level` scales each weir volume first. A factor of exactly one keeps every bit.
    /// `imposed` is the part of `level` other links impose. See [`Context::level`].
    /// Each donor's factor is `(available - margin) / (demand + bound)` when demand exceeds it,
    /// rounded down once more. The margin grows with the donor's link count.
    pub fn exchange(
        &self,
        weir: Tensor<R>,
        (level, imposed): (Tensor<R>, Tensor<R>),
        available: &Tensor<R>,
    ) -> Result<Exchange<R>, SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let one = self.scalar(1.0)?;
        water_transfers::finite(c, &weir, "weir link volume").map_err(transfer_error)?;
        let proposed = c.mul(&weir, &level)?;
        let donors = Donors {
            tree: &self.donors,
            a: &self.link_a,
            b: &self.link_b,
            groups: self.group_count(),
        };
        let limit = donors::limit(c, &donors, &proposed, available, &zero, &one, &TEXTS)
            .map_err(LinkError::surface_channel)?;
        Ok(Exchange {
            proposed: weir,
            level,
            imposed,
            signed: limit.signed,
            factor: limit.factor,
            incoming: limit.incoming,
            outgoing: limit.outgoing,
            aggregation_bound: limit.aggregation_bound,
        })
    }

    /// Per-donor sums of `values`.
    pub(super) fn aggregate(&self, values: &Tensor<R>) -> Result<Tensor<R>, SurfaceChannelError> {
        let aggregate = self
            .donors
            .aggregate(&self.client, values, AggregationKind::Water)
            .map_err(transfer_error)?;
        Ok(aggregate.value)
    }

    /// Returns a retryable rejection when any element of `mask` is nonzero.
    pub fn reject(&self, mask: &Tensor<R>, message: &str) -> Result<(), SurfaceChannelError> {
        water_transfers::reject(&self.client, mask, message).map_err(transfer_error)
    }
}

/// Surface/channel donor-limit rejection texts.
const TEXTS: DonorTexts = DonorTexts {
    unrepresentable: "positive donor water cannot hold a conservative shared allocation: rescale storage",
    factor_underflow: "positive donor allocation factor underflows: rescale the competing link demands",
    accepted: "accepted link volume",
    accepted_underflow: "positive accepted link volume underflows: rescale the exchange",
    totals: "donor totals exceed reserved donor water",
};
