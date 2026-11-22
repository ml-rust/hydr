//! Lifetime delivered volume of every delivery lane, with error-free addition bounds.

use super::attempt::RunoffAttempt;
use super::topology::RunoffTopology;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops, transfer_error};
use crate::numerics::water_transfers::upward_add;
use numr::dtype::DType;
use numr::ops::{TwoSumOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Lifetime delivery counters, each `[N]` in delivery-lane order.
#[derive(Clone)]
pub(in super::super) struct RunoffCounters<R: Runtime<DType = DType>> {
    /// Lifetime delivered volume of every delivery lane, in m³.
    pub delivered_m3: Tensor<R>,
    /// Upward-rounded sum of every addition error of each counter, in m³.
    pub roundoff_bound_m3: Tensor<R>,
    /// Upward-rounded sum of every interval's split bound, in m³.
    pub split_bound_m3: f64,
}

impl<R: Runtime<DType = DType>> RunoffCounters<R> {
    /// Zero counters of every lane of `topology` on `client`.
    pub fn zero(
        client: &R::Client,
        topology: &RunoffTopology<R>,
    ) -> Result<Self, SurfaceChannelError> {
        let zero = Tensor::<R>::zeros(&[topology.lane_count()], DType::F64, client.device())?;
        Ok(Self {
            delivered_m3: zero.clone(),
            roundoff_bound_m3: zero,
            split_bound_m3: 0.0,
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Adds the delivered volumes of `attempt` to `counters`.
    pub fn advance_runoff(
        &self,
        counters: &RunoffCounters<R>,
        attempt: &RunoffAttempt<R>,
    ) -> Result<RunoffCounters<R>, SurfaceChannelError> {
        let c = &self.client;
        let split_bound_m3 = if attempt.split_bound_m3 == 0.0 {
            counters.split_bound_m3
        } else {
            (counters.split_bound_m3 + attempt.split_bound_m3).next_up()
        };
        if attempt.delivered_m3.is_empty() {
            return Ok(RunoffCounters {
                split_bound_m3,
                ..counters.clone()
            });
        }
        let (delivered_m3, low) = c.two_sum(&counters.delivered_m3, &attempt.delivered)?;
        let roundoff_bound_m3 =
            upward_add(c, &counters.roundoff_bound_m3, &c.abs(&low)?).map_err(transfer_error)?;
        Ok(RunoffCounters {
            delivered_m3,
            roundoff_bound_m3,
            split_bound_m3,
        })
    }
}
