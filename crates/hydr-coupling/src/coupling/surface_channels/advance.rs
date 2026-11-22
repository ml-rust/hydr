//! Lifetime counter additions of one accepted interval.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use super::pool_ends::PoolEndInterval;
use super::ports::PortInterval;
use super::runoff::RunoffAttempt;
use super::state::LinkCounters;
use crate::numerics::water_transfers::upward_add;
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps, TwoSumOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Adds one interval's directional link, port, and runoff volumes with error-free addition
    /// bounds.
    pub fn advance_counters(
        &self,
        counters: &LinkCounters<R>,
        signed: &Tensor<R>,
        (ports, ends): (&PortInterval, &PoolEndInterval),
        runoff: Option<&RunoffAttempt<R>>,
    ) -> Result<LinkCounters<R>, SurfaceChannelError> {
        let runoff = match (&counters.runoff, runoff) {
            (Some(lanes), Some(attempt)) => Some(self.advance_runoff(lanes, attempt)?),
            (None, None) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "runoff counters differ from the runoff delivery".into(),
                ));
            }
        };
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let to_channel = c.where_cond(&c.gt(signed, &zero)?, signed, &zero)?;
        let to_surface = c.where_cond(&c.lt(signed, &zero)?, &c.neg(signed)?, &zero)?;
        let (forward, forward_low) = c.two_sum(&counters.to_channel_m3, &to_channel)?;
        let (backward, backward_low) = c.two_sum(&counters.to_surface_m3, &to_surface)?;
        let low =
            upward_add(c, &c.abs(&forward_low)?, &c.abs(&backward_low)?).map_err(transfer_error)?;
        let roundoff_bound_m3 =
            upward_add(c, &counters.roundoff_bound_m3, &low).map_err(transfer_error)?;
        Ok(LinkCounters {
            to_channel_m3: forward,
            to_surface_m3: backward,
            roundoff_bound_m3,
            ports: counters.ports.advance(ports)?,
            pool_ends: counters.pool_ends.advance(ends)?,
            runoff,
        })
    }
}
