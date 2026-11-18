//! Link and pool-end results of one attempted common interval.

use super::SurfaceChannelError;
use super::accounting::PoolInflowStep;
use super::context::Context;
use super::exchange::Exchange;
use super::pool_ends::PoolEndInterval;
use super::pools::PoolChild;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Link results of one attempt: the converged fixed point, or nothing without links and pool
/// ends.
pub(super) struct LinkOutcome<R: Runtime<DType = DType>> {
    pub iterations: usize,
    pub contraction: f64,
    pub head_residual_m: f64,
    pub transfer_residual_m3: f64,
    pub midpoint_surface_m: Tensor<R>,
    pub midpoint_channel_m: Tensor<R>,
    pub regime: Tensor<R>,
    pub exchange: Exchange<R>,
    /// Candidate pools with the external inflow of the interval, or `None` without pools.
    pub pools: Option<(PoolChild<R>, PoolInflowStep)>,
    /// Pool-end volumes and reservoir heads. Empty without pool ends.
    pub pool_ends: PoolEndInterval,
}

impl<R: Runtime<DType = DType>> LinkOutcome<R> {
    /// The outcome of a coordinator without links and pool ends: no update and empty tensors.
    pub fn empty(context: &Context<R>) -> Result<Self, SurfaceChannelError> {
        let empty = Tensor::<R>::zeros(&[0], DType::F64, context.client.device())?;
        Ok(Self {
            iterations: 0,
            contraction: 0.0,
            head_residual_m: 0.0,
            transfer_residual_m3: 0.0,
            midpoint_surface_m: empty.clone(),
            midpoint_channel_m: empty.clone(),
            regime: empty.clone(),
            exchange: Exchange::empty(&empty),
            pools: None,
            pool_ends: PoolEndInterval::default(),
        })
    }
}
