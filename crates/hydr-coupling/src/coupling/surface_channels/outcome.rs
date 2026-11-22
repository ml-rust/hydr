//! Link and pool-end results of one attempted common interval.

use super::SurfaceChannelError;
use super::accounting::PoolInflowStep;
use super::context::{Context, Ops};
use super::exchange::Exchange;
use super::iteration::Solved;
use super::pool_ends::PoolEndInterval;
use super::pools::{PoolAttempt, PoolChild};
use crate::channel::ChannelReach;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::IndexingOps;
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

/// Candidate children and link results of a converged fixed point.
pub(super) struct Converged<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    /// Advanced linked reaches in `Context::linked` order.
    pub linked: Vec<ChannelReach<R>>,
    /// Advanced pool-ended reaches without links, in `PoolEndTopology::ended` order.
    pub ended: Vec<ChannelReach<R>>,
    /// Candidate end stages and wet masks at every node, for decision samples.
    pub nodes: (Tensor<R>, Tensor<R>),
    pub outcome: LinkOutcome<R>,
}

impl<R: Runtime<DType = DType>> Converged<R> {
    /// Splits the accepted evaluation of `solved` into its children and link results.
    ///
    /// `pools` holds the attempt's pool operands, whose external inflow the candidate pools
    /// take.
    pub fn new(
        context: &Context<R>,
        solved: Solved<R>,
        pools: Option<&PoolAttempt<R>>,
    ) -> Result<Self, SurfaceChannelError>
    where
        R::Client: Ops<R>,
    {
        let evaluation = solved.evaluation;
        let (n, m) = (context.group_count(), context.transfer_count());
        let nodes = (
            evaluation.packet.narrow(0, 0, n)?,
            evaluation.packet.narrow(0, n + m, n)?,
        );
        let pools = match (evaluation.pools, pools) {
            (Some(batch), Some(pool_attempt)) => {
                let summary = &pool_attempt.external.summary;
                Some((
                    PoolChild {
                        batch,
                        external: pool_attempt.external.next.clone(),
                    },
                    PoolInflowStep {
                        volume_m3: summary.volume_m3,
                        bound_m3: summary.reduction_bound_m3 + summary.assembly_bound_m3,
                    },
                ))
            }
            _ => None,
        };
        let host = |values: Option<Tensor<R>>| -> Result<Vec<f64>, SurfaceChannelError> {
            Ok(values
                .map(|v| v.try_to_vec::<f64>())
                .transpose()?
                .unwrap_or_default())
        };
        let pool_ends = PoolEndInterval {
            volume_m3: host(evaluation.end_volume)?,
            head_m: host(evaluation.end_heads)?,
        };
        let c = &context.client;
        let outcome = LinkOutcome {
            iterations: solved.iterations,
            contraction: solved.contraction,
            head_residual_m: solved.residuals.head_m,
            transfer_residual_m3: solved.residuals.transfer_m3,
            midpoint_surface_m: c.index_select(&evaluation.midpoint, 0, &context.link_a)?,
            midpoint_channel_m: c.index_select(&evaluation.midpoint, 0, &context.link_b)?,
            regime: evaluation.regime,
            exchange: evaluation.exchange,
            pools,
            pool_ends,
        };
        Ok(Self {
            surface: evaluation.surface,
            linked: evaluation.linked,
            ended: evaluation.ended,
            nodes,
            outcome,
        })
    }
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
