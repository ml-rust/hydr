//! The pool child: accepted storage, trial forcing, attempt operands, and completion.
//!
//! Pool calls that sort or interpolate run through [`super::PoolFns`], bound at construction.

use super::super::context::{Context, Ops};
use super::super::memo::{TrialKey, TrialMemo};
use super::super::{SurfaceChannelError, SurfaceChannelForcingInterval};
use super::CheckedPools;
use super::capacity::PoolOverflow;
use crate::coupling::CoupledLevelPoolError;
use crate::coupling::level_pools::{BoundInflows, InflowState, PreparedInflows};
use crate::storage::{LevelPoolBatch, LevelPoolError, LevelPoolSources};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Accepted pool storage and external inflow ledgers.
#[derive(Clone)]
pub(in super::super) struct PoolChild<R: Runtime<DType = DType>> {
    pub batch: LevelPoolBatch<R>,
    pub external: InflowState<R>,
}

/// Pool operands of one trial.
pub(in super::super) struct PoolStart<R: Runtime<DType = DType>> {
    /// Source rates in ascending pool identity, or `None` without sources.
    pub sources: Option<Vec<LevelPoolSources>>,
    pub inflows: BoundInflows<R>,
}

/// Pool operands of one attempt.
pub(in super::super) struct PoolAttempt<R: Runtime<DType = DType>> {
    pub end: f64,
    /// Post-source water of every pool, `[P]`. Incoming water funds no withdrawal.
    pub available: Tensor<R>,
    /// External inflow volumes and the next inflow ledgers.
    pub external: PreparedInflows<R>,
}

/// A pool candidate with the rounding bound of its incoming merge, in m³.
pub(in super::super) struct PoolCompletion<R: Runtime<DType = DType>> {
    /// Candidate pools, or `None` for a candidate above a top curve knot.
    pub batch: Option<LevelPoolBatch<R>>,
    /// Saturated pool stages of a candidate above a top curve knot.
    pub overflow: Option<PoolOverflow<R>>,
    pub bound_m3: f64,
}

/// The surface/channel error of a pool rejection. Only a numerical rejection is retryable.
pub(in super::super) fn pool_error(error: LevelPoolError) -> SurfaceChannelError {
    match error {
        LevelPoolError::Numerical { .. } => {
            SurfaceChannelError::Numerical(format!("pools: {error}"))
        }
        _ => SurfaceChannelError::Input(format!("pools: {error}")),
    }
}

/// The surface/channel error of a pool inflow rejection.
pub(in super::super) fn inflow_error(error: CoupledLevelPoolError) -> SurfaceChannelError {
    match error {
        CoupledLevelPoolError::Numerical(message) => {
            SurfaceChannelError::Numerical(format!("pool inflows: {message}"))
        }
        CoupledLevelPoolError::Runtime(error) => SurfaceChannelError::Runtime(error),
        CoupledLevelPoolError::Pool(error) => pool_error(error),
        other => SurfaceChannelError::Input(format!("pool inflows: {other}")),
    }
}

/// Outward sum of two nonnegative rounding bounds, in m³.
pub(in super::super) fn charge(a: f64, b: f64) -> Result<f64, SurfaceChannelError> {
    crate::numerics::outward::add_adjustment_charge(a, b)
        .map_err(|error| SurfaceChannelError::Numerical(error.to_string()))
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Pool operands of a trial under the pool forcing of `forcing`.
    ///
    /// Source rates are present exactly when the pools declare source policies. External inputs
    /// cover every declared inflow binding.
    pub fn pool_start(
        &self,
        forcing: &SurfaceChannelForcingInterval,
        pools: Option<&CheckedPools>,
    ) -> Result<Option<PoolStart<R>>, SurfaceChannelError> {
        let (Some(topology), Some(pools)) = (&self.pools, pools) else {
            if forcing.pool_sources.is_some() || !forcing.pool_inflows.is_empty() {
                return Err(SurfaceChannelError::input(
                    "pool forcing without pools: omit pool sources and inflows",
                ));
            }
            return Ok(None);
        };
        let sources = match (&forcing.pool_sources, topology.sources) {
            (None, false) => None,
            (Some(sources), true) if sources.len() == topology.ids.len() => {
                Some(pools.ordered(sources))
            }
            _ => {
                return Err(SurfaceChannelError::input(format!(
                    "pool sources={:?} with source policies={}: supply one source rate set per pool exactly when the pools declare source policies",
                    forcing.pool_sources.as_ref().map(Vec::len),
                    topology.sources
                )));
            }
        };
        let inflows = topology
            .inflows
            .bind_inputs(&self.client, &forcing.pool_inflows)
            .map_err(inflow_error)?;
        Ok(Some(PoolStart { sources, inflows }))
    }

    /// Stage and wet mask of every pool of `batch`, each `[P]`.
    pub fn pool_stages(
        &self,
        batch: &LevelPoolBatch<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), SurfaceChannelError> {
        let topology = self.pool_topology()?;
        let stage = (topology.fns.stages)(batch).map_err(pool_error)?;
        let wet = topology
            .local()
            .wet(&self.client, batch.volumes_tensor())
            .map_err(crate::coupling::links::LinkError::surface_channel)?;
        Ok((stage, wet))
    }

    /// Pool operands of the attempt ending at `end`, the `steps`-th accepted interval.
    ///
    /// Post-source water plus external inflow above a top curve knot rejects with
    /// `pool-capacity-exceeded`. No link exchange fixes it, so no shorter interval retries it.
    pub fn pool_attempt(
        &self,
        child: &PoolChild<R>,
        start: &PoolStart<R>,
        end: f64,
        steps: usize,
    ) -> Result<PoolAttempt<R>, SurfaceChannelError> {
        let topology = self.pool_topology()?;
        let dt = end - child.batch.time();
        let external = topology
            .inflows
            .prepare(&self.client, &start.inflows, &child.external, dt, steps)
            .map_err(inflow_error)?;
        if let Some(sources) = &start.sources {
            self.check_source_capacity(child, sources, &external.incoming, dt)?;
        }
        let available = (topology.fns.available)(&child.batch, end, start.sources.as_deref())
            .map_err(pool_error)?;
        let none = Tensor::<R>::zeros(available.shape(), DType::F64, self.client.device())?;
        if let Some(overflow) = self.overflow(
            &available,
            &none,
            &external.incoming,
            "before any link exchange",
        )? {
            return Err(SurfaceChannelError::Input(overflow.message));
        }
        Ok(PoolAttempt {
            end,
            available,
            external,
        })
    }

    /// Completes the pools with link `incoming` and `outgoing` volumes, each `[P]`.
    ///
    /// External inflows join the link inflows. A candidate above the top curve knot runs no pool
    /// trial: every pool stage saturates at the curve, and the completion holds the overflow. An
    /// iterate may overshoot, so only a converged overflow rejects. A candidate whose operands
    /// repeat a trial kept in `trials` takes the kept result. Keys hold the unclamped operands.
    pub fn complete_pools<S: Clone, C: Clone>(
        &self,
        (child, start, attempt): (&PoolChild<R>, &PoolStart<R>, &PoolAttempt<R>),
        incoming: &Tensor<R>,
        outgoing: &Tensor<R>,
        trials: &TrialMemo<S, C, LevelPoolBatch<R>>,
    ) -> Result<PoolCompletion<R>, SurfaceChannelError> {
        let c = &self.client;
        let topology = self.pool_topology()?;
        let merged = topology
            .inflows
            .merge_incoming(c, &incoming.contiguous()?, &attempt.external)
            .map_err(inflow_error)?;
        let bound_m3 = charge(
            attempt.external.summary.assembly_bound_m3,
            merged.roundoff_bound_m3,
        )?;
        if let Some(overflow) = self.overflow(
            &attempt.available,
            outgoing,
            &merged.volume,
            "at the converged interval",
        )? {
            return Ok(PoolCompletion {
                batch: None,
                overflow: Some(overflow),
                bound_m3,
            });
        }
        let key = TrialKey::of(&c.cat(&[&merged.volume, outgoing], 0)?)?;
        let outgoing = outgoing.contiguous()?;
        let batch = trials.pool(&key, || {
            (topology.fns.trial)(
                &child.batch,
                attempt.end,
                start.sources.as_deref(),
                &merged.volume,
                &outgoing,
            )
            .map_err(pool_error)
        })?;
        Ok(PoolCompletion {
            batch: Some(batch),
            overflow: None,
            bound_m3,
        })
    }

    /// Pre-exchange end stage of every pool, `[P]`: its post-source water plus external inflow,
    /// clamped to its curve, at the curve stage.
    pub fn levelling_pool_stages(
        &self,
        attempt: &PoolAttempt<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        let topology = self.pool_topology()?;
        let available = attempt.available.try_to_vec::<f64>()?;
        let external = attempt.external.incoming.try_to_vec::<f64>()?;
        let stages: Vec<f64> = topology
            .plans
            .iter()
            .zip(available.iter().zip(&external))
            .map(|(plan, (volume, inflow))| plan.stage(volume + inflow))
            .collect();
        Ok(Tensor::<R>::from_slice(
            &stages,
            &[stages.len()],
            self.client.device(),
        )?)
    }
}
