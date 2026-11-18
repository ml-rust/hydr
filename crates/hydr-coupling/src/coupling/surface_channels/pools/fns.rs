//! Pool calls that sort or interpolate, bound once at construction.
//!
//! [`PoolParts::new`] binds them where the client supplies those operations, so the trial,
//! projection, and import bounds of the coordinator need no sorting or interpolation.

use super::super::SurfaceChannelError;
use super::{CheckedPools, pool_error};
use crate::coupling::links::pool_plan::PoolPlan;
use crate::storage::{
    LevelPoolBatch, LevelPoolError, LevelPoolSnapshotData, LevelPoolSources, LevelPoolState,
    LevelPoolTotals,
};
use hydr_storage::internal::LevelPoolCut;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

/// Source rates in ascending pool identity, or `None` without sources.
pub(in super::super) type Sources<'a> = Option<&'a [LevelPoolSources]>;

/// A pool call result.
type PoolResult<T> = Result<T, LevelPoolError>;

/// Post-source water of an interval ending at the given time.
type AvailableFn<R> = fn(&LevelPoolBatch<R>, f64, Sources<'_>) -> PoolResult<Tensor<R>>;

/// The accepted candidate of an interval with incoming and outgoing volumes.
type TrialFn<R> = fn(
    &LevelPoolBatch<R>,
    f64,
    Sources<'_>,
    &Tensor<R>,
    &Tensor<R>,
) -> PoolResult<LevelPoolBatch<R>>;

/// Upper volume bound at every cut.
type CutVolumesFn<R> = fn(&LevelPoolBatch<R>, &[LevelPoolCut]) -> PoolResult<Tensor<R>>;

/// A copy of the pools restored from a caller-owned package.
type RestoredFn<R> =
    fn(&LevelPoolBatch<R>, &LevelPoolSnapshotData) -> PoolResult<LevelPoolBatch<R>>;

/// Pool calls that sort or interpolate, bound to one runtime.
pub(in super::super) struct PoolFns<R: Runtime<DType = DType>> {
    /// Accepted stage of every pool. A dry pool takes its curve origin.
    pub stages: fn(&LevelPoolBatch<R>) -> PoolResult<Tensor<R>>,
    /// Checked aggregate accounting.
    pub totals: fn(&LevelPoolBatch<R>) -> PoolResult<LevelPoolTotals>,
    pub available: AvailableFn<R>,
    pub trial: TrialFn<R>,
    pub cut_volumes: CutVolumesFn<R>,
    /// Complete caller-owned package.
    pub snapshot: fn(&LevelPoolBatch<R>) -> PoolResult<LevelPoolSnapshotData>,
    /// A copy restored from a checked caller-owned package.
    pub restored: RestoredFn<R>,
}

impl<R: Runtime<DType = DType>> Clone for PoolFns<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime<DType = DType>> Copy for PoolFns<R> {}

/// The pool child and its bound calls at construction.
pub(in super::super) struct PoolParts<R: Runtime<DType = DType>> {
    pub batch: LevelPoolBatch<R>,
    pub fns: PoolFns<R>,
    /// Plan areas of every pool in ascending identity.
    pub plans: Vec<PoolPlan>,
}

impl<R: Runtime<DType = DType>> PoolParts<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    /// The pool child of `pools` at `states`, in setup pool order, and accepted time `time`.
    pub fn new(
        client: R::Client,
        pools: &CheckedPools,
        states: &[LevelPoolState],
        time: f64,
    ) -> Result<Self, SurfaceChannelError> {
        let count = pools.ids.len();
        if states.len() != count {
            return Err(SurfaceChannelError::input(format!(
                "initial pool state count={}, pools={count}: supply one state per pool in pool order",
                states.len()
            )));
        }
        let elements = pools.elements();
        let states = pools.ordered(states);
        let storage = pools.declared.storage.clone();
        let batch = match &pools.declared.sources {
            None => LevelPoolBatch::new(client, elements, states, time, storage),
            Some(policies) => LevelPoolBatch::new_with_sources(
                client,
                elements,
                states,
                time,
                storage,
                pools.ordered(policies),
            ),
        }
        .map_err(pool_error)?;
        let plans = batch
            .elements()
            .iter()
            .map(|e| PoolPlan::new(&e.curve))
            .collect();
        Ok(Self {
            batch,
            fns: PoolFns {
                stages: stages::<R>,
                totals: totals::<R>,
                available: available::<R>,
                trial: trial::<R>,
                cut_volumes: cut_volumes::<R>,
                snapshot: snapshot::<R>,
                restored: restored::<R>,
            },
            plans,
        })
    }
}

fn stages<R: Runtime<DType = DType>>(batch: &LevelPoolBatch<R>) -> Result<Tensor<R>, LevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    batch.stages_tensor()
}

fn totals<R: Runtime<DType = DType>>(
    batch: &LevelPoolBatch<R>,
) -> Result<LevelPoolTotals, LevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    batch.totals()
}

fn available<R: Runtime<DType = DType>>(
    batch: &LevelPoolBatch<R>,
    end: f64,
    sources: Sources<'_>,
) -> Result<Tensor<R>, LevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    match sources {
        None => Ok(batch.volumes_tensor().clone()),
        Some(sources) => {
            let plan = batch.bind_source_rates(sources)?;
            Ok(plan.prepare(end)?.available_volume().clone())
        }
    }
}

fn trial<R: Runtime<DType = DType>>(
    batch: &LevelPoolBatch<R>,
    end: f64,
    sources: Sources<'_>,
    incoming: &Tensor<R>,
    outgoing: &Tensor<R>,
) -> Result<LevelPoolBatch<R>, LevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    Ok(match sources {
        None => batch.trial_to_tensors(end, incoming, outgoing)?.into_next(),
        Some(sources) => {
            let plan = batch.bind_source_rates(sources)?;
            plan.prepare(end)?
                .trial_to_tensors(incoming, outgoing)?
                .into_next()
        }
    })
}

fn cut_volumes<R: Runtime<DType = DType>>(
    batch: &LevelPoolBatch<R>,
    cuts: &[LevelPoolCut],
) -> Result<Tensor<R>, LevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    Ok(batch.bind_coupling_cuts(cuts)?.upper_volume_m3().clone())
}

fn snapshot<R: Runtime<DType = DType>>(
    batch: &LevelPoolBatch<R>,
) -> Result<LevelPoolSnapshotData, LevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    batch.snapshot_data()
}

fn restored<R: Runtime<DType = DType>>(
    batch: &LevelPoolBatch<R>,
    data: &LevelPoolSnapshotData,
) -> Result<LevelPoolBatch<R>, LevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    let mut batch = batch.clone();
    batch.restore_data(data)?;
    Ok(batch)
}
