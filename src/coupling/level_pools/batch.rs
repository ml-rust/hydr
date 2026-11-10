use super::{
    CoupledLevelPoolError, CoupledLevelPoolInitialState, CoupledLevelPoolSetup, context::Context,
    trial::StoredStep,
};
use crate::storage::{LevelPoolBatch, LevelPoolSourcePolicy};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use std::sync::Arc;

/// Accepted hydraulic storage with explicit physical connections and optional local sources.
#[derive(Clone)]
pub struct CoupledLevelPoolBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) pools: LevelPoolBatch<R>,
    pub(super) last: Option<Arc<StoredStep<R>>>,
}
impl<R: Runtime<DType = DType>> std::fmt::Debug for CoupledLevelPoolBatch<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoupledLevelPoolBatch")
            .field("setup", &self.context.setup)
            .field("has_last_step", &self.last.is_some())
            .finish()
    }
}
impl<R: Runtime<DType = DType>> CoupledLevelPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Construct declared physical pools, static operations, and accepted storage.
    pub fn new(
        client: R::Client,
        setup: CoupledLevelPoolSetup,
        initial: CoupledLevelPoolInitialState,
    ) -> Result<Self, CoupledLevelPoolError> {
        Self::construct(client, setup, initial, None)
    }
    /// Enable declared local sources while retaining the selected runtime client.
    pub fn new_with_sources(
        client: R::Client,
        setup: CoupledLevelPoolSetup,
        initial: CoupledLevelPoolInitialState,
        policies: Vec<LevelPoolSourcePolicy>,
    ) -> Result<Self, CoupledLevelPoolError> {
        Self::construct(client, setup, initial, Some(policies))
    }
    fn construct(
        client: R::Client,
        setup: CoupledLevelPoolSetup,
        initial: CoupledLevelPoolInitialState,
        policies: Option<Vec<LevelPoolSourcePolicy>>,
    ) -> Result<Self, CoupledLevelPoolError> {
        let pools = match policies {
            None => LevelPoolBatch::<R>::new(
                client.clone(),
                setup.pools.clone(),
                initial.states,
                initial.time_s,
                setup.config.storage.clone(),
            )?,
            Some(policies) => LevelPoolBatch::<R>::new_with_sources(
                client.clone(),
                setup.pools.clone(),
                initial.states,
                initial.time_s,
                setup.config.storage.clone(),
                policies,
            )?,
        };
        let context = Arc::new(Context::<R>::new(client, setup)?);
        Ok(Self {
            context,
            pools,
            last: None,
        })
    }
    /// Native immutable source-enabling evidence, absent for exchange-only pools.
    pub fn source_policies(&self) -> Option<&[LevelPoolSourcePolicy]> {
        self.pools.source_policies()
    }
    /// Accepted time in seconds.
    pub fn time(&self) -> f64 {
        self.pools.time()
    }
    /// Accepted storage and independently checked pool accounting.
    pub fn pools(&self) -> &LevelPoolBatch<R> {
        &self.pools
    }
    /// Immutable physical definitions and dimensional controls.
    pub fn setup(&self) -> &CoupledLevelPoolSetup {
        &self.context.setup
    }
}
