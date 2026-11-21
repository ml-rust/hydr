use super::inflows::InflowState;
use super::{
    CoupledLevelPoolError, CoupledLevelPoolInitialState, CoupledLevelPoolSetup, context::Context,
    trial::StoredStep,
};
use crate::storage::{LevelPoolBatch, LevelPoolSourcePolicy};
use crate::structures::CulvertClient;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use std::sync::Arc;

/// Accepted hydraulic storage with explicit physical connections and forcing.
#[derive(Clone)]
pub struct CoupledLevelPoolBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) pools: LevelPoolBatch<R>,
    pub(super) external: InflowState<R>,
    pub(super) last: Option<Arc<StoredStep<R>>>,
}
impl<R: Runtime<DType = DType>> CoupledLevelPoolBatch<R> {
    pub(crate) fn matches_setup(&self, setup: &CoupledLevelPoolSetup) -> bool {
        super::identity::same_setup(&self.context.setup, setup)
    }
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
        + CorrectlyRoundedOps<R>
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
    ///
    /// A culvert law needs certified roots and quadrature on `client`, which
    /// [`CulvertClient`] names.
    pub fn new(
        client: R::Client,
        setup: CoupledLevelPoolSetup,
        initial: CoupledLevelPoolInitialState,
    ) -> Result<Self, CoupledLevelPoolError>
    where
        R::Client: CulvertClient<R>,
    {
        let pools = match setup.source_policies.clone() {
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
        let external = context.inflows.zero_state(&context.client)?;
        Ok(Self {
            context,
            pools,
            external,
            last: None,
        })
    }
    /// Native immutable source-enabling evidence, absent for exchange-only pools.
    pub fn source_policies(&self) -> Option<&[LevelPoolSourcePolicy]> {
        self.pools.source_policies()
    }
    /// Explicitly read cumulative external volume and addition bounds in boundary order.
    pub fn external_inflows(
        &self,
    ) -> Result<Vec<super::CoupledPoolInflowLedger>, CoupledLevelPoolError> {
        self.context
            .inflows
            .project_state(&self.context.client, &self.external)
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
