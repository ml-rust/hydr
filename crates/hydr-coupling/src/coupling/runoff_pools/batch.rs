use super::{
    RunoffPoolError, RunoffPoolInitialState, RunoffPoolSetup, RunoffPoolStepDiagnostics,
    context::Context,
};
use crate::coupling::{CoupledLevelPoolBatch, CoupledLevelPoolInitialState};
use crate::runoff::SubcatchmentBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::sync::Arc;

/// Atomic accepted runoff, soil, routing, and receiving-pool state.
#[derive(Clone)]
pub struct RunoffPoolBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) runoff: SubcatchmentBatch<R>,
    pub(super) pools: CoupledLevelPoolBatch<R>,
    pub(super) last: Option<Arc<RunoffPoolStepDiagnostics>>,
}
impl<R: Runtime<DType = DType>> std::fmt::Debug for RunoffPoolBatch<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunoffPoolBatch")
            .field("setup", &self.context.setup)
            .field("initial_time", &self.context.initial_time)
            .field("has_last_step", &self.last.is_some())
            .finish()
    }
}
impl<R: Runtime<DType = DType>> RunoffPoolBatch<R>
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
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Construct both native engines from one configured client and one original epoch.
    pub fn new(
        client: R::Client,
        setup: RunoffPoolSetup,
        initial: RunoffPoolInitialState,
    ) -> Result<Self, RunoffPoolError> {
        let runoff = SubcatchmentBatch::<R>::new(
            client.clone(),
            setup.subcatchments.clone(),
            initial.subcatchments,
            initial.time_s,
            setup.runoff_config.clone(),
        )?;
        let pools = CoupledLevelPoolBatch::<R>::new(
            client.clone(),
            setup.pools.clone(),
            CoupledLevelPoolInitialState {
                time_s: initial.time_s,
                states: initial.pools,
            },
        )?;
        let context = Arc::new(Context::<R>::new(client, setup, initial.time_s)?);
        let result = Self {
            context,
            runoff,
            pools,
            last: None,
        };
        result.ledger()?;
        Ok(result)
    }
    /// Current common accepted time, in seconds.
    pub fn time(&self) -> f64 {
        self.runoff.time()
    }
    /// Immutable original common epoch, in seconds.
    pub fn initial_time(&self) -> f64 {
        self.context.initial_time
    }
    /// Immutable physical setup and common admission controls.
    pub fn setup(&self) -> &RunoffPoolSetup {
        &self.context.setup
    }
    /// Read-only native runoff, routing, soil, and accounting state.
    pub fn runoff(&self) -> &SubcatchmentBatch<R> {
        &self.runoff
    }
    /// Read-only native receiving pools, sources, and boundary ledgers.
    pub fn pools(&self) -> &CoupledLevelPoolBatch<R> {
        &self.pools
    }
    /// Explicitly return the complete final common receipt.
    pub fn last_step(&self) -> Result<Option<RunoffPoolStepDiagnostics>, RunoffPoolError> {
        Ok(self.last.as_ref().map(|s| (**s).clone()))
    }
}
