//! Atomic persistent continuation with lazy complete output projections.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::accounting::{StepTotals, aggregate_step, check_ledger};
use super::projection::project_step;
use super::{
    SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentError, SubcatchmentInput,
    SubcatchmentLedger, SubcatchmentOutput, SubcatchmentState, SubcatchmentStepDiagnostics,
};
use crate::numerics::subcatchment::{self, TensorSubcatchmentStep};
use crate::tensor::subcatchment::{
    TensorSubcatchmentContext, TensorSubcatchmentInput, TensorSubcatchmentState,
};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub start: f64,
    pub end: f64,
    pub step: TensorSubcatchmentStep<R>,
    pub totals: StepTotals,
}

/// Complete immutable three-subarea component state on persistent configured runtime tensors.
#[derive(Clone, Debug)]
pub struct SubcatchmentBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<TensorSubcatchmentContext<R>>,
    pub(super) state: TensorSubcatchmentState<R>,
    pub(super) time: f64,
    pub(super) ledger: SubcatchmentLedger,
    pub(super) steps: usize,
    pub(super) last: Option<Arc<StoredStep<R>>>,
}

/// Complete candidate storage and both native-soil roles. Discarding it leaves accepted state unchanged.
#[derive(Debug)]
pub struct SubcatchmentTrial<R: Runtime<DType = DType>> {
    pub(super) next: SubcatchmentBatch<R>,
}

impl<R> SubcatchmentBatch<R>
where
    R: Runtime<DType = DType>,
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Construct explicit checked geometry, antecedent state, time, and numerical controls.
    pub fn new(
        client: R::Client,
        definitions: Vec<SubcatchmentDefinition>,
        states: Vec<SubcatchmentState>,
        time: f64,
        config: SubcatchmentConfig,
    ) -> Result<Self, SubcatchmentError> {
        if !time.is_finite() || time < 0.0 {
            return Err(SubcatchmentError::input(
                "accepted time must be finite and nonnegative",
            ));
        }
        let context = Arc::new(TensorSubcatchmentContext::<R>::new(
            client,
            definitions,
            config,
        )?);
        let state = TensorSubcatchmentState::<R>::new(&context, &states)?;
        let storage = context.storage_volume(&state.depth)?;
        let lower = subcatchment::initial_storage_scale_lower(&context, &state)?;
        let ledger = SubcatchmentLedger::new(storage, lower);
        check_ledger(&context, &ledger, 0)?;
        Ok(Self {
            context,
            state,
            time,
            ledger,
            steps: 0,
            last: None,
        })
    }
    /// Immutable ordered physical definitions.
    pub fn definitions(&self) -> &[SubcatchmentDefinition] {
        &self.context.definitions
    }
    /// Immutable numerical and separate soil-adjustment controls.
    pub fn config(&self) -> &SubcatchmentConfig {
        &self.context.config
    }
    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Complete accepted physical and uncertainty accounting.
    pub fn ledger(&self) -> &SubcatchmentLedger {
        &self.ledger
    }
    /// Number of accepted intervals in this component history.
    pub fn accepted_steps(&self) -> usize {
        self.steps
    }
    /// Explicitly project complete accepted depths and native memory.
    pub fn states(&self) -> Result<Vec<SubcatchmentState>, SubcatchmentError> {
        self.state.project(&self.context)
    }
    /// Explicitly project all last accepted source and paired-interface receipts.
    pub fn last_step(&self) -> Result<Option<SubcatchmentStepDiagnostics>, SubcatchmentError> {
        self.last
            .as_ref()
            .map(|s| project_step(&self.context, s))
            .transpose()
    }
    /// Advance the exact represented interval without mutating any accepted state or counter.
    pub fn trial_to(
        &self,
        end_time: f64,
        inputs: &[SubcatchmentInput],
    ) -> Result<SubcatchmentTrial<R>, SubcatchmentError> {
        let dt = end_time - self.time;
        if !end_time.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return Err(SubcatchmentError::input(format!(
                "time={},end={end_time},dt={dt}: require finite strictly increasing time",
                self.time
            )));
        }
        let input = TensorSubcatchmentInput::<R>::new(&self.context, inputs)?;
        let (state, step) = subcatchment::advance(&self.context, &self.state, dt, &input)?;
        subcatchment::check_step(&self.context, &step, dt)?;
        let totals = aggregate_step(&self.context, &step)?;
        let steps = self.steps.checked_add(1).ok_or_else(|| {
            SubcatchmentError::numerical(
                "accepted-step counter overflows. Start another component history",
            )
        })?;
        let ledger = self.ledger.advance(&self.context, &step, &totals, steps)?;
        Ok(SubcatchmentTrial {
            next: Self {
                context: Arc::clone(&self.context),
                state,
                time: end_time,
                ledger,
                steps,
                last: Some(Arc::new(StoredStep {
                    start: self.time,
                    end: end_time,
                    step,
                    totals,
                })),
            },
        })
    }
}

impl<R> SubcatchmentTrial<R>
where
    R: Runtime<DType = DType>,
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Explicitly project supported-class interval output without accepting the candidate.
    pub fn outputs(&self) -> Result<Vec<SubcatchmentOutput>, SubcatchmentError> {
        super::output::outputs(self.diagnostics()?.receipts)
    }
    /// Explicitly project complete candidate source, transfer, and drainage evidence.
    pub fn diagnostics(&self) -> Result<SubcatchmentStepDiagnostics, SubcatchmentError> {
        let stored = self.next.last.as_ref().ok_or_else(|| {
            SubcatchmentError::numerical("candidate lacks complete interval receipts")
        })?;
        project_step(&self.next.context, stored)
    }
    /// Accept complete storage, both native-soil roles, counters, and receipts by ownership without projection.
    pub fn into_next(self) -> SubcatchmentBatch<R> {
        self.next
    }
}
