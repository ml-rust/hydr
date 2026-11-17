//! Immutable tensor continuation and lazy application output boundaries.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

use super::accounting::aggregate_step;
use super::output::project_step;
use super::{
    ReservoirConfig, ReservoirElement, ReservoirError, ReservoirInput, ReservoirLedger,
    ReservoirOutput, ReservoirState, ReservoirStepDiagnostics,
};
use crate::runoff::numerics::reservoir::TensorReservoirStep;
use crate::runoff::tensor::reservoir_storage::{TensorReservoirContext, TensorReservoirState};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub start: f64,
    pub end: f64,
    pub step: TensorReservoirStep<R>,
}

/// Persistent standalone SI reservoir batch on an explicit configured numr runtime client.
#[derive(Clone, Debug)]
pub struct ReservoirBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<TensorReservoirContext<R>>,
    pub(super) state: TensorReservoirState<R>,
    pub(super) time: f64,
    pub(super) ledger: ReservoirLedger,
    pub(super) steps: usize,
    pub(super) last: Option<Arc<StoredStep<R>>>,
}

/// Complete immutable candidate. Acceptance moves ownership without projecting tensors.
#[derive(Debug)]
pub struct ReservoirTrial<R: Runtime<DType = DType>> {
    next: ReservoirBatch<R>,
}

impl<R> ReservoirBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Construct explicit storage with checked immutable support and complete antecedent depths.
    pub fn new(
        client: R::Client,
        elements: Vec<ReservoirElement>,
        states: Vec<ReservoirState>,
        time: f64,
        config: ReservoirConfig,
    ) -> Result<Self, ReservoirError> {
        if !time.is_finite() || time < 0.0 {
            return Err(ReservoirError::input(format!(
                "time={time}: require finite nonnegative accepted time"
            )));
        }
        let context = Arc::new(TensorReservoirContext::<R>::new(client, elements, config)?);
        let state = TensorReservoirState::<R>::new(&context, &states)?;
        let storage = context.storage_volume(&state.depth)?;
        Ok(Self {
            context,
            state,
            time,
            ledger: ReservoirLedger::new(storage),
            steps: 0,
            last: None,
        })
    }
    /// Immutable ordered element identities and physical definitions.
    pub fn elements(&self) -> &[ReservoirElement] {
        &self.context.elements
    }
    /// Immutable numerical controls.
    pub fn config(&self) -> &ReservoirConfig {
        &self.context.config
    }
    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Complete lifetime represented water ledger.
    pub fn ledger(&self) -> &ReservoirLedger {
        &self.ledger
    }
    /// Number of accepted intervals since construction or restored origin.
    pub fn accepted_steps(&self) -> usize {
        self.steps
    }
    /// Project complete current depths at an application boundary.
    pub fn states(&self) -> Result<Vec<ReservoirState>, ReservoirError> {
        self.state.project(&self.context)
    }
    /// Lazily project the complete last accepted interval, when present.
    pub fn last_step(&self) -> Result<Option<ReservoirStepDiagnostics>, ReservoirError> {
        self.last
            .as_ref()
            .map(|step| project_step(&self.context, step))
            .transpose()
    }
    /// Integrate the exact represented interval `end_time-self.time()` without mutating accepted storage.
    pub fn trial_to(
        &self,
        end_time: f64,
        inputs: &[ReservoirInput],
    ) -> Result<ReservoirTrial<R>, ReservoirError> {
        let dt = end_time - self.time;
        if !end_time.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return Err(ReservoirError::input(format!(
                "time={}, end_time={end_time}, dt={dt}: require finite strictly increasing time",
                self.time
            )));
        }
        if inputs.len() != self.elements().len() {
            return Err(ReservoirError::input(format!(
                "inputs={}: require {} ordered elements",
                inputs.len(),
                self.elements().len()
            )));
        }
        for (index, input) in inputs.iter().enumerate() {
            if !input.incoming_rate_m_per_s.is_finite() || input.incoming_rate_m_per_s < 0.0 {
                return Err(ReservoirError::input(format!(
                    "incoming_rate_m_per_s={}: require finite nonnegative supply",
                    input.incoming_rate_m_per_s
                ))
                .at_element(index));
            }
        }
        let incoming = self.context.tensor(
            &inputs
                .iter()
                .map(|i| i.incoming_rate_m_per_s)
                .collect::<Vec<_>>(),
        )?;
        let step =
            crate::runoff::numerics::reservoir::advance(&self.context, &self.state, dt, &incoming)?;
        crate::runoff::numerics::reservoir::check_step(&self.context, &step, dt)?;
        let totals = aggregate_step(&self.context, &step)?;
        let ledger = self.ledger.advance(&self.context, &totals)?;
        let steps = self.steps.checked_add(1).ok_or_else(|| {
            ReservoirError::numerical(
                "accepted-step counter overflows. Start another component history",
            )
        })?;
        let state = TensorReservoirState {
            depth: step.depth.clone(),
        };
        Ok(ReservoirTrial {
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
                })),
            },
        })
    }
}

impl<R> ReservoirTrial<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Project candidate per-element physical output, without accepting it.
    pub fn outputs(&self) -> Result<Vec<ReservoirOutput>, ReservoirError> {
        Ok(self
            .diagnostics()?
            .receipts
            .into_iter()
            .map(|r| ReservoirOutput {
                element: r.element,
                depth_m: r.depth_m,
                storage_m3: r.storage_after_m3,
                incoming_volume_m3: r.incoming_volume_m3,
                export_volume_m3: r.export_volume_m3,
                endpoint_discharge_m3_per_s: r.endpoint_discharge_m3_per_s,
                water_residual_m3: r.water_residual_m3,
                constitutive_residual_m3: r.constitutive_residual_m3,
            })
            .collect())
    }
    /// Project complete candidate diagnostics at an explicit application boundary.
    pub fn diagnostics(&self) -> Result<ReservoirStepDiagnostics, ReservoirError> {
        let step = self.next.last.as_ref().ok_or_else(|| {
            ReservoirError::numerical("candidate lacks complete interval receipt")
        })?;
        project_step(&self.next.context, step)
    }
    /// Accept every tensor, time, and counter by ownership, without output projection.
    pub fn into_next(self) -> ReservoirBatch<R> {
        self.next
    }
}
