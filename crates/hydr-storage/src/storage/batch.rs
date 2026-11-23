//! Immutable authoritative-volume trials with lazy output projection.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

use super::{
    LevelPoolConfig, LevelPoolElement, LevelPoolElementLedger, LevelPoolError, LevelPoolInput,
    LevelPoolOutput, LevelPoolState, LevelPoolStepDiagnostics, LevelPoolTotals,
};
use crate::storage::tensor::{TensorPoolContext, TensorPoolState, TensorPoolStep};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub start: f64,
    pub end: f64,
    pub step: TensorPoolStep<R>,
    pub context_generation: u64,
}

/// Complete persistent pool continuation with optional explicit sources and no outlet law.
#[derive(Clone, Debug)]
pub struct LevelPoolBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<TensorPoolContext<R>>,
    pub(super) state: TensorPoolState<R>,
    pub(super) time: f64,
    pub(super) steps: usize,
    pub(super) last: Option<Arc<StoredStep<R>>>,
    pub(super) context_generation: u64,
}

/// Complete candidate state. Dropping it preserves accepted volume and accounting.
#[derive(Debug)]
pub struct LevelPoolTrial<R: Runtime<DType = DType>> {
    pub(super) next: LevelPoolBatch<R>,
}

impl<R: Runtime<DType = DType>> LevelPoolBatch<R> {
    /// Complete immutable physical definitions in numerical order.
    pub fn elements(&self) -> &[LevelPoolElement] {
        &self.context.elements
    }
    /// Independent dimensional controls.
    pub fn config(&self) -> &LevelPoolConfig {
        &self.context.config
    }
    /// Immutable source-enabling evidence, absent for exchange-only batches.
    pub fn source_policies(&self) -> Option<&[super::LevelPoolSourcePolicy]> {
        self.context.source_policies.as_deref()
    }
    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Number of accepted exchange intervals.
    pub fn accepted_steps(&self) -> usize {
        self.steps
    }
}

impl<R> LevelPoolBatch<R>
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
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Construct explicit pool definitions, accepted volumes, elapsed time, and dimensional controls.
    pub fn new(
        client: R::Client,
        elements: Vec<LevelPoolElement>,
        states: Vec<LevelPoolState>,
        time: f64,
        config: LevelPoolConfig,
    ) -> Result<Self, LevelPoolError> {
        Self::construct(client, elements, states, time, config, None)
    }
    /// Enable the frozen-area source policy with explicit evidence for every ordered pool.
    pub fn new_with_sources(
        client: R::Client,
        elements: Vec<LevelPoolElement>,
        states: Vec<LevelPoolState>,
        time: f64,
        config: LevelPoolConfig,
        source_policies: Vec<super::LevelPoolSourcePolicy>,
    ) -> Result<Self, LevelPoolError> {
        Self::construct(
            client,
            elements,
            states,
            time,
            config,
            Some(source_policies),
        )
    }
    fn construct(
        client: R::Client,
        elements: Vec<LevelPoolElement>,
        states: Vec<LevelPoolState>,
        time: f64,
        config: LevelPoolConfig,
        source_policies: Option<Vec<super::LevelPoolSourcePolicy>>,
    ) -> Result<Self, LevelPoolError> {
        if !time.is_finite() || time < 0.0 {
            return Err(LevelPoolError::input(format!(
                "time={time}: supply finite nonnegative accepted time"
            )));
        }
        let context = Arc::new(match source_policies {
            None => TensorPoolContext::<R>::new(client, elements, config)?,
            Some(policies) => {
                TensorPoolContext::<R>::new_with_sources(client, elements, config, Some(policies))?
            }
        });
        let state = TensorPoolState::<R>::new(&context, &states)?;
        crate::storage::numerics::curves::evaluate(&context, &state.volume_m3)?;
        crate::storage::numerics::accounting::check_state(&context, &state, 0)?;
        super::accounting::totals(&context, &state)?;
        Ok(Self {
            context,
            state,
            time,
            steps: 0,
            last: None,
            context_generation: 0,
        })
    }
    /// Explicitly project authoritative accepted element volumes.
    pub fn states(&self) -> Result<Vec<LevelPoolState>, LevelPoolError> {
        Ok(
            super::accounting::read(&self.state.volume_m3, self.elements().len())?
                .into_iter()
                .map(|volume_m3| LevelPoolState { volume_m3 })
                .collect(),
        )
    }
    /// Explicitly project checked physical stage, with None for dry storage.
    pub fn outputs(&self) -> Result<Vec<LevelPoolOutput>, LevelPoolError> {
        super::projection::outputs(&self.context, &self.state)
    }
    /// Explicitly project complete per-element cumulative accounting.
    pub fn element_ledgers(&self) -> Result<Vec<LevelPoolElementLedger>, LevelPoolError> {
        super::projection::ledgers(&self.context, &self.state)
    }
    /// Reduce persistent counters and independently check aggregate physical accounting.
    pub fn totals(&self) -> Result<LevelPoolTotals, LevelPoolError> {
        super::accounting::totals(&self.context, &self.state)
    }
    /// Explicitly project the complete last accepted interval.
    pub fn last_step(&self) -> Result<Option<LevelPoolStepDiagnostics>, LevelPoolError> {
        self.last
            .as_ref()
            .map(|s| super::projection::step(&self.context, s))
            .transpose()
    }
    /// Trial the exact original interval without incoming credit for the donor budget.
    pub fn trial_to(
        &self,
        end: f64,
        inputs: &[LevelPoolInput],
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        self.trial_inputs(end, inputs, None)
    }
    /// Apply frozen-area rain, capped bottom demand, and capped evaporation before original exchanges.
    pub fn trial_with_sources_to(
        &self,
        end: f64,
        inputs: &[LevelPoolInput],
        sources: &[super::LevelPoolSources],
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        self.trial_inputs(end, inputs, Some(sources))
    }
    fn trial_inputs(
        &self,
        end: f64,
        inputs: &[LevelPoolInput],
        sources: Option<&[super::LevelPoolSources]>,
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        let dt = super::trial::duration(self.time, end)?;
        if inputs.len() != self.elements().len() {
            return Err(LevelPoolError::input(format!(
                "input count={},element count={}: supply one original exchange per element",
                inputs.len(),
                self.elements().len()
            )));
        }
        for (index, input) in inputs.iter().enumerate() {
            if !input.incoming_volume_m3.is_finite()
                || input.incoming_volume_m3 < 0.0
                || !input.outgoing_volume_m3.is_finite()
                || input.outgoing_volume_m3 < 0.0
            {
                return Err(LevelPoolError::input(format!(
                    "exchange={input:?}: supply finite nonnegative original volumes"
                ))
                .at(index));
            }
        }
        let steps = super::trial::next_step_count(self.steps)?;
        let incoming = self.context.tensor(
            &inputs
                .iter()
                .map(|i| i.incoming_volume_m3)
                .collect::<Vec<_>>(),
        )?;
        let outgoing = self.context.tensor(
            &inputs
                .iter()
                .map(|i| i.outgoing_volume_m3)
                .collect::<Vec<_>>(),
        )?;
        self.trial_prepared(end, dt, steps, &incoming, &outgoing, sources)
    }
}
impl<R> LevelPoolTrial<R>
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
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Explicitly project candidate physical storage and wet stage.
    pub fn outputs(&self) -> Result<Vec<LevelPoolOutput>, LevelPoolError> {
        self.next.outputs()
    }
    /// Explicitly project all original operands and independent arithmetic/physical gates.
    pub fn diagnostics(&self) -> Result<LevelPoolStepDiagnostics, LevelPoolError> {
        self.next
            .last_step()?
            .ok_or_else(|| LevelPoolError::numerical("candidate lacks complete interval receipts"))
    }
    /// Accept complete persistent state by ownership without numerical projection.
    pub fn into_next(self) -> LevelPoolBatch<R> {
        self.next
    }
}
