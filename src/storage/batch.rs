//! Immutable authoritative-volume trials with lazy output projection.

use super::{
    LevelPoolConfig, LevelPoolElement, LevelPoolElementLedger, LevelPoolError, LevelPoolInput,
    LevelPoolOutput, LevelPoolState, LevelPoolStepDiagnostics, LevelPoolTotals,
};
use crate::cpu::level_pool::{CpuPoolContext, CpuPoolState, CpuPoolStep};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct StoredStep {
    pub start: f64,
    pub end: f64,
    pub step: CpuPoolStep,
}

/// Complete persistent closed-storage continuation without atmospheric or outlet laws.
#[derive(Clone, Debug)]
pub struct LevelPoolBatch {
    pub(super) context: Arc<CpuPoolContext>,
    pub(super) state: CpuPoolState,
    pub(super) time: f64,
    pub(super) steps: usize,
    pub(super) last: Option<Arc<StoredStep>>,
}

/// Complete candidate state. Dropping it preserves accepted volume and accounting.
#[derive(Debug)]
pub struct LevelPoolTrial {
    pub(super) next: LevelPoolBatch,
}

impl LevelPoolBatch {
    /// Construct explicit pool definitions, accepted volumes, elapsed time, and dimensional controls.
    pub fn new(
        elements: Vec<LevelPoolElement>,
        states: Vec<LevelPoolState>,
        time: f64,
        config: LevelPoolConfig,
    ) -> Result<Self, LevelPoolError> {
        if !time.is_finite() || time < 0.0 {
            return Err(LevelPoolError::input(format!(
                "time={time}: supply finite nonnegative accepted time"
            )));
        }
        let context = Arc::new(CpuPoolContext::new(elements, config)?);
        let state = CpuPoolState::new(&context, &states)?;
        crate::numerics::storage::curves::evaluate(&context, &state.volume_m3)?;
        crate::numerics::storage::accounting::check_state(&context, &state, 0)?;
        super::accounting::totals(&context, &state)?;
        Ok(Self {
            context,
            state,
            time,
            steps: 0,
            last: None,
        })
    }
    /// Complete immutable physical definitions in numerical order.
    pub fn elements(&self) -> &[LevelPoolElement] {
        &self.context.elements
    }
    /// Independent dimensional controls.
    pub fn config(&self) -> &LevelPoolConfig {
        &self.context.config
    }
    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Number of accepted exchange intervals.
    pub fn accepted_steps(&self) -> usize {
        self.steps
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
    ) -> Result<LevelPoolTrial, LevelPoolError> {
        let dt = end - self.time;
        if !end.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return Err(LevelPoolError::input(format!(
                "time={},end={end},dt={dt}: require finite strictly increasing time",
                self.time
            )));
        }
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
        let steps = self.steps.checked_add(1).ok_or_else(|| {
            LevelPoolError::numerical(
                "accepted interval count overflows. Start another component history",
            )
        })?;
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
        let (state, step) = crate::numerics::storage::advance(
            &self.context,
            &self.state,
            &incoming,
            &outgoing,
            steps,
        )?;
        crate::numerics::storage::check_step(&self.context, &step)?;
        super::accounting::step_totals(&self.context, &step)?;
        super::accounting::totals(&self.context, &state)?;
        Ok(LevelPoolTrial {
            next: Self {
                context: Arc::clone(&self.context),
                state,
                time: end,
                steps,
                last: Some(Arc::new(StoredStep {
                    start: self.time,
                    end,
                    step,
                })),
            },
        })
    }
}
impl LevelPoolTrial {
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
    pub fn into_next(self) -> LevelPoolBatch {
        self.next
    }
}
