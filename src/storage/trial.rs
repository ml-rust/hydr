//! Shared completion and accounting for host and device pool inputs.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use std::sync::Arc;

use super::batch::StoredStep;
use super::{LevelPoolBatch, LevelPoolError, LevelPoolSources, LevelPoolTrial};

pub(super) fn duration(time: f64, end: f64) -> Result<f64, LevelPoolError> {
    let dt = end - time;
    if !end.is_finite() || !dt.is_finite() || dt <= 0.0 {
        return Err(LevelPoolError::input(format!(
            "time={time},end={end},dt={dt}: require finite strictly increasing time"
        )));
    }
    Ok(dt)
}

pub(super) fn next_step_count(steps: usize) -> Result<usize, LevelPoolError> {
    steps.checked_add(1).ok_or_else(|| {
        LevelPoolError::numerical(
            "accepted interval count overflows. Start another component history",
        )
    })
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
    pub(super) fn trial_prepared(
        &self,
        end: f64,
        dt: f64,
        steps: usize,
        incoming: &Tensor<R>,
        outgoing: &Tensor<R>,
        sources: Option<&[LevelPoolSources]>,
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        let (state, step) = if let Some(sources) = sources {
            let input =
                crate::storage::tensor::TensorPoolSourceInput::<R>::new(&self.context, sources)?;
            crate::storage::numerics::advance_with_sources(
                &self.context,
                &self.state,
                incoming,
                outgoing,
                steps,
                dt,
                &input,
            )?
        } else {
            crate::storage::numerics::advance(
                &self.context,
                &self.state,
                incoming,
                outgoing,
                steps,
            )?
        };
        self.complete_trial(end, steps, state, step)
    }

    pub(super) fn complete_trial(
        &self,
        end: f64,
        steps: usize,
        state: crate::storage::tensor::TensorPoolState<R>,
        step: crate::storage::tensor::TensorPoolStep<R>,
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        crate::storage::numerics::check_step(&self.context, &step)?;
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
