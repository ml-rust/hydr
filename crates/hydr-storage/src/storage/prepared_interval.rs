//! Cached native source preparation for coordinated exchange trials.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use super::{LevelPoolBatch, LevelPoolError, LevelPoolSources, LevelPoolTrial};
use crate::storage::numerics::sources::PreparedSources;
use crate::storage::tensor::TensorPoolSourceInput;

#[doc(hidden)]
pub struct LevelPoolSourcePlan<'a, R: Runtime<DType = DType>> {
    accepted: &'a LevelPoolBatch<R>,
    input: TensorPoolSourceInput<R>,
}

#[doc(hidden)]
pub struct PreparedLevelPoolInterval<'a, R: Runtime<DType = DType>> {
    accepted: &'a LevelPoolBatch<R>,
    end: f64,
    steps: usize,
    sources: Option<PreparedSources<R>>,
}

impl<R: Runtime<DType = DType>> LevelPoolBatch<R>
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
    #[doc(hidden)]
    pub fn bind_source_rates(
        &self,
        sources: &[LevelPoolSources],
    ) -> Result<LevelPoolSourcePlan<'_, R>, LevelPoolError> {
        Ok(LevelPoolSourcePlan {
            accepted: self,
            input: TensorPoolSourceInput::new(&self.context, sources)?,
        })
    }
}

impl<'a, R: Runtime<DType = DType>> LevelPoolSourcePlan<'a, R>
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
    #[doc(hidden)]
    pub fn prepare(&self, end: f64) -> Result<PreparedLevelPoolInterval<'a, R>, LevelPoolError> {
        let dt = super::trial::duration(self.accepted.time, end)?;
        let steps = super::trial::next_step_count(self.accepted.steps)?;
        let sources = crate::storage::numerics::sources::prepare(
            &self.accepted.context,
            &self.accepted.state,
            dt,
            &self.input,
        )?;
        Ok(PreparedLevelPoolInterval {
            accepted: self.accepted,
            end,
            steps,
            sources,
        })
    }
}

impl<R: Runtime<DType = DType>> PreparedLevelPoolInterval<'_, R> {
    #[doc(hidden)]
    pub fn available_volume(&self) -> &Tensor<R> {
        self.sources
            .as_ref()
            .map_or(&self.accepted.state.volume_m3, |(step, _)| {
                &step.post_source_volume_m3
            })
    }
}

impl<R: Runtime<DType = DType>> PreparedLevelPoolInterval<'_, R>
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
    #[doc(hidden)]
    pub fn trial_to_tensors(
        &self,
        incoming: &Tensor<R>,
        outgoing: &Tensor<R>,
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        self.accepted.check_exchange_tensors(incoming, outgoing)?;
        let (state, step) = crate::storage::numerics::advance_prepared(
            &self.accepted.context,
            &self.accepted.state,
            incoming,
            outgoing,
            self.steps,
            self.sources.as_ref(),
        )?;
        self.accepted
            .complete_trial(self.end, self.steps, state, step)
    }

    #[doc(hidden)]
    pub fn source_totals(
        &self,
    ) -> Result<super::source_summary::LevelPoolIntervalSources, LevelPoolError> {
        super::source_summary::interval(
            &self.accepted.context,
            self.sources.as_ref().map(|(s, _)| s),
        )
    }
}
