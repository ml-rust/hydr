//! Device-resident access for coordinated pool exchanges.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use super::{LevelPoolBatch, LevelPoolError, LevelPoolTrial};

impl<R: Runtime<DType = DType>> LevelPoolBatch<R> {
    pub(crate) fn volumes_tensor(&self) -> &Tensor<R> {
        &self.state.volume_m3
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
    /// Zero storage retains the numerical curve origin, not a physical water stage.
    pub(crate) fn stages_tensor(&self) -> Result<Tensor<R>, LevelPoolError> {
        if let Some(last) = &self.last {
            return Ok(last.step.stage_m.clone());
        }
        Ok(crate::numerics::storage::curves::evaluate(&self.context, &self.state.volume_m3)?.stage)
    }

    pub(crate) fn trial_to_tensors(
        &self,
        end: f64,
        incoming: &Tensor<R>,
        outgoing: &Tensor<R>,
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        let dt = super::trial::duration(self.time, end)?;
        if self.source_policies().is_some() {
            return Err(LevelPoolError::input(
                "tensor exchange trial has source-enabled pools. Supply exchange-only pools for closed coordination",
            ));
        }
        self.check_exchange_tensors(incoming, outgoing)?;
        let steps = super::trial::next_step_count(self.steps)?;
        self.trial_prepared(end, dt, steps, incoming, outgoing, None)
    }

    pub(super) fn check_exchange_tensors(
        &self,
        incoming: &Tensor<R>,
        outgoing: &Tensor<R>,
    ) -> Result<(), LevelPoolError> {
        for (quantity, value) in [("incoming volume", incoming), ("outgoing volume", outgoing)] {
            if value.shape() != [self.elements().len()]
                || value.dtype() != DType::F64
                || !value.device().is_same(self.context.client.device())
            {
                return Err(LevelPoolError::input(format!(
                    "{quantity}: shape={:?},dtype={:?}. Supply F64 shape [{}] on the configured client device",
                    value.shape(),
                    value.dtype(),
                    self.elements().len()
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "coupling_tests.rs"]
mod tests;
