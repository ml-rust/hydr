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

/// Borrowed native counters retain each pool's original arithmetic operands.
#[doc(hidden)]
pub struct LevelPoolNativeAccounting<'a, R: Runtime<DType = DType>> {
    pub volume_m3: &'a Tensor<R>,
    pub cumulative_incoming_volume_m3: &'a Tensor<R>,
    pub cumulative_outgoing_volume_m3: &'a Tensor<R>,
    pub incoming_roundoff_bound_m3: &'a Tensor<R>,
    pub outgoing_roundoff_bound_m3: &'a Tensor<R>,
}

#[doc(hidden)]
pub struct LevelPoolNativeInterval<'a, R: Runtime<DType = DType>> {
    pub old_volume_m3: &'a Tensor<R>,
    pub volume_m3: &'a Tensor<R>,
    pub incoming_volume_m3: &'a Tensor<R>,
    pub outgoing_volume_m3: &'a Tensor<R>,
    pub outgoing_witness_m3: &'a Tensor<R>,
    pub incoming_witness_m3: &'a Tensor<R>,
    pub witness_consistency_bound_m3: &'a Tensor<R>,
}

impl<R: Runtime<DType = DType>> LevelPoolBatch<R> {
    #[doc(hidden)]
    pub fn volumes_tensor(&self) -> &Tensor<R> {
        &self.state.volume_m3
    }

    #[doc(hidden)]
    pub fn coupling_native_accounting(&self) -> LevelPoolNativeAccounting<'_, R> {
        let state = &self.state;
        LevelPoolNativeAccounting {
            volume_m3: &state.volume_m3,
            cumulative_incoming_volume_m3: &state.cumulative_incoming_volume_m3,
            cumulative_outgoing_volume_m3: &state.cumulative_outgoing_volume_m3,
            incoming_roundoff_bound_m3: &state.incoming_roundoff_bound_m3,
            outgoing_roundoff_bound_m3: &state.outgoing_roundoff_bound_m3,
        }
    }

    #[doc(hidden)]
    pub fn coupling_native_interval(&self) -> Option<LevelPoolNativeInterval<'_, R>> {
        self.last.as_ref().map(|last| {
            let step = &last.step;
            LevelPoolNativeInterval {
                old_volume_m3: &step.old_volume_m3,
                volume_m3: &step.volume_m3,
                incoming_volume_m3: &step.incoming_volume_m3,
                outgoing_volume_m3: &step.outgoing_volume_m3,
                outgoing_witness_m3: &step.outgoing_witness_m3,
                incoming_witness_m3: &step.incoming_witness_m3,
                witness_consistency_bound_m3: &step.witness_consistency_bound_m3,
            }
        })
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
    #[doc(hidden)]
    pub fn stages_tensor(&self) -> Result<Tensor<R>, LevelPoolError> {
        if let Some(last) = &self.last {
            return Ok(last.step.stage_m.clone());
        }
        Ok(crate::storage::numerics::curves::evaluate(&self.context, &self.state.volume_m3)?.stage)
    }

    #[doc(hidden)]
    pub fn trial_to_tensors(
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
