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
        if let Some(last) = &self.last
            && last.context_generation == self.context_generation
        {
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
mod tests {
    use super::*;
    use crate::storage::fixture::batch;
    use crate::storage::{LevelPoolInput, LevelPoolSources, LevelPoolState};
    use numr::runtime::cpu::CpuRuntime;
    use std::sync::Arc;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn tensor_trial_matches_host_trial_and_preserves_configured_context() -> TestResult {
        let batch = batch(false)?;
        let before = batch.snapshot_data()?;
        let context = Arc::clone(&batch.context);
        let incoming = batch.context.tensor(&[0.5, 1.0])?;
        let outgoing = batch.context.tensor(&[0.25, 0.0])?;
        let host = batch
            .trial_to(
                1.0,
                &[
                    LevelPoolInput {
                        incoming_volume_m3: 0.5,
                        outgoing_volume_m3: 0.25,
                    },
                    LevelPoolInput {
                        incoming_volume_m3: 1.0,
                        outgoing_volume_m3: 0.0,
                    },
                ],
            )?
            .into_next();
        let device = batch
            .trial_to_tensors(1.0, &incoming, &outgoing)?
            .into_next();
        assert_eq!(device.snapshot_data()?, host.snapshot_data()?);
        assert_eq!(batch.snapshot_data()?, before);
        assert!(Arc::ptr_eq(&device.context, &context));
        assert_eq!(device.context.client.parallelism().chunk_size, Some(64));
        assert_eq!(device.accepted_steps(), 1);
        assert_eq!(
            device.stages_tensor()?.try_to_vec::<f64>()?,
            vec![2.25, -1.0]
        );
        Ok(())
    }

    #[test]
    fn tensor_access_keeps_dry_origin_distinct_from_physical_stage() -> TestResult {
        let batch = batch(false)?;
        assert_eq!(batch.volumes_tensor().try_to_vec::<f64>()?, vec![2.0, 0.0]);
        assert_eq!(batch.stages_tensor()?.try_to_vec::<f64>()?, vec![2.0, -2.0]);
        assert_eq!(batch.outputs()?[0].stage_m, Some(2.0));
        assert_eq!(batch.outputs()?[1].stage_m, None);
        Ok(())
    }

    #[test]
    fn tensor_metadata_errors_preserve_complete_accepted_state() -> TestResult {
        let batch = batch(false)?;
        let before = batch.snapshot_data()?;
        let zero = batch.context.tensor(&[0.0, 0.0])?;
        let device = batch.context.client.device();
        let invalid = [
            Tensor::<CpuRuntime>::zeros(&[], DType::F64, device)?,
            Tensor::<CpuRuntime>::zeros(&[1], DType::F64, device)?,
            Tensor::<CpuRuntime>::zeros(&[1, 2], DType::F64, device)?,
            Tensor::<CpuRuntime>::zeros(&[3], DType::F64, device)?,
            Tensor::<CpuRuntime>::zeros(&[2], DType::F32, device)?,
        ];
        for value in invalid {
            assert!(matches!(
                batch.trial_to_tensors(1.0, &value, &zero),
                Err(LevelPoolError::InvalidInput { .. })
            ));
            assert!(matches!(
                batch.trial_to_tensors(1.0, &zero, &value),
                Err(LevelPoolError::InvalidInput { .. })
            ));
            assert_eq!(batch.snapshot_data()?, before);
        }
        Ok(())
    }

    #[test]
    fn tensor_physical_errors_preserve_complete_accepted_state() -> TestResult {
        let batch = batch(false)?;
        let before = batch.snapshot_data()?;
        let zero = batch.context.tensor(&[0.0, 0.0])?;
        for values in [[-0.1, 0.0], [f64::NAN, 0.0], [f64::INFINITY, 0.0]] {
            let input = batch.context.tensor(&values)?;
            assert!(batch.trial_to_tensors(1.0, &input, &zero).is_err());
            assert!(batch.trial_to_tensors(1.0, &zero, &input).is_err());
            assert_eq!(batch.snapshot_data()?, before);
        }
        let unavailable = batch.context.tensor(&[3.0, 0.0])?;
        assert!(batch.trial_to_tensors(1.0, &zero, &unavailable).is_err());
        for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                batch.trial_to_tensors(end, &zero, &zero),
                Err(LevelPoolError::InvalidInput { .. })
            ));
        }
        assert_eq!(batch.snapshot_data()?, before);
        Ok(())
    }

    #[test]
    fn closed_tensor_trial_rejects_source_enabled_state_without_changing_public_source_trials()
    -> TestResult {
        let batch = batch(true)?;
        let before = batch.snapshot_data()?;
        let zero = batch.context.tensor(&[0.0, 0.0])?;
        assert!(matches!(
            batch.trial_to_tensors(1.0, &zero, &zero),
            Err(LevelPoolError::InvalidInput { .. })
        ));
        assert_eq!(batch.snapshot_data()?, before);
        let inputs = [LevelPoolInput {
            incoming_volume_m3: 0.0,
            outgoing_volume_m3: 0.0,
        }; 2];
        let exchange_only = batch.trial_to(1.0, &inputs)?.into_next();
        assert_eq!(exchange_only.states()?, batch.states()?);
        let source_trial = batch
            .trial_with_sources_to(
                1.0,
                &inputs,
                &[LevelPoolSources {
                    rain_rate_m_per_s: 0.125,
                    potential_evaporation_rate_m_per_s: 0.0,
                    bottom_loss_rate_m3_per_s: 0.0,
                }; 2],
            )?
            .into_next();
        assert_eq!(
            source_trial.states()?,
            vec![
                LevelPoolState { volume_m3: 2.125 },
                LevelPoolState { volume_m3: 0.125 }
            ]
        );
        assert_eq!(batch.snapshot_data()?, before);
        Ok(())
    }
}
