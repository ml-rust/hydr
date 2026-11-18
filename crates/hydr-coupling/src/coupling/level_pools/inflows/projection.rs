//! Explicit fallible external-ledger and interval-receipt projections.

use super::super::CoupledLevelPoolError;
use super::{
    BoundInflows, CoupledPoolInflow, CoupledPoolInflowLedger, CoupledPoolInflowReceipt,
    CoupledPoolInflowValue, InflowContext, InflowState, InflowStep,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> InflowContext<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn project_inputs(
        &self,
        client: &R::Client,
        inputs: &BoundInflows<R>,
    ) -> Result<Vec<CoupledPoolInflow>, CoupledLevelPoolError> {
        let checked = self.restore_inputs(client, inputs)?;
        let values = checked.values.try_to_vec::<f64>()?;
        checked
            .projection_order
            .iter()
            .map(|&index| {
                let binding = self.bindings.get(index).ok_or_else(|| {
                    CoupledLevelPoolError::input(
                        "forcing projection index exceeds immutable boundary count",
                    )
                })?;
                let scalar = *values.get(index).ok_or_else(|| {
                    CoupledLevelPoolError::input("forcing projection lacks original input operand")
                })?;
                let is_rate = *checked.input_is_rate.get(index).ok_or_else(|| {
                    CoupledLevelPoolError::input("forcing projection lacks original input kind")
                })?;
                Ok(CoupledPoolInflow {
                    boundary_id: binding.boundary_id.clone(),
                    input_id: binding.input_id.clone(),
                    value: if is_rate {
                        CoupledPoolInflowValue::Rate {
                            rate_m3_per_s: scalar,
                        }
                    } else {
                        CoupledPoolInflowValue::IntervalVolume { volume_m3: scalar }
                    },
                })
            })
            .collect()
    }

    pub(crate) fn project_state(
        &self,
        client: &R::Client,
        state: &InflowState<R>,
    ) -> Result<Vec<CoupledPoolInflowLedger>, CoupledLevelPoolError> {
        self.check_tensor(client, &state.volume)?;
        self.check_tensor(client, &state.bound)?;
        let volumes = state.volume.try_to_vec::<f64>()?;
        let bounds = state.bound.try_to_vec::<f64>()?;
        Ok(self
            .bindings
            .iter()
            .zip(volumes)
            .zip(bounds)
            .map(
                |((binding, cumulative_volume_m3), roundoff_bound_m3)| CoupledPoolInflowLedger {
                    boundary_id: binding.boundary_id.clone(),
                    input_id: binding.input_id.clone(),
                    cumulative_volume_m3,
                    roundoff_bound_m3,
                },
            )
            .collect())
    }

    pub(crate) fn project_step(
        &self,
        client: &R::Client,
        step: &InflowStep<R>,
    ) -> Result<Vec<CoupledPoolInflowReceipt>, CoupledLevelPoolError> {
        let count = self.bindings.len();
        for value in [
            &step.input_values,
            &step.volumes,
            &step.before.volume,
            &step.next.volume,
            &step.witness,
            &step.before.bound,
            &step.next.bound,
        ] {
            if value.shape() != [count]
                || value.dtype() != DType::F64
                || !value.device().is_same(client.device())
            {
                return Err(CoupledLevelPoolError::input(
                    "external receipt requires complete F64 tensors on the configured client device",
                ));
            }
        }
        if step.input_is_rate.len() != count {
            return Err(CoupledLevelPoolError::input(
                "external receipt kinds differ from declared boundary count",
            ));
        }
        let inputs = step
            .input_is_rate
            .iter()
            .zip(step.input_values.try_to_vec::<f64>()?)
            .map(|(is_rate, value)| {
                if *is_rate {
                    CoupledPoolInflowValue::Rate {
                        rate_m3_per_s: value,
                    }
                } else {
                    CoupledPoolInflowValue::IntervalVolume { volume_m3: value }
                }
            });
        let volumes = step.volumes.try_to_vec::<f64>()?;
        let before = step.before.volume.try_to_vec::<f64>()?;
        let after = step.next.volume.try_to_vec::<f64>()?;
        let witnesses = step.witness.try_to_vec::<f64>()?;
        let bound_before = step.before.bound.try_to_vec::<f64>()?;
        let bound_after = step.next.bound.try_to_vec::<f64>()?;
        Ok(self
            .bindings
            .iter()
            .zip(inputs)
            .zip(volumes)
            .zip(before)
            .zip(after)
            .zip(witnesses)
            .zip(bound_before)
            .zip(bound_after)
            .map(
                |(
                    (
                        (
                            (
                                (((binding, input), volume_m3), cumulative_before_m3),
                                cumulative_after_m3,
                            ),
                            addition_witness_m3,
                        ),
                        bound_before_m3,
                    ),
                    bound_after_m3,
                )| CoupledPoolInflowReceipt {
                    boundary_id: binding.boundary_id.clone(),
                    input_id: binding.input_id.clone(),
                    input,
                    dt_s: step.dt_s,
                    volume_m3,
                    cumulative_before_m3,
                    cumulative_after_m3,
                    addition_witness_m3,
                    bound_before_m3,
                    bound_after_m3,
                },
            )
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{binding, client, ledger, pools, rate, volume};
    use super::*;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn original_interval_volume_survives_cumulative_counter_absorption() -> TestResult {
        let client = client();
        let context =
            InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
        let original = context.import_state(&client, &[ledger(1.0, 0.0)], 1)?;
        let value = f64::EPSILON / 4.0;
        let rates = context.bind_inputs(&client, &[rate("boundary", value)])?;
        let prepared = context.prepare(&client, &rates, &original, 1.0, 2)?;
        assert_eq!(prepared.summary.volume_m3, value);
        assert!(prepared.summary.counter_roundoff_bound_m3 >= value);
        let receipt = context
            .project_step(&client, &prepared.step)?
            .pop()
            .ok_or("missing external receipt")?;
        assert_eq!(receipt.volume_m3, value);
        assert_eq!(receipt.addition_witness_m3, value);
        assert_eq!(receipt.cumulative_before_m3, 1.0);
        assert_eq!(receipt.cumulative_after_m3, 1.0);
        assert!(receipt.bound_after_m3 >= value);
        let incoming = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0], &[2], client.device())?;
        assert!(
            context
                .merge_incoming(&client, &incoming, &prepared)
                .is_err()
        );
        assert_eq!(
            context.project_state(&client, &original)?,
            vec![ledger(1.0, 0.0)]
        );
        Ok(())
    }

    #[test]
    fn authoritative_subnormal_volume_never_passes_through_a_rate() -> TestResult {
        let client = client();
        let context =
            InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
        let original = context.zero_state(&client)?;
        let quantum = f64::from_bits(1);
        assert_eq!(quantum / 2.0, 0.0);
        let inputs = context.bind_inputs(&client, &[volume("boundary", quantum)])?;
        assert!(inputs.has_interval_volumes());
        let prepared = context.prepare(&client, &inputs, &original, 2.0, 1)?;
        assert_eq!(
            prepared
                .incoming
                .try_to_vec::<f64>()?
                .first()
                .map(|value| value.to_bits()),
            Some(quantum.to_bits())
        );
        let receipt = context
            .project_step(&client, &prepared.step)?
            .pop()
            .ok_or("missing interval-volume receipt")?;
        assert!(
            receipt
                .input
                .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: quantum })
        );
        assert_eq!(receipt.volume_m3.to_bits(), quantum.to_bits());
        assert_eq!(receipt.cumulative_after_m3.to_bits(), quantum.to_bits());
        Ok(())
    }

    #[test]
    fn mixed_inputs_preserve_original_variants_and_mask_underflow_checks() -> TestResult {
        let client = client();
        let mut receiving_rate = binding("rate");
        receiving_rate.pool_id = "pool-1".into();
        let context = InflowContext::<CpuRuntime>::new(
            &client,
            &pools(4.0)?,
            &[binding("volume"), receiving_rate],
        )?;
        let original = context.zero_state(&client)?;
        let quantum = f64::from_bits(1);
        let inputs =
            context.bind_inputs(&client, &[rate("rate", 0.25), volume("volume", quantum)])?;
        let prepared = context.prepare(&client, &inputs, &original, 2.0, 1)?;
        assert_eq!(prepared.incoming.try_to_vec::<f64>()?, vec![quantum, 0.5]);
        let receipts = context.project_step(&client, &prepared.step)?;
        let first = receipts.first().ok_or("missing integrated receipt")?;
        assert!(
            first
                .input
                .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: quantum })
        );
        let mut altered = first.clone();
        altered.input = CoupledPoolInflowValue::Rate {
            rate_m3_per_s: quantum,
        };
        assert!(!first.same_bits(&altered));
        let last = receipts.last().ok_or("missing rate receipt")?;
        assert!(last.input.same_bits(&CoupledPoolInflowValue::Rate {
            rate_m3_per_s: 0.25
        }));
        let underflow =
            context.bind_inputs(&client, &[volume("volume", quantum), rate("rate", quantum)])?;
        assert!(
            context
                .prepare(&client, &underflow, &original, 0.25, 1)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn zero_integrated_input_keeps_its_kind_and_original_operand_bits() -> TestResult {
        let client = client();
        let context =
            InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
        let original = context.zero_state(&client)?;
        let inputs = context.bind_inputs(&client, &[volume("boundary", -0.0)])?;
        assert!(inputs.has_interval_volumes());
        let prepared = context.prepare(&client, &inputs, &original, 1.0, 1)?;
        let receipt = context
            .project_step(&client, &prepared.step)?
            .pop()
            .ok_or("missing zero-volume receipt")?;
        assert!(
            receipt
                .input
                .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: -0.0 })
        );
        assert!(
            !receipt
                .input
                .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.0 })
        );
        assert!(!receipt.input.same_bits(&CoupledPoolInflowValue::Rate {
            rate_m3_per_s: -0.0
        }));
        assert_eq!(receipt.volume_m3.to_bits(), (-0.0f64).to_bits());
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                context
                    .bind_inputs(&client, &[volume("boundary", invalid)])
                    .is_err()
            );
        }
        assert_eq!(
            context.project_state(&client, &original)?,
            vec![ledger(0.0, 0.0)]
        );
        Ok(())
    }
}
