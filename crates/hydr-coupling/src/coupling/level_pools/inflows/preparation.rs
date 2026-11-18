//! Original inflow volumes, immutable ledger additions, and shared incoming merges.

use super::super::{
    CoupledLevelPoolError,
    topology::{finite, reject, upward_add},
};
use super::accounting::{charge, totals};
use super::{BoundInflows, InflowContext, InflowIntervalSummary, InflowState};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct InflowStep<R: Runtime<DType = DType>> {
    pub(super) input_values: Tensor<R>,
    pub(super) input_is_rate: Arc<Vec<bool>>,
    pub(super) volumes: Tensor<R>,
    pub(super) before: InflowState<R>,
    pub(super) next: InflowState<R>,
    pub(super) witness: Tensor<R>,
    pub(super) dt_s: f64,
}

pub(crate) struct PreparedInflows<R: Runtime<DType = DType>> {
    pub(crate) incoming: Tensor<R>,
    pub(crate) next: InflowState<R>,
    pub(crate) step: InflowStep<R>,
    pub(crate) summary: InflowIntervalSummary,
}

pub(crate) struct MergedIncoming<R: Runtime<DType = DType>> {
    pub(crate) volume: Tensor<R>,
    pub(crate) roundoff_bound_m3: f64,
}

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
    pub(crate) fn prepare(
        &self,
        client: &R::Client,
        inputs: &BoundInflows<R>,
        state: &InflowState<R>,
        dt_s: f64,
        accepted_steps_after: usize,
    ) -> Result<PreparedInflows<R>, CoupledLevelPoolError> {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(CoupledLevelPoolError::input(
                "external inflow duration must be finite and strictly positive",
            ));
        }
        if !Arc::ptr_eq(&self.bindings, &inputs.bindings) {
            return Err(CoupledLevelPoolError::input(
                "external inputs belong to another immutable boundary context",
            ));
        }
        let steps_before = accepted_steps_after.checked_sub(1).ok_or_else(|| {
            CoupledLevelPoolError::input(
                "external preparation requires a positive next accepted-step count",
            )
        })?;
        self.check_state(client, state, steps_before)?;
        self.check_tensor(client, &inputs.values)?;
        let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
        let rate_values = if inputs.has_interval_volumes() {
            client.where_cond(&inputs.rate_mask, &inputs.values, &zero)?
        } else {
            inputs.values.clone()
        };
        let rate_volumes = client.mul_scalar(&rate_values, dt_s)?;
        let volumes = if inputs.has_interval_volumes() {
            client.where_cond(&inputs.rate_mask, &rate_volumes, &inputs.values)?
        } else {
            rate_volumes
        };
        if !self.bindings.is_empty() {
            finite(client, &volumes, "original external interval volume")?;
            reject(
                client,
                &client.mul(
                    &client.gt(&rate_values, &zero)?,
                    &client.eq(&volumes, &zero)?,
                )?,
                "positive external rate-times-duration volume underflows. Rescale the inflow or interval",
            )?;
        }
        self.prepare_volumes(client, inputs, state, dt_s, accepted_steps_after, volumes)
    }

    // Original represented volumes enter assembly directly. This path performs no V/dt conversion.
    fn prepare_volumes(
        &self,
        client: &R::Client,
        inputs: &BoundInflows<R>,
        state: &InflowState<R>,
        dt_s: f64,
        accepted_steps_after: usize,
        volumes: Tensor<R>,
    ) -> Result<PreparedInflows<R>, CoupledLevelPoolError> {
        self.check_tensor(client, &volumes)?;
        let aggregated = self.tree.aggregate(client, &volumes)?;
        if self.bindings.is_empty() {
            return Ok(PreparedInflows {
                incoming: aggregated.volume,
                next: state.clone(),
                step: InflowStep {
                    input_values: inputs.values.clone(),
                    input_is_rate: Arc::clone(&inputs.input_is_rate),
                    volumes,
                    before: state.clone(),
                    next: state.clone(),
                    witness: Tensor::zeros(&[0], DType::F64, client.device())?,
                    dt_s,
                },
                summary: InflowIntervalSummary::default(),
            });
        }
        let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
        reject(
            client,
            &client.gt(&volumes, &self.capacity)?,
            "external interval volume exceeds receiving-pool capacity. Shorten the interval",
        )?;
        let (sum, low) = client.two_sum(&state.volume, &volumes)?;
        let inactive = client.eq(&volumes, &zero)?;
        let cumulative = client.where_cond(&inactive, &state.volume, &sum)?;
        let witness = client.where_cond(&inactive, &zero, &low)?;
        let bound = upward_add(client, &state.bound, &client.abs(&witness)?)?;
        let next = InflowState {
            volume: cumulative,
            bound,
        };
        self.check_state(client, &next, accepted_steps_after)?;
        let (volume_m3, reduction_bound_m3) = totals(client, self.boundary_reduction, &volumes)?;
        let summary = InflowIntervalSummary {
            volume_m3,
            reduction_bound_m3,
            assembly_bound_m3: charge(client, self.pool_reduction, &aggregated.bound)?,
            counter_roundoff_bound_m3: charge(
                client,
                self.boundary_reduction,
                &client.abs(&witness)?,
            )?,
        };
        Ok(PreparedInflows {
            incoming: aggregated.volume,
            next: next.clone(),
            step: InflowStep {
                input_values: inputs.values.clone(),
                input_is_rate: Arc::clone(&inputs.input_is_rate),
                volumes,
                before: state.clone(),
                next,
                witness,
                dt_s,
            },
            summary,
        })
    }

    pub(crate) fn merge_incoming(
        &self,
        client: &R::Client,
        internal: &Tensor<R>,
        external: &PreparedInflows<R>,
    ) -> Result<MergedIncoming<R>, CoupledLevelPoolError> {
        if internal.shape() != [self.pool_count]
            || internal.dtype() != DType::F64
            || !internal.device().is_same(client.device())
        {
            return Err(CoupledLevelPoolError::input(
                "internal incoming requires one F64 value per pool on the configured device",
            ));
        }
        if self.bindings.is_empty() {
            return Ok(MergedIncoming {
                volume: internal.clone(),
                roundoff_bound_m3: 0.0,
            });
        }
        finite(client, internal, "internal incoming volume")?;
        let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
        reject(
            client,
            &client.lt(internal, &zero)?,
            "internal incoming volume is negative",
        )?;
        let (sum, low) = client.two_sum(internal, &external.incoming)?;
        finite(client, &sum, "combined internal and external incoming")?;
        let positive = client.mul(
            &client.gt(internal, &zero)?,
            &client.gt(&external.incoming, &zero)?,
        )?;
        let absorbed = client.mul(
            &positive,
            &client.maximum(
                &client.le(&sum, internal)?,
                &client.le(&sum, &external.incoming)?,
            )?,
        )?;
        reject(
            client,
            &absorbed,
            "positive internal or external incoming is absorbed in their shared merge. Rescale the receiving support",
        )?;
        let inactive = client.eq(&external.incoming, &zero)?;
        let volume = client.where_cond(&inactive, internal, &sum)?;
        let witness = client.where_cond(&inactive, &zero, &low)?;
        Ok(MergedIncoming {
            volume,
            roundoff_bound_m3: charge(client, self.pool_reduction, &client.abs(&witness)?)?,
        })
    }
}

impl<R: Runtime<DType = DType>> InflowStep<R> {
    pub(crate) fn all_interval_volumes(&self) -> bool {
        self.input_is_rate.iter().all(|is_rate| !is_rate)
    }

    pub(crate) fn original_volumes(&self) -> Tensor<R> {
        self.volumes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{binding, client, pools, rate, volume};
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn empty_and_explicit_zero_inflows_preserve_incoming_bits() -> TestResult {
        let client = client();
        for bindings in [vec![], vec![binding("boundary")]] {
            let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &bindings)?;
            let rates = if bindings.is_empty() {
                vec![]
            } else {
                vec![rate("boundary", 0.0)]
            };
            let rates = context.bind_inputs(&client, &rates)?;
            let original = context.zero_state(&client)?;
            let prepared = context.prepare(&client, &rates, &original, 1.0, 1)?;
            let incoming =
                Tensor::<CpuRuntime>::from_slice(&[-0.0f64, 1.0], &[2], client.device())?;
            let merged = context.merge_incoming(&client, &incoming, &prepared)?;
            assert_eq!(
                merged
                    .volume
                    .try_to_vec::<f64>()?
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                vec![(-0.0f64).to_bits(), 1.0f64.to_bits()]
            );
            assert_eq!(merged.roundoff_bound_m3, 0.0);
            assert_eq!(prepared.summary.volume_m3, 0.0);
            assert_eq!(prepared.summary.reduction_bound_m3, 0.0);
            assert_eq!(
                context.project_state(&client, &original)?,
                context.project_state(&client, &prepared.next)?
            );
        }
        Ok(())
    }

    #[test]
    fn grouped_boundaries_reject_absorbed_positive_volumes() -> TestResult {
        let client = client();
        let context = InflowContext::<CpuRuntime>::new(
            &client,
            &pools(4.0)?,
            &[binding("large"), binding("small")],
        )?;
        let rates = context.bind_inputs(
            &client,
            &[rate("small", f64::EPSILON / 4.0), rate("large", 1.0)],
        )?;
        let original = context.zero_state(&client)?;
        assert!(context.prepare(&client, &rates, &original, 1.0, 1).is_err());
        assert!(
            context
                .project_state(&client, &original)?
                .iter()
                .all(|entry| entry.cumulative_volume_m3 == 0.0 && entry.roundoff_bound_m3 == 0.0)
        );
        Ok(())
    }

    #[test]
    fn original_products_reject_positive_underflow_and_overflow() -> TestResult {
        let client = client();
        let context =
            InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
        let original = context.zero_state(&client)?;
        for (value, duration) in [(f64::from_bits(1), 0.25), (f64::MAX, 2.0)] {
            let rates = context.bind_inputs(&client, &[rate("boundary", value)])?;
            assert!(
                context
                    .prepare(&client, &rates, &original, duration, 1)
                    .is_err()
            );
        }
        let rates = context.bind_inputs(&client, &[rate("boundary", 0.25)])?;
        let prepared = context.prepare(&client, &rates, &original, 2.0, 1)?;
        assert_eq!(prepared.incoming.dtype(), DType::F64);
        assert_eq!(prepared.incoming.try_to_vec::<f64>()?, vec![0.5, 0.0]);
        Ok(())
    }

    #[test]
    fn integrated_lanes_are_masked_before_unused_rate_multiplication() -> TestResult {
        let client = client();
        let context =
            InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
        let original = context.zero_state(&client)?;
        let inputs = context.bind_inputs(&client, &[volume("boundary", 3.0)])?;
        let prepared = context.prepare(&client, &inputs, &original, f64::MAX, 1)?;
        assert_eq!(prepared.incoming.try_to_vec::<f64>()?, vec![3.0, 0.0]);
        let rate_inputs = context.bind_inputs(&client, &[rate("boundary", 3.0)])?;
        assert!(!rate_inputs.has_interval_volumes());
        assert!(
            context
                .prepare(&client, &rate_inputs, &original, f64::MAX, 1)
                .is_err()
        );
        Ok(())
    }
}
