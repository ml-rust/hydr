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
