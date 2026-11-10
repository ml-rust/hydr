//! Explicit fallible external-ledger and interval-receipt projections.

use super::super::CoupledLevelPoolError;
use super::{
    CoupledPoolInflowLedger, CoupledPoolInflowReceipt, InflowContext, InflowState, InflowStep,
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
            &step.rates,
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
        let rates = step.rates.try_to_vec::<f64>()?;
        let volumes = step.volumes.try_to_vec::<f64>()?;
        let before = step.before.volume.try_to_vec::<f64>()?;
        let after = step.next.volume.try_to_vec::<f64>()?;
        let witnesses = step.witness.try_to_vec::<f64>()?;
        let bound_before = step.before.bound.try_to_vec::<f64>()?;
        let bound_after = step.next.bound.try_to_vec::<f64>()?;
        Ok(self
            .bindings
            .iter()
            .zip(rates)
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
                                (((binding, rate_m3_per_s), volume_m3), cumulative_before_m3),
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
                    rate_m3_per_s,
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
