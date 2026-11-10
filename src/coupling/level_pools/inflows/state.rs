//! Capacity-bounded cumulative external counters on the selected runtime.

use super::super::{
    CoupledLevelPoolError,
    topology::{finite, reject},
};
use super::{CoupledPoolInflowLedger, InflowContext};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(crate) struct InflowState<R: Runtime<DType = DType>> {
    pub(super) volume: Tensor<R>,
    pub(super) bound: Tensor<R>,
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
    pub(crate) fn zero_state(
        &self,
        client: &R::Client,
    ) -> Result<InflowState<R>, CoupledLevelPoolError> {
        Ok(InflowState {
            volume: Tensor::zeros(&[self.bindings.len()], DType::F64, client.device())?,
            bound: Tensor::zeros(&[self.bindings.len()], DType::F64, client.device())?,
        })
    }

    pub(super) fn check_tensor(
        &self,
        client: &R::Client,
        value: &Tensor<R>,
    ) -> Result<(), CoupledLevelPoolError> {
        if value.shape() != [self.bindings.len()]
            || value.dtype() != DType::F64
            || !value.device().is_same(client.device())
        {
            return Err(CoupledLevelPoolError::input(
                "external ledger requires ordered F64 boundary tensors on the configured client device",
            ));
        }
        if !self.bindings.is_empty() {
            finite(client, value, "external ledger")?;
            let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
            reject(
                client,
                &client.lt(value, &zero)?,
                "external ledger contains negative volume or bound",
            )?;
        }
        Ok(())
    }

    pub(crate) fn check_state(
        &self,
        client: &R::Client,
        state: &InflowState<R>,
        steps: usize,
    ) -> Result<(), CoupledLevelPoolError> {
        self.check_tensor(client, &state.volume)?;
        self.check_tensor(client, &state.bound)?;
        if self.bindings.is_empty() {
            return Ok(());
        }
        let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
        if steps == 0 {
            reject(
                client,
                &client.maximum(
                    &client.ne(&state.volume, &zero)?,
                    &client.ne(&state.bound, &zero)?,
                )?,
                "zero-history external ledgers require zero counters and bounds",
            )?;
            return Ok(());
        }
        reject(
            client,
            &client.mul(
                &client.eq(&state.volume, &zero)?,
                &client.ne(&state.bound, &zero)?,
            )?,
            "zero external throughput requires a zero addition bound",
        )?;
        // Each boundary volume is credited after withdrawals. Nonnegative incoming
        // and final storage <= capacity imply each accepted boundary volume <= capacity.
        // The cumulative rounding witness bound covers every counter addition.
        let count = if steps as u128 > (1u128 << 53) {
            (steps as f64).next_up()
        } else {
            steps as f64
        };
        let product = client.mul_scalar(&self.capacity, count)?;
        let product_padding = client.add_scalar(
            &client.mul_scalar(&client.abs(&product)?, f64::EPSILON)?,
            f64::from_bits(1),
        )?;
        let product_upper = client.add(&product, &product_padding)?;
        let total = client.add(&product_upper, &state.bound)?;
        let padding = client.add_scalar(
            &client.mul_scalar(&client.abs(&total)?, f64::EPSILON)?,
            f64::from_bits(1),
        )?;
        let theoretical = client.add(&total, &padding)?;
        // Only theoretical overflow removes the finite capacity ceiling.
        let maximum = Tensor::<R>::full_scalar(&[], DType::F64, f64::MAX, client.device())?;
        let ceiling = client.where_cond(&client.isinf(&theoretical)?, &maximum, &theoretical)?;
        finite(client, &ceiling, "external capacity-and-step ceiling")?;
        reject(
            client,
            &client.gt(&state.volume, &ceiling)?,
            "external cumulative counter exceeds receiving capacity and accepted-step ceiling",
        )
    }

    pub(crate) fn restore_state(
        &self,
        client: &R::Client,
        state: &InflowState<R>,
        steps: usize,
    ) -> Result<InflowState<R>, CoupledLevelPoolError> {
        self.check_state(client, state, steps)?;
        Ok(state.clone())
    }

    pub(crate) fn import_state(
        &self,
        client: &R::Client,
        data: &[CoupledPoolInflowLedger],
        steps: usize,
    ) -> Result<InflowState<R>, CoupledLevelPoolError> {
        if data.len() != self.bindings.len() {
            return Err(CoupledLevelPoolError::input(
                "external ledger count differs from declared boundaries",
            ));
        }
        let mut volumes = Vec::with_capacity(data.len());
        let mut bounds = Vec::with_capacity(data.len());
        for (binding, ledger) in self.bindings.iter().zip(data) {
            if ledger.boundary_id != binding.boundary_id || ledger.input_id != binding.input_id {
                return Err(CoupledLevelPoolError::input(
                    "external ledger identities or numerical order differ from declared bindings",
                ));
            }
            for value in [ledger.cumulative_volume_m3, ledger.roundoff_bound_m3] {
                if !value.is_finite() || value < 0.0 || (value == 0.0 && value.to_bits() != 0) {
                    return Err(CoupledLevelPoolError::input(
                        "external ledger values require finite nonnegative volumes and canonical zero",
                    ));
                }
            }
            volumes.push(ledger.cumulative_volume_m3);
            bounds.push(ledger.roundoff_bound_m3);
        }
        let state = InflowState {
            volume: Tensor::from_slice(&volumes, &[volumes.len()], client.device())?,
            bound: Tensor::from_slice(&bounds, &[bounds.len()], client.device())?,
        };
        self.check_state(client, &state, steps)?;
        Ok(state)
    }
}
