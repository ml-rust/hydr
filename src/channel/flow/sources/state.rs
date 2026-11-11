use super::super::{ChannelError, ChannelLateralStateData};
use super::arithmetic::{reported_sum, scalar_charge, upper_sum, zeros};
use super::{LateralContext, LateralState, LateralTotals, PreparedLateral};
use crate::numerics::water_transfers::{finite, reject};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LateralContext<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn initial_state(&self, client: &R::Client) -> Result<LateralState<R>, ChannelError> {
        Ok(LateralState {
            cumulative_volume: zeros(client, self.count())?,
            counter_roundoff_bound: zeros(client, self.count())?,
            cumulative_assembly_bound_m3: 0.0,
        })
    }

    pub fn check_state(
        &self,
        client: &R::Client,
        state: &LateralState<R>,
    ) -> Result<(), ChannelError> {
        for (field, value) in [
            ("lateral cumulative volumes", &state.cumulative_volume),
            ("lateral counter bounds", &state.counter_roundoff_bound),
        ] {
            if value.shape() != [self.count()]
                || value.dtype() != DType::F64
                || !value.device().is_same(client.device())
            {
                return Err(ChannelError::input(format!(
                    "{field} shape={:?}: restore one F64 counter per binding on the destination client",
                    value.shape()
                )));
            }
            finite(client, value, field)?;
            reject(
                client,
                &client.lt(value, &zeros(client, 1)?)?,
                "negative lateral counter or bound: restore nonnegative continuation values",
            )?;
        }
        if !state.cumulative_assembly_bound_m3.is_finite()
            || state.cumulative_assembly_bound_m3 < 0.0
            || (self.count() == 0 && state.cumulative_assembly_bound_m3 != 0.0)
        {
            return Err(ChannelError::input(format!(
                "lateral assembly bound={}: restore a finite nonnegative charge, zero for an empty source layout",
                state.cumulative_assembly_bound_m3
            )));
        }
        Ok(())
    }

    pub fn totals(
        &self,
        client: &R::Client,
        state: &LateralState<R>,
    ) -> Result<LateralTotals, ChannelError> {
        self.check_state(client, state)?;
        if self.count() == 0 {
            return Ok(LateralTotals {
                incoming_m3: 0.0,
                withdrawal_m3: 0.0,
                counter_abs_bound_m3: 0.0,
                assembly_abs_bound_m3: state.cumulative_assembly_bound_m3,
            });
        }
        let zero = zeros(client, 1)?;
        let incoming = client.where_cond(&self.inflow_mask, &state.cumulative_volume, &zero)?;
        let outgoing = client.where_cond(&self.inflow_mask, &zero, &state.cumulative_volume)?;
        let (incoming_m3, in_error) = reported_sum(client, &incoming, self.binding_reduction)?;
        let (withdrawal_m3, out_error) = reported_sum(client, &outgoing, self.binding_reduction)?;
        let counter_error = upper_sum(
            client,
            &state.counter_roundoff_bound,
            self.binding_reduction,
        )?;
        Ok(LateralTotals {
            incoming_m3,
            withdrawal_m3,
            counter_abs_bound_m3: scalar_charge(
                counter_error,
                scalar_charge(in_error, out_error)?,
            )?,
            assembly_abs_bound_m3: state.cumulative_assembly_bound_m3,
        })
    }

    pub fn finalize_state(
        &self,
        prepared: &PreparedLateral<R>,
        assembly_charge: f64,
    ) -> Result<LateralState<R>, ChannelError> {
        let mut next = prepared.next_state.clone();
        next.cumulative_assembly_bound_m3 =
            scalar_charge(next.cumulative_assembly_bound_m3, assembly_charge)?;
        Ok(next)
    }

    pub fn state_data(
        &self,
        client: &R::Client,
        state: &LateralState<R>,
    ) -> Result<ChannelLateralStateData, ChannelError> {
        self.check_state(client, state)?;
        Ok(ChannelLateralStateData {
            cumulative_volume_m3: state.cumulative_volume.try_to_vec::<f64>()?,
            counter_roundoff_bound_m3: state.counter_roundoff_bound.try_to_vec::<f64>()?,
            cumulative_assembly_bound_m3: state.cumulative_assembly_bound_m3,
        })
    }

    pub fn restore_state(
        &self,
        client: &R::Client,
        data: &ChannelLateralStateData,
    ) -> Result<LateralState<R>, ChannelError> {
        let state = LateralState {
            cumulative_volume: Tensor::from_slice(
                &data.cumulative_volume_m3,
                &[data.cumulative_volume_m3.len()],
                client.device(),
            )?,
            counter_roundoff_bound: Tensor::from_slice(
                &data.counter_roundoff_bound_m3,
                &[data.counter_roundoff_bound_m3.len()],
                client.device(),
            )?,
            cumulative_assembly_bound_m3: data.cumulative_assembly_bound_m3,
        };
        self.check_state(client, &state)?;
        Ok(state)
    }
}
