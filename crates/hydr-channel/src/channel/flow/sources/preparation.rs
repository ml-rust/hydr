use super::super::{ChannelError, forcing::BoundLateralInputs};
use super::arithmetic::{
    add_bounds, product_roundoff, reported_sum, scalar_charge, upper_sum, zeros,
};
use super::{LateralContext, LateralReceipt, LateralState, LateralSummary, PreparedLateral};
use crate::numerics::water_transfers::{AggregationKind, finite, reject, upward_add};
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
    pub fn prepare(
        &self,
        client: &R::Client,
        inputs: &BoundLateralInputs<R>,
        previous: &LateralState<R>,
        donor_volume: &Tensor<R>,
        donor_velocity: &Tensor<R>,
        dt: f64,
    ) -> Result<PreparedLateral<R>, ChannelError> {
        self.check_state(client, previous)?;
        if !self.definition.same_bits(&inputs.definition)
            || !dt.is_finite()
            || dt <= 0.0
            || !inputs.amounts.device().is_same(client.device())
        {
            return Err(ChannelError::input(format!(
                "lateral dt={dt}: use matching native source layout, finite positive duration, and the configured client"
            )));
        }
        for value in [donor_volume, donor_velocity] {
            if value.shape() != [self.cells()]
                || value.dtype() != DType::F64
                || !value.device().is_same(client.device())
            {
                return Err(ChannelError::input(
                    "lateral donor state requires one F64 value per cell on the configured client",
                ));
            }
            finite(client, value, "lateral donor state").map_err(ChannelError::from_transfer)?;
        }
        if self.count() == 0 {
            let cell_zero = zeros(client, self.cells())?;
            let binding_zero = zeros(client, 0)?;
            return Ok(PreparedLateral {
                incoming_volume: cell_zero.clone(),
                withdrawal_volume: cell_zero.clone(),
                incoming_momentum: cell_zero.clone(),
                withdrawal_momentum: cell_zero.clone(),
                incoming_volume_bound: cell_zero.clone(),
                withdrawal_volume_bound: cell_zero.clone(),
                incoming_momentum_bound: cell_zero.clone(),
                withdrawal_momentum_bound: cell_zero,
                next_state: previous.clone(),
                summary: LateralSummary {
                    incoming_m3: 0.0,
                    withdrawal_m3: 0.0,
                    physical_total_roundoff_bound_m3: 0.0,
                    counter_addition_bound_m3: 0.0,
                },
                receipt: LateralReceipt {
                    volume: binding_zero.clone(),
                    momentum: binding_zero.clone(),
                    applied_velocity: binding_zero.clone(),
                    velocity_defined: binding_zero.clone(),
                    counter_addition_low: binding_zero.clone(),
                    next_counter: previous.cumulative_volume.clone(),
                    next_counter_bound: previous.counter_roundoff_bound.clone(),
                },
                has_nonzero_values: false,
            });
        }
        let zero = zeros(client, 1)?;
        let rates = client.where_cond(&inputs.rate_mask, &inputs.amounts, &zero)?;
        let rate_volumes = client.mul_scalar(&rates, dt)?;
        finite(client, &rate_volumes, "lateral rate-times-duration")
            .map_err(ChannelError::from_transfer)?;
        reject(
            client,
            &client.mul(
                &client.gt(&rates, &zero)?,
                &client.eq(&rate_volumes, &zero)?,
            )?,
            "positive prescribed lateral rate produces zero volume: use a representable interval or rescale the input",
        )
        .map_err(ChannelError::from_transfer)?;
        let volumes = client.where_cond(&inputs.rate_mask, &rate_volumes, &inputs.amounts)?;
        let donor = client.index_select(donor_volume, 0, &self.cell_indices)?;
        let donor_u = client.index_select(donor_velocity, 0, &self.cell_indices)?;
        let withdrawal = client.where_cond(&self.inflow_mask, &zero, &volumes)?;
        reject(
            client,
            &client.mul(&client.gt(&withdrawal, &zero)?, &client.le(&donor, &zero)?)?,
            "positive lateral withdrawal has a dry donor: remove the withdrawal or supply existing donor water",
        )
        .map_err(ChannelError::from_transfer)?;
        let velocity = if self.inflow_indices.numel() == 0 {
            donor_u
        } else {
            client.index_put(&donor_u, 0, &self.inflow_indices, &inputs.inflow_velocities)?
        };
        let velocity_defined = client.maximum(&self.inflow_mask, &client.gt(&donor, &zero)?)?;
        let momentum = client.mul(&volumes, &velocity)?;
        let product_bound = product_roundoff(client, &volumes, &velocity, &momentum)?;
        let water = self
            .tree
            .aggregate(client, &volumes, AggregationKind::Water)
            .map_err(ChannelError::from_transfer)?;
        let momentum_group = self
            .tree
            .aggregate(client, &momentum, AggregationKind::Momentum)
            .map_err(ChannelError::from_transfer)?;
        let momentum_products = self
            .tree
            .aggregate(client, &product_bound, AggregationKind::Momentum)
            .map_err(ChannelError::from_transfer)?;
        let momentum_bound = add_bounds(
            client,
            &[
                &momentum_group.bound,
                &momentum_products.value,
                &momentum_products.bound,
            ],
        )?;
        let count = self.cells();
        let incoming_volume = water.value.narrow(0, 0, count)?;
        let withdrawal_volume = water.value.narrow(0, count, count)?;
        let (incoming_m3, in_error) = reported_sum(client, &incoming_volume, self.cell_reduction)?;
        let (withdrawal_m3, out_error) =
            reported_sum(client, &withdrawal_volume, self.cell_reduction)?;
        let physical_total_roundoff_bound_m3 = scalar_charge(in_error, out_error)?;
        let (next_counter, low) = client.two_sum(&previous.cumulative_volume, &volumes)?;
        finite(client, &next_counter, "lateral cumulative counter addition")
            .map_err(ChannelError::from_transfer)?;
        let next_counter = client.where_cond(
            &client.eq(&volumes, &zero)?,
            &previous.cumulative_volume,
            &next_counter,
        )?;
        let low = client.where_cond(&client.eq(&volumes, &zero)?, &zero, &low)?;
        let next_counter_bound =
            upward_add(client, &previous.counter_roundoff_bound, &client.abs(&low)?)
                .map_err(ChannelError::from_transfer)?;
        let counter_addition_bound_m3 =
            upper_sum(client, &client.abs(&low)?, self.binding_reduction)?;
        let signed_momentum =
            client.where_cond(&self.inflow_mask, &momentum, &client.neg(&momentum)?)?;
        Ok(PreparedLateral {
            incoming_volume,
            withdrawal_volume,
            incoming_momentum: momentum_group.value.narrow(0, 0, count)?,
            withdrawal_momentum: momentum_group.value.narrow(0, count, count)?,
            incoming_volume_bound: water.bound.narrow(0, 0, count)?,
            withdrawal_volume_bound: water.bound.narrow(0, count, count)?,
            incoming_momentum_bound: momentum_bound.narrow(0, 0, count)?,
            withdrawal_momentum_bound: momentum_bound.narrow(0, count, count)?,
            next_state: LateralState {
                cumulative_volume: next_counter.clone(),
                counter_roundoff_bound: next_counter_bound.clone(),
                cumulative_assembly_bound_m3: previous.cumulative_assembly_bound_m3,
            },
            summary: LateralSummary {
                incoming_m3,
                withdrawal_m3,
                physical_total_roundoff_bound_m3,
                counter_addition_bound_m3,
            },
            receipt: LateralReceipt {
                volume: volumes,
                momentum: signed_momentum,
                applied_velocity: velocity,
                velocity_defined,
                counter_addition_low: low,
                next_counter,
                next_counter_bound,
            },
            has_nonzero_values: inputs.has_nonzero_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{TestResult, client, definition, forcing, inflow};
    use super::*;
    use crate::channel::flow::{
        ChannelLateralDirection, ChannelLateralInput, ChannelLateralStateData,
    };
    use crate::forcing::WaterAmount;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn integrated_operands_never_enter_unused_rate_multiplication() -> TestResult {
        let client = client();
        let source = LateralContext::<CpuRuntime>::new(
            &client,
            definition(&[ChannelLateralDirection::Inflow]),
        )?;
        let bound = source.bind_host(&client, &forcing(vec![inflow(0, f64::MAX, 0.0)]))?;
        let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
        let prepared = source.prepare(
            &client,
            &bound.lateral,
            &source.initial_state(&client)?,
            &make(1.0)?,
            &make(0.0)?,
            2.0,
        )?;
        assert_eq!(
            prepared.receipt.volume.item::<f64>()?.to_bits(),
            f64::MAX.to_bits()
        );
        assert_eq!(prepared.incoming_volume.item::<f64>()?, f64::MAX);
        assert_eq!(prepared.incoming_momentum.item::<f64>()?, 0.0);
        Ok(())
    }

    #[test]
    fn positive_water_absorption_rejects_grouping_and_receiving_storage() -> TestResult {
        let client = client();
        let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
        let source = LateralContext::<CpuRuntime>::new(
            &client,
            definition(&[ChannelLateralDirection::Inflow; 2]),
        )?;
        let bound = source.bind_host(
            &client,
            &forcing(vec![inflow(0, 2.0_f64.powi(53), 0.0), inflow(1, 1.0, 0.0)]),
        )?;
        assert!(matches!(
            source.prepare(
                &client,
                &bound.lateral,
                &source.initial_state(&client)?,
                &make(1.0)?,
                &make(0.0)?,
                1.0
            ),
            Err(ChannelError::Numerical { .. })
        ));
        let source = LateralContext::<CpuRuntime>::new(
            &client,
            definition(&[ChannelLateralDirection::Inflow]),
        )?;
        let bound = source.bind_host(&client, &forcing(vec![inflow(0, 1.0, 0.0)]))?;
        let prepared = source.prepare(
            &client,
            &bound.lateral,
            &source.initial_state(&client)?,
            &make(2.0_f64.powi(53))?,
            &make(0.0)?,
            1.0,
        )?;
        assert!(matches!(
            source.apply_lateral_and_transport(
                &client,
                &make(2.0_f64.powi(53))?,
                &make(0.0)?,
                &make(0.0)?,
                &make(0.0)?,
                &prepared
            ),
            Err(ChannelError::Numerical { .. })
        ));
        Ok(())
    }

    #[test]
    fn signed_momentum_cancellation_retains_an_absolute_bound() -> TestResult {
        let client = client();
        let source = LateralContext::<CpuRuntime>::new(
            &client,
            definition(&[ChannelLateralDirection::Inflow; 3]),
        )?;
        let inputs = [2.0_f64.powi(53), 1.0, -2.0_f64.powi(53)]
            .into_iter()
            .enumerate()
            .map(|(i, u)| inflow(i, 1.0, u))
            .collect();
        let bound = source.bind_host(&client, &forcing(inputs))?;
        let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
        let prepared = source.prepare(
            &client,
            &bound.lateral,
            &source.initial_state(&client)?,
            &make(1.0)?,
            &make(0.0)?,
            1.0,
        )?;
        assert_eq!(prepared.incoming_volume.item::<f64>()?, 3.0);
        assert_eq!(prepared.incoming_momentum.item::<f64>()?, 0.0);
        assert!(prepared.incoming_momentum_bound.item::<f64>()? >= 1.0);
        Ok(())
    }

    #[test]
    fn counter_addition_retains_low_components_and_dry_zero_velocity() -> TestResult {
        let client = client();
        let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
        let source = LateralContext::<CpuRuntime>::new(
            &client,
            definition(&[ChannelLateralDirection::Inflow]),
        )?;
        let previous = source.restore_state(
            &client,
            &ChannelLateralStateData {
                cumulative_volume_m3: vec![2.0_f64.powi(53)],
                counter_roundoff_bound_m3: vec![0.0],
                cumulative_assembly_bound_m3: 0.0,
            },
        )?;
        let bound = source.bind_host(&client, &forcing(vec![inflow(0, 1.0, 0.0)]))?;
        let prepared = source.prepare(
            &client,
            &bound.lateral,
            &previous,
            &make(1.0)?,
            &make(0.0)?,
            1.0,
        )?;
        assert_eq!(prepared.receipt.counter_addition_low.item::<f64>()?, 1.0);
        assert_eq!(
            prepared.receipt.next_counter.item::<f64>()?,
            2.0_f64.powi(53)
        );
        assert!(prepared.receipt.next_counter_bound.item::<f64>()? >= 1.0);
        let source = LateralContext::<CpuRuntime>::new(
            &client,
            definition(&[ChannelLateralDirection::Withdrawal]),
        )?;
        let bound = source.bind_host(
            &client,
            &forcing(vec![ChannelLateralInput::Withdrawal {
                boundary_id: "source-0".into(),
                amount: WaterAmount::IntervalVolume { volume_m3: 0.0 },
            }]),
        )?;
        let previous = source.initial_state(&client)?;
        let prepared = source.prepare(
            &client,
            &bound.lateral,
            &previous,
            &make(0.0)?,
            &make(0.0)?,
            1.0,
        )?;
        let receipts =
            source.project_receipts(&client, &bound.lateral, &previous, &prepared.receipt)?;
        assert_eq!(receipts[0].applied_longitudinal_velocity_m_per_s, None);
        Ok(())
    }
}
