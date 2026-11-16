use super::super::ChannelError;
use super::super::conserved_point::ConservedPoint;
use super::super::conserved_update::{ExpansionClient, Parts, select_sum};
use super::arithmetic::{
    add_bounds, product_roundoff, propagated_bound, scalar_charge, upper_sum, zeros,
};
use super::{LateralContext, PreparedLateral, SourceApplication};
use crate::numerics::subcatchment::bounds::rounding_with_client;
use crate::numerics::water_transfers::{finite, reject};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
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
    pub fn apply_lateral_and_transport(
        &self,
        client: &R::Client,
        original_volume: &Tensor<R>,
        original_momentum: &Tensor<R>,
        face_water_delta: &Tensor<R>,
        face_momentum_delta: &Tensor<R>,
        prepared: &PreparedLateral<R>,
    ) -> Result<SourceApplication<R>, ChannelError> {
        if !prepared.has_nonzero_values {
            return Ok(SourceApplication {
                volume: client.add(original_volume, face_water_delta)?,
                momentum: client.add(original_momentum, face_momentum_delta)?,
                volume_low: zeros(client, self.cells())?,
                momentum_low: zeros(client, self.cells())?,
                water_selection_bound: zeros(client, self.cells())?,
                momentum_selection_bound: zeros(client, self.cells())?,
                water_assembly_bound: zeros(client, self.cells())?,
                momentum_assembly_bound: zeros(client, self.cells())?,
                water_assembly_charge_m3: 0.0,
            });
        }
        let zero = zeros(client, 1)?;
        let withdrawal_active = client.gt(&prepared.withdrawal_volume, &zero)?;
        reject(
            client,
            &client.gt(&prepared.withdrawal_volume, original_volume)?,
            "lateral withdrawal exceeds original donor water: shorten the interval or recompute integrated inputs",
        )?;
        let (retained, withdrawal_low) =
            client.two_sum(original_volume, &client.neg(&prepared.withdrawal_volume)?)?;
        finite(client, &retained, "lateral retained water")?;
        reject(
            client,
            &client.mul(&withdrawal_active, &client.ge(&retained, original_volume)?)?,
            "positive prescribed withdrawal is absorbed in donor storage: rescale the input or storage",
        )?;
        let (transported_volume, transport_low) = client.two_sum(&retained, face_water_delta)?;
        finite(
            client,
            &transported_volume,
            "post-withdrawal transport water",
        )?;
        reject(
            client,
            &client.lt(&transported_volume, &zero)?,
            "native face outflow plus lateral withdrawal exceeds original water: shorten the interval",
        )?;
        let (volume, incoming_low) =
            client.two_sum(&transported_volume, &prepared.incoming_volume)?;
        finite(client, &volume, "lateral accepted water")?;
        let both_positive = client.mul(
            &client.gt(&transported_volume, &zero)?,
            &client.gt(&prepared.incoming_volume, &zero)?,
        )?;
        let absorbed = client.maximum(
            &client.le(&volume, &transported_volume)?,
            &client.le(&volume, &prepared.incoming_volume)?,
        )?;
        reject(
            client,
            &client.mul(&both_positive, &absorbed)?,
            "positive prescribed inflow is absorbed in cell storage: rescale the input or receiving storage",
        )?;
        let water_assembly_bound = add_bounds(
            client,
            &[
                &prepared.incoming_volume_bound,
                &prepared.withdrawal_volume_bound,
                &client.abs(&withdrawal_low)?,
                &client.abs(&transport_low)?,
                &client.abs(&incoming_low)?,
            ],
        )?;
        let water_assembly_charge_m3 = scalar_charge(
            upper_sum(client, &water_assembly_bound, self.cell_reduction)?,
            prepared.summary.physical_total_roundoff_bound_m3,
        )?;

        let wet = client.gt(original_volume, &zero)?;
        let one = client.add_scalar(&zero, 1.0)?;
        let denominator = client.where_cond(&wet, original_volume, &one)?;
        let fraction = client.div(&retained, &denominator)?;
        finite(client, &fraction, "lateral retained-water fraction")?;
        let retained_product = client.mul(original_momentum, &fraction)?;
        let retained_momentum =
            client.where_cond(&withdrawal_active, &retained_product, original_momentum)?;
        finite(
            client,
            &retained_momentum,
            "lateral retained-water momentum",
        )?;
        let ratio_active = client.mul(&withdrawal_active, &client.gt(&retained, &zero)?)?;
        let ratio_bound = client.where_cond(
            &ratio_active,
            &rounding_with_client::<R>(client, &fraction)?,
            &zero,
        )?;
        let retained_bound = add_bounds(
            client,
            &[
                &propagated_bound(client, &ratio_bound, original_momentum)?,
                &client.where_cond(
                    &withdrawal_active,
                    &product_roundoff(client, original_momentum, &fraction, &retained_product)?,
                    &zero,
                )?,
            ],
        )?;
        let (recorded_retained, recorded_low) = client.two_sum(
            original_momentum,
            &client.neg(&prepared.withdrawal_momentum)?,
        )?;
        finite(
            client,
            &recorded_retained,
            "recorded lateral donor momentum",
        )?;
        let (relationship, relationship_low) =
            client.two_sum(&retained_momentum, &client.neg(&recorded_retained)?)?;
        finite(
            client,
            &relationship,
            "retained momentum source relationship",
        )?;
        let (transported_momentum, momentum_transport_low) =
            client.two_sum(&retained_momentum, face_momentum_delta)?;
        finite(
            client,
            &transported_momentum,
            "post-withdrawal transport momentum",
        )?;
        let (momentum, momentum_incoming_low) =
            client.two_sum(&transported_momentum, &prepared.incoming_momentum)?;
        finite(client, &momentum, "lateral accepted momentum")?;
        let momentum_assembly_bound = add_bounds(
            client,
            &[
                &prepared.incoming_momentum_bound,
                &prepared.withdrawal_momentum_bound,
                &retained_bound,
                &client.abs(&recorded_low)?,
                &client.abs(&relationship)?,
                &client.abs(&relationship_low)?,
                &client.abs(&momentum_transport_low)?,
                &client.abs(&momentum_incoming_low)?,
            ],
        )?;
        Ok(SourceApplication {
            volume,
            momentum,
            volume_low: zeros(client, self.cells())?,
            momentum_low: zeros(client, self.cells())?,
            water_selection_bound: zeros(client, self.cells())?,
            momentum_selection_bound: zeros(client, self.cells())?,
            water_assembly_bound,
            momentum_assembly_bound,
            water_assembly_charge_m3,
        })
    }
}

impl<R: Runtime<DType = DType>> LateralContext<R>
where
    R::Client: ExpansionClient<R>,
{
    /// Apply surveyed transfers without lateral exchange as two-part expansion sums.
    ///
    /// Each state selects its center once. The discarded radius is the selection bound.
    pub fn apply_transport_expanded(
        &self,
        client: &R::Client,
        original_volume: &ConservedPoint<R>,
        original_momentum: &ConservedPoint<R>,
        water: &Parts<R>,
        momentum: &Parts<R>,
    ) -> Result<SourceApplication<R>, ChannelError> {
        let state = |point: &ConservedPoint<R>| -> Result<Parts<R>, ChannelError> {
            Ok(Parts {
                high: point.high.clone(),
                low: point.low.clone(),
                bound: zeros(client, self.cells())?,
            })
        };
        let (volume, water_selection_bound) = select_sum(
            client,
            &state(original_volume)?,
            water,
            "surveyed water transfer",
        )?;
        let (momentum, momentum_selection_bound) = select_sum(
            client,
            &state(original_momentum)?,
            momentum,
            "surveyed momentum transfer",
        )?;
        Ok(SourceApplication {
            volume: volume.high,
            momentum: momentum.high,
            volume_low: volume.low,
            momentum_low: momentum.low,
            water_selection_bound,
            momentum_selection_bound,
            water_assembly_bound: zeros(client, self.cells())?,
            momentum_assembly_bound: zeros(client, self.cells())?,
            water_assembly_charge_m3: 0.0,
        })
    }
}
