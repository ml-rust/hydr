use super::super::ChannelError;
use super::super::conserved_point::ConservedPoint;
use super::super::conserved_update::{ExpansionClient, Parts, select_sum};
use super::arithmetic::{
    add_bounds, product_roundoff, propagated_bound, scalar_charge, upper_sum, zeros,
};
use super::{LateralContext, PreparedLateral, SourceApplication};
use crate::numerics::rounding::rounding_with_client;
use crate::numerics::water_transfers::{finite, reject};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
pub(super) const WITHDRAWAL_EXCEEDS: &str = "lateral withdrawal exceeds original donor water: shorten the interval or recompute integrated inputs";
pub(super) const WITHDRAWAL_ABSORBED: &str =
    "positive prescribed withdrawal is absorbed in donor storage: rescale the input or storage";
pub(super) const OUTFLOW_EXCEEDS: &str =
    "native face outflow plus lateral withdrawal exceeds original water: shorten the interval";
pub(super) const INFLOW_ABSORBED: &str = "positive prescribed inflow is absorbed in cell storage: rescale the input or receiving storage";

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
                withdrawn_momentum: None,
            });
        }
        let zero = zeros(client, 1)?;
        let withdrawal_active = client.gt(&prepared.withdrawal_volume, &zero)?;
        reject(
            client,
            &client.gt(&prepared.withdrawal_volume, original_volume)?,
            WITHDRAWAL_EXCEEDS,
        )
        .map_err(ChannelError::from_transfer)?;
        let (retained, withdrawal_low) =
            client.two_sum(original_volume, &client.neg(&prepared.withdrawal_volume)?)?;
        finite(client, &retained, "lateral retained water").map_err(ChannelError::from_transfer)?;
        reject(
            client,
            &client.mul(&withdrawal_active, &client.ge(&retained, original_volume)?)?,
            WITHDRAWAL_ABSORBED,
        )
        .map_err(ChannelError::from_transfer)?;
        let (transported_volume, transport_low) = client.two_sum(&retained, face_water_delta)?;
        finite(
            client,
            &transported_volume,
            "post-withdrawal transport water",
        )
        .map_err(ChannelError::from_transfer)?;
        reject(
            client,
            &client.lt(&transported_volume, &zero)?,
            OUTFLOW_EXCEEDS,
        )
        .map_err(ChannelError::from_transfer)?;
        let (volume, incoming_low) =
            client.two_sum(&transported_volume, &prepared.incoming_volume)?;
        finite(client, &volume, "lateral accepted water").map_err(ChannelError::from_transfer)?;
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
            INFLOW_ABSORBED,
        )
        .map_err(ChannelError::from_transfer)?;
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
        finite(client, &fraction, "lateral retained-water fraction")
            .map_err(ChannelError::from_transfer)?;
        let retained_product = client.mul(original_momentum, &fraction)?;
        let retained_momentum =
            client.where_cond(&withdrawal_active, &retained_product, original_momentum)?;
        finite(
            client,
            &retained_momentum,
            "lateral retained-water momentum",
        )
        .map_err(ChannelError::from_transfer)?;
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
        )
        .map_err(ChannelError::from_transfer)?;
        let (relationship, relationship_low) =
            client.two_sum(&retained_momentum, &client.neg(&recorded_retained)?)?;
        finite(
            client,
            &relationship,
            "retained momentum source relationship",
        )
        .map_err(ChannelError::from_transfer)?;
        let (transported_momentum, momentum_transport_low) =
            client.two_sum(&retained_momentum, face_momentum_delta)?;
        finite(
            client,
            &transported_momentum,
            "post-withdrawal transport momentum",
        )
        .map_err(ChannelError::from_transfer)?;
        let (momentum, momentum_incoming_low) =
            client.two_sum(&transported_momentum, &prepared.incoming_momentum)?;
        finite(client, &momentum, "lateral accepted momentum")
            .map_err(ChannelError::from_transfer)?;
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
            withdrawn_momentum: None,
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
            withdrawn_momentum: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{TestResult, client, definition};
    use super::*;
    use crate::channel::flow::{
        ChannelBoundary, ChannelInitialState, ChannelLateralDirection, ChannelReach,
        forcing::{ChannelTensorLateralInput, ChannelTensorWaterAmount},
    };
    use crate::forcing::WaterAmount;
    use numr::ops::{CorrectlyRoundedOps, HypotOps, MidpointOps, NextRepresentableOps};
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::CpuRuntime;
    use solvr::integrate::CertifiedSimpsonAlgorithms;
    use solvr::optimize::CertifiedMappedBisectionAlgorithms;
    use solvr::optimize::roots::{
        BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
    };

    fn native_case<R: Runtime<DType = DType>>(client: R::Client) -> TestResult
    where
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + CorrectlyRoundedOps<R>
            + HypotOps<R>
            + IndexingOps<R>
            + MidpointOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + TwoSumOps<R>
            + UnaryOps<R>
            + NextRepresentableOps<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>
            + BatchedBisectionAlgorithms<R>,
    {
        let source = LateralContext::<R>::new(
            &client,
            definition(&[
                ChannelLateralDirection::Inflow,
                ChannelLateralDirection::Withdrawal,
            ]),
        )?;
        let values = Tensor::<R>::from_slice(
            &[f64::NAN, 0.125, f64::INFINITY, 0.5],
            &[2, 2],
            client.device(),
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1));
        assert!(!values.is_contiguous());
        let incoming = values.narrow(0, 0, 1)?;
        let withdrawal = values.narrow(0, 1, 1)?;
        let velocity = Tensor::<R>::from_slice(&[-1.0], &[1], client.device())?;
        let bound = source.bind_native(
            &client,
            0.0,
            2.0,
            crate::channel::ChannelTensorBoundary::Wall,
            crate::channel::ChannelTensorBoundary::Wall,
            vec![
                ChannelTensorLateralInput::Withdrawal {
                    boundary_id: "source-1".into(),
                    amount: ChannelTensorWaterAmount::Rate {
                        rate_m3_per_s: withdrawal,
                    },
                },
                ChannelTensorLateralInput::Inflow {
                    boundary_id: "source-0".into(),
                    amount: ChannelTensorWaterAmount::IntervalVolume {
                        volume_m3: incoming,
                    },
                    longitudinal_velocity_m_per_s: velocity,
                },
            ],
        )?;
        let projected = bound.project(&client)?;
        assert_eq!(projected.lateral_inputs[0].boundary_id(), "source-1");
        assert_eq!(
            projected.lateral_inputs[0].amount(),
            WaterAmount::Rate { rate_m3_per_s: 0.5 }
        );
        assert_eq!(
            projected.lateral_inputs[1].amount(),
            WaterAmount::IntervalVolume { volume_m3: 0.125 }
        );
        client.copy_into(
            &values,
            &Tensor::<R>::from_slice(&[f64::NAN, f64::NAN], &[2], client.device())?,
        )?;
        assert_eq!(
            bound.lateral.original[1]
                .amount()
                .tensor()
                .item::<f64>()?
                .to_bits(),
            0.125_f64.to_bits()
        );
        let make = |value| Tensor::<R>::from_slice(&[value], &[1], client.device());
        let previous = source.initial_state(&client)?;
        let prepared = source.prepare(
            &client,
            &bound.lateral,
            &previous,
            &make(8.0)?,
            &make(3.0)?,
            1.0,
        )?;
        let application = source.apply_lateral_and_transport(
            &client,
            &make(8.0)?,
            &make(24.0)?,
            &make(0.0)?,
            &make(0.0)?,
            &prepared,
        )?;
        assert_eq!(application.volume.item::<f64>()?, 7.625);
        assert_eq!(application.momentum.item::<f64>()?, 22.375);
        let receipts =
            source.project_receipts(&client, &bound.lateral, &previous, &prepared.receipt)?;
        assert_eq!(receipts[0].volume_m3, 0.125);
        assert_eq!(receipts[0].momentum_m4_per_s, -0.125);
        assert_eq!(receipts[1].volume_m3, 0.5);
        assert_eq!(receipts[1].momentum_m4_per_s, -1.5);
        assert_eq!(
            previous.cumulative_volume.try_to_vec::<f64>()?,
            vec![0.0, 0.0]
        );
        let reach = ChannelReach::<R>::new(
            client.clone(),
            (*source.definition).clone(),
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![8.0],
                longitudinal_momentum_m4_per_s: vec![24.0],
            },
        )?;
        let endpoint = ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: 3.0,
        };
        let native = reach.context.lateral.bind_native(
            &client,
            0.0,
            0.01,
            crate::channel::ChannelTensorBoundary::from_host(&client, &endpoint)?,
            crate::channel::ChannelTensorBoundary::from_host(&client, &endpoint)?,
            bound.lateral.original.clone(),
        )?;
        let advanced = reach.step_bound(0.01, &native)?.next;
        let state = advanced.state()?;
        assert!((state[0].volume_m3 - 8.12).abs() <= 1e-12);
        assert!((state[0].longitudinal_momentum_m4_per_s - 23.86).abs() <= 1e-12);
        let data = advanced.snapshot_data()?;
        let last = data
            .last_interval
            .as_ref()
            .ok_or("missing native final interval")?;
        assert_eq!(last.diagnostics.retries, 0);
        assert_eq!(last.diagnostics.end_s.to_bits(), 0.01_f64.to_bits());
        assert!(
            last.forcing
                .lateral_inputs
                .iter()
                .zip(&projected.lateral_inputs)
                .all(|(a, b)| a.same_bits(b))
        );
        let cold = reach.import_snapshot(&data)?;
        assert_eq!(cold.snapshot_data()?, data);
        Ok(())
    }

    #[test]
    fn native_cpu_sources_retain_strides_input_order_and_momentum() -> TestResult {
        native_case::<CpuRuntime>(client())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_sources_retain_strides_input_order_and_momentum() -> TestResult {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        native_case::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
