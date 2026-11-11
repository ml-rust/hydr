use super::super::{
    ChannelBoundary, ChannelError, ChannelForcingInterval, ChannelLateralInput,
    sources::LateralContext,
};
use super::{BoundChannelForcing, BoundLateralInputs, NativeLateralInput, NativeWaterAmount};
use crate::forcing::WaterAmount;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeMap;
use std::sync::Arc;

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
    pub fn bind_host(
        &self,
        client: &R::Client,
        forcing: &ChannelForcingInterval,
    ) -> Result<BoundChannelForcing<R>, ChannelError> {
        let make = |value| Tensor::<R>::from_slice(&[value], &[1], client.device());
        let mut inputs = Vec::with_capacity(forcing.lateral_inputs.len());
        for input in &forcing.lateral_inputs {
            let amount = match input.amount() {
                WaterAmount::Rate { rate_m3_per_s } => NativeWaterAmount::Rate {
                    rate_m3_per_s: make(rate_m3_per_s)?,
                },
                WaterAmount::IntervalVolume { volume_m3 } => NativeWaterAmount::IntervalVolume {
                    volume_m3: make(volume_m3)?,
                },
            };
            inputs.push(match input {
                ChannelLateralInput::Inflow {
                    boundary_id,
                    longitudinal_velocity_m_per_s,
                    ..
                } => NativeLateralInput::Inflow {
                    boundary_id: boundary_id.clone(),
                    amount,
                    longitudinal_velocity_m_per_s: make(*longitudinal_velocity_m_per_s)?,
                },
                ChannelLateralInput::Withdrawal { boundary_id, .. } => {
                    NativeLateralInput::Withdrawal {
                        boundary_id: boundary_id.clone(),
                        amount,
                    }
                }
            });
        }
        self.bind_native(
            client,
            forcing.start_s,
            forcing.end_s,
            forcing.left.clone(),
            forcing.right.clone(),
            inputs,
        )
    }

    pub fn bind_native(
        &self,
        client: &R::Client,
        start_s: f64,
        end_s: f64,
        left: ChannelBoundary,
        right: ChannelBoundary,
        inputs: Vec<NativeLateralInput<R>>,
    ) -> Result<BoundChannelForcing<R>, ChannelError> {
        if (left.is_scalar() || right.is_scalar())
            && (R::float_arithmetic_contract(DType::F64)
                != numr::runtime::FloatArithmeticContract::Ieee754Basic
                || R::square_root_contract(DType::F64)
                    != numr::runtime::SquareRootContract::Ieee754RoundToNearest)
        {
            return Err(ChannelError::input(format!(
                "scalar endpoint F64 contracts: arithmetic={:?}, square_root={:?}. Select a runtime declaring IEEE 754 basic arithmetic and round-to-nearest square roots",
                R::float_arithmetic_contract(DType::F64),
                R::square_root_contract(DType::F64)
            )));
        }
        if !start_s.is_finite() {
            return Err(ChannelError::input(format!(
                "forcing start={start_s}: supply a finite accepted time"
            )));
        }
        super::super::definition::check_endpoint_interval(
            start_s,
            end_s,
            start_s,
            end_s,
            &left,
            &right,
            &self.definition,
        )?;
        if inputs.len() != self.count() {
            return Err(ChannelError::input(format!(
                "lateral input count={}, bindings={}: supply every binding exactly once",
                inputs.len(),
                self.count()
            )));
        }
        let mut ordered = BTreeMap::new();
        let mut has_interval_volumes = false;
        let mut has_nonzero_values = false;
        for input in &inputs {
            if ordered.insert(input.boundary_id(), input).is_some() {
                return Err(ChannelError::input(format!(
                    "duplicate boundary_id={:?}: supply one lateral input per binding",
                    input.boundary_id()
                )));
            }
            let tensor = input.amount().tensor();
            check_scalar(client, tensor, "lateral amount")?;
            let value = tensor.item::<f64>()?;
            if !value.is_finite() || value < 0.0 {
                return Err(ChannelError::input(format!(
                    "boundary_id={:?}, amount={value}: supply a finite nonnegative rate or integrated volume",
                    input.boundary_id()
                )));
            }
            has_nonzero_values |= value != 0.0;
            has_interval_volumes |= !input.amount().is_rate();
            if let NativeLateralInput::Inflow {
                longitudinal_velocity_m_per_s,
                ..
            } = input
            {
                check_scalar(
                    client,
                    longitudinal_velocity_m_per_s,
                    "lateral inflow velocity",
                )?;
                let velocity = longitudinal_velocity_m_per_s.item::<f64>()?;
                if !velocity.is_finite() {
                    return Err(ChannelError::input(format!(
                        "boundary_id={:?}, velocity={velocity}: supply a finite explicit inflow velocity",
                        input.boundary_id()
                    )));
                }
            }
        }
        let mut amounts = Vec::with_capacity(inputs.len());
        let mut rates = Vec::with_capacity(inputs.len());
        let mut velocities = Vec::new();
        for binding in &self.definition.lateral_bindings {
            let input = ordered
                .remove(binding.boundary_id.as_str())
                .ok_or_else(|| {
                    ChannelError::input(format!(
                        "missing boundary_id={:?}: supply every declared lateral input",
                        binding.boundary_id
                    ))
                })?;
            if input.direction() != binding.direction {
                return Err(ChannelError::input(format!(
                    "boundary_id={:?}: supply the declared lateral direction",
                    binding.boundary_id
                )));
            }
            amounts.push(input.amount().tensor());
            rates.push(if input.amount().is_rate() { 1.0 } else { 0.0 });
            if let NativeLateralInput::Inflow {
                longitudinal_velocity_m_per_s,
                ..
            } = input
            {
                velocities.push(longitudinal_velocity_m_per_s);
            }
        }
        let pack = |values: &[&Tensor<R>]| -> Result<Tensor<R>, ChannelError> {
            if values.is_empty() {
                Ok(Tensor::zeros(&[0], DType::F64, client.device())?)
            } else {
                Ok(client.cat(values, 0)?)
            }
        };
        let lateral = BoundLateralInputs {
            definition: Arc::clone(&self.definition),
            amounts: pack(&amounts)?,
            rate_mask: Tensor::from_slice(&rates, &[rates.len()], client.device())?,
            inflow_velocities: pack(&velocities)?,
            original: inputs,
            has_interval_volumes,
            has_nonzero_values,
        };
        Ok(BoundChannelForcing {
            definition: Arc::clone(&self.definition),
            start_s,
            end_s,
            left,
            right,
            lateral,
        })
    }
}

fn check_scalar<R: Runtime<DType = DType>>(
    client: &R::Client,
    tensor: &Tensor<R>,
    name: &str,
) -> Result<(), ChannelError> {
    if tensor.shape() != [1]
        || tensor.dtype() != DType::F64
        || !tensor.device().is_same(client.device())
    {
        return Err(ChannelError::input(format!(
            "{name} shape={:?}, dtype={:?}: supply an actual F64 scalar tensor on the configured client",
            tensor.shape(),
            tensor.dtype()
        )));
    }
    Ok(())
}
