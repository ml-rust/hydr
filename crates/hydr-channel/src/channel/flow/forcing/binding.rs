use super::super::{
    ChannelError, ChannelForcingInterval, ChannelLateralInput, sources::LateralContext,
};
use super::admission::{all, check_scalar, finite};
use super::{
    BoundChannelForcing, BoundLateralInputs, ChannelTensorBoundary, ChannelTensorLateralInput,
    ChannelTensorWaterAmount,
};
use crate::forcing::WaterAmount;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
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
        let inputs = self.host_inputs(client, &forcing.lateral_inputs)?;
        self.bind_owned(
            client,
            forcing.start_s,
            forcing.end_s,
            ChannelTensorBoundary::from_host(client, &forcing.left)?,
            ChannelTensorBoundary::from_host(client, &forcing.right)?,
            inputs,
        )
    }

    pub(in crate::channel::flow) fn host_inputs(
        &self,
        client: &R::Client,
        lateral_inputs: &[ChannelLateralInput],
    ) -> Result<Vec<ChannelTensorLateralInput<R>>, ChannelError> {
        let make = |value| Tensor::<R>::from_slice(&[value], &[1], client.device());
        let mut inputs = Vec::with_capacity(lateral_inputs.len());
        for input in lateral_inputs {
            let amount = match input.amount() {
                WaterAmount::Rate { rate_m3_per_s } => ChannelTensorWaterAmount::Rate {
                    rate_m3_per_s: make(rate_m3_per_s)?,
                },
                WaterAmount::IntervalVolume { volume_m3 } => {
                    ChannelTensorWaterAmount::IntervalVolume {
                        volume_m3: make(volume_m3)?,
                    }
                }
            };
            inputs.push(match input {
                ChannelLateralInput::Inflow {
                    boundary_id,
                    longitudinal_velocity_m_per_s,
                    ..
                } => ChannelTensorLateralInput::Inflow {
                    boundary_id: boundary_id.clone(),
                    amount,
                    longitudinal_velocity_m_per_s: make(*longitudinal_velocity_m_per_s)?,
                },
                ChannelLateralInput::Withdrawal { boundary_id, .. } => {
                    ChannelTensorLateralInput::Withdrawal {
                        boundary_id: boundary_id.clone(),
                        amount,
                    }
                }
            });
        }
        Ok(inputs)
    }

    pub fn bind_native(
        &self,
        client: &R::Client,
        start_s: f64,
        end_s: f64,
        left: ChannelTensorBoundary<R>,
        right: ChannelTensorBoundary<R>,
        inputs: Vec<ChannelTensorLateralInput<R>>,
    ) -> Result<BoundChannelForcing<R>, ChannelError> {
        let left = left.into_owned(client)?;
        let right = right.into_owned(client)?;
        let inputs = inputs
            .into_iter()
            .map(|input| input.into_owned(client))
            .collect::<Result<Vec<_>, _>>()?;
        self.bind_owned(client, start_s, end_s, left, right, inputs)
    }

    fn bind_owned(
        &self,
        client: &R::Client,
        start_s: f64,
        end_s: f64,
        left: ChannelTensorBoundary<R>,
        right: ChannelTensorBoundary<R>,
        inputs: Vec<ChannelTensorLateralInput<R>>,
    ) -> Result<BoundChannelForcing<R>, ChannelError> {
        self.check_endpoints(client, start_s, end_s, Some(&left), Some(&right))?;
        let lateral = self.bind_lateral_owned(client, inputs)?;
        Ok(BoundChannelForcing {
            definition: Arc::clone(&self.definition),
            start_s,
            end_s,
            left,
            right,
            lateral,
        })
    }

    pub(in crate::channel::flow) fn check_endpoints(
        &self,
        client: &R::Client,
        start_s: f64,
        end_s: f64,
        left: Option<&ChannelTensorBoundary<R>>,
        right: Option<&ChannelTensorBoundary<R>>,
    ) -> Result<(), ChannelError> {
        if (left.is_some_and(ChannelTensorBoundary::is_scalar)
            || right.is_some_and(ChannelTensorBoundary::is_scalar))
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
        super::super::definition::check_forcing_interval(start_s, end_s, start_s, end_s)?;
        if let Some(left) = left {
            left.check(client, &self.definition, 0)?;
        }
        let last = self
            .definition
            .geometry
            .len()
            .checked_sub(1)
            .ok_or_else(|| {
                ChannelError::input("cell count=0: construct a nonempty channel reach")
            })?;
        if let Some(right) = right {
            right.check(client, &self.definition, last)?;
        }
        Ok(())
    }

    pub(in crate::channel::flow) fn bind_lateral_owned(
        &self,
        client: &R::Client,
        inputs: Vec<ChannelTensorLateralInput<R>>,
    ) -> Result<BoundLateralInputs<R>, ChannelError> {
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
            let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
            if !finite(client, tensor)? || !all(client, &client.ge(tensor, &zero)?)? {
                return Err(ChannelError::input(format!(
                    "boundary_id={:?}: supply a finite nonnegative rate or integrated volume",
                    input.boundary_id()
                )));
            }
            has_nonzero_values |= !all(client, &client.eq(tensor, &zero)?)?;
            has_interval_volumes |= !input.amount().is_rate();
            if let ChannelTensorLateralInput::Inflow {
                longitudinal_velocity_m_per_s,
                ..
            } = input
            {
                check_scalar(
                    client,
                    longitudinal_velocity_m_per_s,
                    "lateral inflow velocity",
                )?;
                if !finite(client, longitudinal_velocity_m_per_s)? {
                    return Err(ChannelError::input(format!(
                        "boundary_id={:?}: supply a finite explicit inflow velocity",
                        input.boundary_id()
                    )));
                }
            }
        }
        let mut amounts = Vec::with_capacity(inputs.len());
        let mut rates = Vec::with_capacity(inputs.len());
        let mut velocities = Vec::with_capacity(self.inflow_indices.numel());
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
            if let ChannelTensorLateralInput::Inflow {
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
        Ok(lateral)
    }
}
