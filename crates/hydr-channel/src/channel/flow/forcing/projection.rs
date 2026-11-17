use super::super::{ChannelError, ChannelForcingInterval, ChannelLateralInput};
use super::{
    BoundChannelForcing, BoundLateralInputs, ChannelTensorLateralInput, ChannelTensorWaterAmount,
};
use crate::forcing::WaterAmount;
use numr::dtype::DType;
use numr::runtime::{Device, Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> BoundLateralInputs<R> {
    pub fn project(&self, client: &R::Client) -> Result<Vec<ChannelLateralInput>, ChannelError> {
        let mut result = Vec::with_capacity(self.original.len());
        for input in &self.original {
            let tensor = input.amount().tensor();
            if !tensor.device().is_same(client.device()) {
                return Err(ChannelError::input(
                    "native lateral operand belongs to another device: project with its configured client",
                ));
            }
            let amount = match input.amount() {
                ChannelTensorWaterAmount::Rate { rate_m3_per_s } => WaterAmount::Rate {
                    rate_m3_per_s: rate_m3_per_s.item::<f64>()?,
                },
                ChannelTensorWaterAmount::IntervalVolume { volume_m3 } => {
                    WaterAmount::IntervalVolume {
                        volume_m3: volume_m3.item::<f64>()?,
                    }
                }
            };
            result.push(match input {
                ChannelTensorLateralInput::Inflow {
                    boundary_id,
                    longitudinal_velocity_m_per_s,
                    ..
                } => ChannelLateralInput::Inflow {
                    boundary_id: boundary_id.clone(),
                    amount,
                    longitudinal_velocity_m_per_s: longitudinal_velocity_m_per_s.item::<f64>()?,
                },
                ChannelTensorLateralInput::Withdrawal { boundary_id, .. } => {
                    ChannelLateralInput::Withdrawal {
                        boundary_id: boundary_id.clone(),
                        amount,
                    }
                }
            });
        }
        Ok(result)
    }
}

impl<R: Runtime<DType = DType>> BoundChannelForcing<R> {
    pub fn project(&self, client: &R::Client) -> Result<ChannelForcingInterval, ChannelError> {
        Ok(ChannelForcingInterval {
            start_s: self.start_s,
            end_s: self.end_s,
            left: self.left.project()?,
            right: self.right.project()?,
            lateral_inputs: self.lateral.project(client)?,
        })
    }
}
