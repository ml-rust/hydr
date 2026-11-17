use super::super::ChannelError;
use super::admission::check_scalar;
use super::{ChannelTensorBoundary, ChannelTensorLateralInput, ChannelTensorWaterAmount};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn detach_scalar<R: Runtime<DType = DType>>(
    client: &R::Client,
    tensor: &Tensor<R>,
    name: &str,
) -> Result<Tensor<R>, ChannelError>
where
    R::Client: BinaryOps<R>,
{
    check_scalar(client, tensor, name)?;
    // Retained forcing owns storage because callers can overwrite their original tensor aliases.
    let owned = Tensor::zeros(&[1], DType::F64, client.device())?;
    client.copy_into(&owned, tensor)?;
    Ok(owned)
}

impl<R: Runtime<DType = DType>> ChannelTensorBoundary<R>
where
    R::Client: BinaryOps<R>,
{
    pub(in crate::channel::flow) fn into_owned(
        self,
        client: &R::Client,
    ) -> Result<Self, ChannelError> {
        Ok(match self {
            Self::Wall => Self::Wall,
            Self::Stage { elevation_m } => Self::Stage {
                elevation_m: detach_scalar(client, &elevation_m, "stage elevation")?,
            },
            Self::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => Self::QuiescentReservoir {
                elevation_m: detach_scalar(client, &elevation_m, "reservoir elevation")?,
                entrance_loss_coefficient: detach_scalar(
                    client,
                    &entrance_loss_coefficient,
                    "reservoir entrance loss coefficient",
                )?,
            },
            Self::Discharge { discharge_m3_per_s } => Self::Discharge {
                discharge_m3_per_s: detach_scalar(
                    client,
                    &discharge_m3_per_s,
                    "endpoint discharge",
                )?,
            },
            Self::Exterior {
                depth_m,
                velocity_m_per_s,
            } => Self::Exterior {
                depth_m: detach_scalar(client, &depth_m, "exterior depth")?,
                velocity_m_per_s: detach_scalar(client, &velocity_m_per_s, "exterior velocity")?,
            },
        })
    }
}

impl<R: Runtime<DType = DType>> ChannelTensorWaterAmount<R>
where
    R::Client: BinaryOps<R>,
{
    fn into_owned(self, client: &R::Client) -> Result<Self, ChannelError> {
        let value = detach_scalar(client, self.tensor(), "lateral amount")?;
        Ok(match self {
            Self::Rate { .. } => Self::Rate {
                rate_m3_per_s: value,
            },
            Self::IntervalVolume { .. } => Self::IntervalVolume { volume_m3: value },
        })
    }
}

impl<R: Runtime<DType = DType>> ChannelTensorLateralInput<R>
where
    R::Client: BinaryOps<R>,
{
    pub(in crate::channel::flow) fn into_owned(
        self,
        client: &R::Client,
    ) -> Result<Self, ChannelError> {
        Ok(match self {
            Self::Inflow {
                boundary_id,
                amount,
                longitudinal_velocity_m_per_s,
            } => Self::Inflow {
                boundary_id,
                amount: amount.into_owned(client)?,
                longitudinal_velocity_m_per_s: detach_scalar(
                    client,
                    &longitudinal_velocity_m_per_s,
                    "lateral inflow velocity",
                )?,
            },
            Self::Withdrawal {
                boundary_id,
                amount,
            } => Self::Withdrawal {
                boundary_id,
                amount: amount.into_owned(client)?,
            },
        })
    }
}
