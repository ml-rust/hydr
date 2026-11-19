use super::super::{ChannelBoundary, ChannelError, ChannelGeometry, ChannelReachDefinition};
use super::ChannelTensorBoundary;
use super::admission::{all, check_scalar, finite};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> ChannelTensorBoundary<R> {
    pub(in crate::channel::flow) fn from_host(
        client: &R::Client,
        boundary: &ChannelBoundary,
    ) -> Result<Self, ChannelError> {
        let make = |value| Tensor::from_slice(&[value], &[1], client.device());
        Ok(match boundary {
            ChannelBoundary::Wall => Self::Wall,
            ChannelBoundary::Stage { elevation_m } => Self::Stage {
                elevation_m: make(*elevation_m)?,
            },
            ChannelBoundary::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => Self::QuiescentReservoir {
                elevation_m: make(*elevation_m)?,
                entrance_loss_coefficient: make(*entrance_loss_coefficient)?,
            },
            ChannelBoundary::Discharge { discharge_m3_per_s } => Self::Discharge {
                discharge_m3_per_s: make(*discharge_m3_per_s)?,
            },
            ChannelBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } => Self::Exterior {
                depth_m: make(*depth_m)?,
                velocity_m_per_s: make(*velocity_m_per_s)?,
            },
            ChannelBoundary::FluxPort(port) => Self::FluxPort(Box::new(
                super::ChannelTensorFluxPort::from_host(client, port)?,
            )),
        })
    }

    pub(in crate::channel::flow) fn project(&self) -> Result<ChannelBoundary, ChannelError> {
        Ok(match self {
            Self::Wall => ChannelBoundary::Wall,
            Self::Stage { elevation_m } => ChannelBoundary::Stage {
                elevation_m: elevation_m.item::<f64>()?,
            },
            Self::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => ChannelBoundary::QuiescentReservoir {
                elevation_m: elevation_m.item::<f64>()?,
                entrance_loss_coefficient: entrance_loss_coefficient.item::<f64>()?,
            },
            Self::Discharge { discharge_m3_per_s } => ChannelBoundary::Discharge {
                discharge_m3_per_s: discharge_m3_per_s.item::<f64>()?,
            },
            Self::Exterior {
                depth_m,
                velocity_m_per_s,
            } => ChannelBoundary::Exterior {
                depth_m: depth_m.item::<f64>()?,
                velocity_m_per_s: velocity_m_per_s.item::<f64>()?,
            },
            Self::FluxPort(port) => ChannelBoundary::FluxPort(port.project()?),
        })
    }
}

impl<R: Runtime<DType = DType>> ChannelTensorBoundary<R>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ReduceOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    pub(super) fn check(
        &self,
        client: &R::Client,
        definition: &ChannelReachDefinition,
        index: usize,
    ) -> Result<(), ChannelError> {
        let c = client;
        let reject = |message| ChannelError::input(message).at(index);
        if matches!(
            &definition.geometry,
            ChannelGeometry::PrismaticSurveyed { .. }
        ) && self.is_scalar()
        {
            return Err(reject(
                "surveyed scalar boundary is unsupported: supply Wall or complete Exterior depth and velocity",
            ));
        }
        match self {
            Self::Wall => Ok(()),
            Self::FluxPort(port) => port.check(c).map_err(|error| match error {
                ChannelError::InvalidInput { context, .. } => {
                    ChannelError::input(context).at(index)
                }
                other => other,
            }),
            Self::Stage { elevation_m } => {
                check_scalar(c, elevation_m, "stage elevation")?;
                if !finite(c, elevation_m)? {
                    return Err(reject(
                        "stage elevation: supply a finite elevation in the reach datum",
                    ));
                }
                Ok(())
            }
            Self::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => {
                check_scalar(c, elevation_m, "reservoir elevation")?;
                check_scalar(
                    c,
                    entrance_loss_coefficient,
                    "reservoir entrance loss coefficient",
                )?;
                if !finite(c, elevation_m)? {
                    return Err(reject(
                        "reservoir elevation: supply a finite elevation in the reach datum",
                    ));
                }
                let zero = Tensor::zeros(&[1], DType::F64, c.device())?;
                if !finite(c, entrance_loss_coefficient)?
                    || !all(c, &c.ge(entrance_loss_coefficient, &zero)?)?
                {
                    return Err(reject(
                        "reservoir entrance loss coefficient: supply a finite nonnegative dimensionless value",
                    ));
                }
                Ok(())
            }
            Self::Discharge { discharge_m3_per_s } => {
                check_scalar(c, discharge_m3_per_s, "endpoint discharge")?;
                if !finite(c, discharge_m3_per_s)? {
                    return Err(reject(
                        "endpoint discharge: supply a finite longitudinal discharge",
                    ));
                }
                Ok(())
            }
            Self::Exterior {
                depth_m: depth,
                velocity_m_per_s: velocity,
            } => {
                check_scalar(c, depth, "exterior depth")?;
                check_scalar(c, velocity, "exterior velocity")?;
                let zero = Tensor::zeros(&[1], DType::F64, c.device())?;
                if !finite(c, depth)?
                    || !finite(c, velocity)?
                    || !all(c, &c.ge(depth, &zero)?)?
                    || (all(c, &c.eq(depth, &zero)?)? && !all(c, &c.eq(velocity, &zero)?)?)
                {
                    return Err(reject(
                        "exterior state: supply supported finite depth and velocity, with zero velocity when dry",
                    ));
                }
                if matches!(
                    &definition.geometry,
                    ChannelGeometry::PrismaticSurveyed { .. }
                ) {
                    return Ok(());
                }
                let cap = Tensor::full_scalar(
                    &[1],
                    DType::F64,
                    definition.geometry.cell(index).maximum_depth_m,
                    c.device(),
                )?;
                if !all(c, &c.le(depth, &cap)?)? {
                    return Err(reject(
                        "exterior depth exceeds the geometric cap: supply a supported depth",
                    ));
                }
                let nonzero = all(c, &c.gt(depth, &zero)?)? && !all(c, &c.eq(velocity, &zero)?)?;
                let discharge = match &definition.geometry {
                    ChannelGeometry::PrismaticSurveyed { .. } => return Ok(()),
                    ChannelGeometry::Rectangular { cells } => {
                        let unit = c.mul(depth, velocity)?;
                        if !finite(c, &unit)? || (nonzero && all(c, &c.eq(&unit, &zero)?)?) {
                            return Err(reject(
                                "exterior unit discharge: supply a finite represented depth and velocity product",
                            ));
                        }
                        c.mul_scalar(&unit, cells[index].width_m)?
                    }
                    ChannelGeometry::PrismaticTriangular {
                        left_bank_slope_h_per_v: left,
                        right_bank_slope_h_per_v: right,
                        ..
                    } => {
                        let kh = c.mul_scalar(depth, left.midpoint(*right))?;
                        let area = c.mul(&kh, depth)?;
                        c.mul(&area, velocity)?
                    }
                };
                if !finite(c, &discharge)? || (nonzero && all(c, &c.eq(&discharge, &zero)?)?) {
                    return Err(reject(
                        "exterior discharge: supply a finite represented full-section discharge",
                    ));
                }
                Ok(())
            }
        }
    }
}
