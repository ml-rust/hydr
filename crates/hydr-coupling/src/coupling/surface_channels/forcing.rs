//! Caller channel forcing over one common interval, checked and copied once per trial.

use super::SurfaceChannelError;
use super::context::Context;
use super::ports::check_endpoint;
use crate::channel::{
    ChannelBoundary, ChannelLateralInput, ChannelTensorBoundary, ChannelTensorFluxPort,
    ChannelTensorLateralInput, ChannelTensorWaterAmount,
};
use crate::forcing::WaterAmount;
use hydr_channel::internal::{ChannelCouplingEnd, SURFACE_EXCHANGE_BINDING_PREFIX};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeMap;

/// Complete external forcing for one named reach over one common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelReachForcing {
    /// Exact setup reach identity, present once per interval.
    pub reach_id: String,
    /// Policy at the upstream endpoint.
    pub left: ChannelBoundary,
    /// Policy at the downstream endpoint.
    pub right: ChannelBoundary,
    /// One input per caller lateral binding, including zeros.
    ///
    /// The coordinator supplies every `surface-exchange:` binding itself.
    pub lateral_inputs: Vec<ChannelLateralInput>,
}

/// Held channel forcing over one common interval.
///
/// Surface forcing stays in the surface setup's forcing schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelForcingInterval {
    /// Exact common accepted start time, in seconds.
    pub start_s: f64,
    /// End of forcing support, in seconds. Integrated volumes belong to this exact interval.
    pub end_s: f64,
    /// Exactly one record per setup reach, in any order.
    pub reaches: Vec<SurfaceChannelReachForcing>,
}

/// One caller withdrawal at a linked slot. The coordinator reserves it before exchange.
#[derive(Clone, Copy, Debug)]
pub(super) struct SlotWithdrawal {
    pub slot: usize,
    pub amount: WaterAmount,
}

/// Caller forcing of one reach on the configured device.
pub(super) struct BoundReach<R: Runtime<DType = DType>> {
    pub left: ChannelTensorBoundary<R>,
    pub right: ChannelTensorBoundary<R>,
    pub lateral: Vec<ChannelTensorLateralInput<R>>,
    pub withdrawals: Vec<SlotWithdrawal>,
}

/// Checked caller forcing in setup reach order.
pub(super) struct BoundForcing<R: Runtime<DType = DType>> {
    pub reaches: Vec<BoundReach<R>>,
    /// A caller integrated volume fixes the interval to the requested span.
    pub requires_exact_span: bool,
}

impl<R: Runtime<DType = DType>> BoundForcing<R> {
    /// Checks `forcing` against the accepted `start` and `requested` end, then copies it.
    pub fn bind(
        context: &Context<R>,
        reach_ids: &[&str],
        start: f64,
        requested: f64,
        forcing: &SurfaceChannelForcingInterval,
    ) -> Result<Self, SurfaceChannelError> {
        if !requested.is_finite()
            || requested <= start
            || forcing.start_s.to_bits() != start.to_bits()
            || !forcing.end_s.is_finite()
            || forcing.end_s < requested
        {
            return Err(SurfaceChannelError::input(format!(
                "accepted={start}, requested={requested}, forcing=[{},{}]: supply the exact accepted start and finite coverage of the requested end",
                forcing.start_s, forcing.end_s
            )));
        }
        let mut records = BTreeMap::new();
        for (index, record) in forcing.reaches.iter().enumerate() {
            if !reach_ids.contains(&record.reach_id.as_str())
                || records.insert(record.reach_id.as_str(), index).is_some()
            {
                return Err(SurfaceChannelError::input(format!(
                    "forcing reach_id={:?}: supply each setup reach exactly once",
                    record.reach_id
                )));
            }
        }
        let device = context.client.device();
        let scalar = |value: f64| Tensor::<R>::from_slice(&[value], &[1], device);
        let mut reaches = Vec::with_capacity(reach_ids.len());
        let mut requires_exact_span = false;
        for (reach, id) in reach_ids.iter().enumerate() {
            let record = records
                .get(id)
                .and_then(|&index| forcing.reaches.get(index))
                .ok_or_else(|| {
                    SurfaceChannelError::input(format!(
                        "forcing reach_id={id:?} is absent: supply one record per setup reach"
                    ))
                })?;
            let mut lateral = Vec::with_capacity(record.lateral_inputs.len());
            let mut withdrawals = Vec::new();
            for input in &record.lateral_inputs {
                let (boundary_id, amount) = match input {
                    ChannelLateralInput::Inflow {
                        boundary_id,
                        amount,
                        ..
                    }
                    | ChannelLateralInput::Withdrawal {
                        boundary_id,
                        amount,
                    } => (boundary_id, *amount),
                };
                if boundary_id.starts_with(SURFACE_EXCHANGE_BINDING_PREFIX) {
                    return Err(SurfaceChannelError::input(format!(
                        "reach={id:?} lateral input {boundary_id:?} uses the reserved prefix {SURFACE_EXCHANGE_BINDING_PREFIX:?}: omit surface-exchange inputs; the coordinator supplies them"
                    )));
                }
                requires_exact_span |= !amount.is_rate();
                let tensor_amount = match amount {
                    WaterAmount::Rate { rate_m3_per_s } => ChannelTensorWaterAmount::Rate {
                        rate_m3_per_s: scalar(rate_m3_per_s)?,
                    },
                    WaterAmount::IntervalVolume { volume_m3 } => {
                        ChannelTensorWaterAmount::IntervalVolume {
                            volume_m3: scalar(volume_m3)?,
                        }
                    }
                };
                lateral.push(match input {
                    ChannelLateralInput::Inflow {
                        longitudinal_velocity_m_per_s,
                        ..
                    } => ChannelTensorLateralInput::Inflow {
                        boundary_id: boundary_id.clone(),
                        amount: tensor_amount,
                        longitudinal_velocity_m_per_s: scalar(*longitudinal_velocity_m_per_s)?,
                    },
                    ChannelLateralInput::Withdrawal { .. } => {
                        if let Some(&slot) = context
                            .caller_slots
                            .get(reach)
                            .and_then(|slots| slots.get(boundary_id))
                        {
                            withdrawals.push(SlotWithdrawal { slot, amount });
                        }
                        ChannelTensorLateralInput::Withdrawal {
                            boundary_id: boundary_id.clone(),
                            amount: tensor_amount,
                        }
                    }
                });
            }
            for (end, policy) in [
                (ChannelCouplingEnd::Left, &record.left),
                (ChannelCouplingEnd::Right, &record.right),
            ] {
                check_endpoint(id, end, context.port_at(reach, end), Some(policy))?;
            }
            reaches.push(BoundReach {
                left: tensor_boundary(id, &record.left, &scalar)?,
                right: tensor_boundary(id, &record.right, &scalar)?,
                lateral,
                withdrawals,
            });
        }
        if requires_exact_span && forcing.end_s.to_bits() != requested.to_bits() {
            return Err(SurfaceChannelError::input(format!(
                "forcing end={}, requested={requested}: integrated lateral volumes require the exact requested span",
                forcing.end_s
            )));
        }
        Ok(Self {
            reaches,
            requires_exact_span,
        })
    }
}

/// Copies one host endpoint policy of reach `reach_id` onto the configured device.
fn tensor_boundary<R: Runtime<DType = DType>>(
    reach_id: &str,
    boundary: &ChannelBoundary,
    scalar: &impl Fn(f64) -> numr::error::Result<Tensor<R>>,
) -> Result<ChannelTensorBoundary<R>, SurfaceChannelError> {
    Ok(match boundary {
        ChannelBoundary::Wall => ChannelTensorBoundary::Wall,
        ChannelBoundary::Stage { elevation_m } => ChannelTensorBoundary::Stage {
            elevation_m: scalar(*elevation_m)?,
        },
        ChannelBoundary::QuiescentReservoir {
            elevation_m,
            entrance_loss_coefficient,
        } => ChannelTensorBoundary::QuiescentReservoir {
            elevation_m: scalar(*elevation_m)?,
            entrance_loss_coefficient: scalar(*entrance_loss_coefficient)?,
        },
        ChannelBoundary::Discharge { discharge_m3_per_s } => ChannelTensorBoundary::Discharge {
            discharge_m3_per_s: scalar(*discharge_m3_per_s)?,
        },
        ChannelBoundary::Exterior {
            depth_m,
            velocity_m_per_s,
        } => ChannelTensorBoundary::Exterior {
            depth_m: scalar(*depth_m)?,
            velocity_m_per_s: scalar(*velocity_m_per_s)?,
        },
        ChannelBoundary::FluxPort(port) => ChannelTensorBoundary::FluxPort(Box::new(
            ChannelTensorFluxPort::new(
                scalar(port.outward_water_rate_m3_per_s())?,
                scalar(port.outward_momentum_rate_m4_per_s2())?,
                scalar(port.signal_speed_m_per_s())?,
                port.exterior_invariant_m_per_s().map(scalar).transpose()?,
                port.provenance(),
            )
            .map_err(|source| SurfaceChannelError::Channel {
                reach_id: reach_id.to_owned(),
                source,
            })?,
        )),
    })
}
