use super::boundary_common::{Shape, TestResult, definition, initial};
use super::snapshot_bits::assert_same;
use hydr::channel::*;
use hydr::forcing::WaterAmount;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

pub fn operand<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: f64,
    view: usize,
) -> Result<Tensor<R>, numr::error::Error> {
    match view {
        0 => Tensor::from_slice(&[value], &[1], client.device()),
        1 => Tensor::from_slice(&[f64::NAN, value, f64::INFINITY], &[3], client.device())?
            .narrow(0, 1, 1),
        _ => Ok(Tensor::from_slice(
            &[f64::NAN, value, f64::INFINITY, -999.0],
            &[2, 2],
            client.device(),
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1))
        .narrow(0, 0, 1)?),
    }
}

pub fn native<R: Runtime<DType = DType>>(
    client: &R::Client,
    forcing: &ChannelForcingInterval,
    view: usize,
) -> Result<ChannelTensorForcingInterval<R>, numr::error::Error> {
    let value = |v| operand::<R>(client, v, view);
    let boundary = |b: &ChannelBoundary| -> Result<_, numr::error::Error> {
        Ok(match b {
            ChannelBoundary::Wall => ChannelTensorBoundary::Wall,
            ChannelBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } => ChannelTensorBoundary::Exterior {
                depth_m: value(*depth_m)?,
                velocity_m_per_s: value(*velocity_m_per_s)?,
            },
            ChannelBoundary::Stage { elevation_m } => ChannelTensorBoundary::Stage {
                elevation_m: value(*elevation_m)?,
            },
            ChannelBoundary::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => ChannelTensorBoundary::QuiescentReservoir {
                elevation_m: value(*elevation_m)?,
                entrance_loss_coefficient: value(*entrance_loss_coefficient)?,
            },
            ChannelBoundary::Discharge { discharge_m3_per_s } => ChannelTensorBoundary::Discharge {
                discharge_m3_per_s: value(*discharge_m3_per_s)?,
            },
        })
    };
    let amount = |a: WaterAmount| -> Result<_, numr::error::Error> {
        Ok(match a {
            WaterAmount::Rate { rate_m3_per_s } => ChannelTensorWaterAmount::Rate {
                rate_m3_per_s: value(rate_m3_per_s)?,
            },
            WaterAmount::IntervalVolume { volume_m3 } => ChannelTensorWaterAmount::IntervalVolume {
                volume_m3: value(volume_m3)?,
            },
        })
    };
    let lateral_inputs = forcing
        .lateral_inputs
        .iter()
        .map(|input| {
            Ok(match input {
                ChannelLateralInput::Inflow {
                    boundary_id,
                    amount: a,
                    longitudinal_velocity_m_per_s,
                } => ChannelTensorLateralInput::Inflow {
                    boundary_id: boundary_id.clone(),
                    amount: amount(*a)?,
                    longitudinal_velocity_m_per_s: value(*longitudinal_velocity_m_per_s)?,
                },
                ChannelLateralInput::Withdrawal {
                    boundary_id,
                    amount: a,
                } => ChannelTensorLateralInput::Withdrawal {
                    boundary_id: boundary_id.clone(),
                    amount: amount(*a)?,
                },
            })
        })
        .collect::<Result<Vec<_>, numr::error::Error>>()?;
    Ok(ChannelTensorForcingInterval {
        start_s: forcing.start_s,
        end_s: forcing.end_s,
        left: boundary(&forcing.left)?,
        right: boundary(&forcing.right)?,
        lateral_inputs,
    })
}

pub fn with_laterals(mut layout: ChannelReachDefinition) -> ChannelReachDefinition {
    layout.lateral_bindings = [
        ("incoming", ChannelLateralDirection::Inflow),
        ("outgoing", ChannelLateralDirection::Withdrawal),
    ]
    .into_iter()
    .map(|(id, direction)| ChannelLateralBinding {
        boundary_id: id.into(),
        cell_id: "cell-0".into(),
        direction,
        ownership_assertion: "Independent analytical lateral input".into(),
    })
    .collect();
    layout
}

pub fn forcing(start_s: f64, end_s: f64, policy: usize) -> ChannelForcingInterval {
    let stage = ChannelBoundary::Stage { elevation_m: 1.0 };
    let discharge = ChannelBoundary::Discharge {
        discharge_m3_per_s: 0.03,
    };
    let exterior = ChannelBoundary::Exterior {
        depth_m: 1.0,
        velocity_m_per_s: -0.0,
    };
    let (left, right) = match policy {
        0 => (ChannelBoundary::Wall, exterior),
        1 => (exterior, ChannelBoundary::Wall),
        2 => (stage, discharge),
        3 => (discharge, stage),
        _ => (
            ChannelBoundary::Discharge {
                discharge_m3_per_s: -0.0,
            },
            ChannelBoundary::Discharge {
                discharge_m3_per_s: 0.0,
            },
        ),
    };
    ChannelForcingInterval {
        start_s,
        end_s,
        left,
        right,
        lateral_inputs: vec![
            ChannelLateralInput::Inflow {
                boundary_id: "incoming".into(),
                amount: WaterAmount::Rate {
                    rate_m3_per_s: 0.01,
                },
                longitudinal_velocity_m_per_s: -0.0,
            },
            ChannelLateralInput::Withdrawal {
                boundary_id: "outgoing".into(),
                amount: WaterAmount::IntervalVolume {
                    volume_m3: 1.0 / 1048576.0,
                },
            },
        ],
    }
}

pub fn overwrite_aliases<R>(
    client: &R::Client,
    forcing: &ChannelTensorForcingInterval<R>,
) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>,
{
    let replacement = operand::<R>(client, f64::NAN, 0)?;
    for boundary in [&forcing.left, &forcing.right] {
        match boundary {
            ChannelTensorBoundary::Wall => {}
            ChannelTensorBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } => {
                client.copy_into(depth_m, &replacement)?;
                client.copy_into(velocity_m_per_s, &replacement)?;
            }
            ChannelTensorBoundary::Stage { elevation_m } => {
                client.copy_into(elevation_m, &replacement)?
            }
            ChannelTensorBoundary::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => {
                client.copy_into(elevation_m, &replacement)?;
                client.copy_into(entrance_loss_coefficient, &replacement)?;
            }
            ChannelTensorBoundary::Discharge { discharge_m3_per_s } => {
                client.copy_into(discharge_m3_per_s, &replacement)?;
            }
        }
    }
    for input in &forcing.lateral_inputs {
        let amount = match input {
            ChannelTensorLateralInput::Inflow {
                amount,
                longitudinal_velocity_m_per_s,
                ..
            } => {
                client.copy_into(longitudinal_velocity_m_per_s, &replacement)?;
                amount
            }
            ChannelTensorLateralInput::Withdrawal { amount, .. } => amount,
        };
        match amount {
            ChannelTensorWaterAmount::Rate { rate_m3_per_s } => {
                client.copy_into(rate_m3_per_s, &replacement)?
            }
            ChannelTensorWaterAmount::IntervalVolume { volume_m3 } => {
                client.copy_into(volume_m3, &replacement)?
            }
        }
    }
    Ok(())
}

pub fn execute<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    assert_ne!(((0.03_f64 / 7.0) * 7.0).to_bits(), 0.03_f64.to_bits());
    for shape in [
        Shape::Rectangle {
            gravity: 1.0,
            width: 7.0,
        },
        Shape::Triangle {
            gravity: 2.0,
            coefficient: 1.0,
        },
    ] {
        for count in [1, 3] {
            for policy in 0..5 {
                let layout = with_laterals(definition(shape, count, 4.0, 4.0, 1.0));
                let depths: Vec<_> = (0..count).map(|i| 1.0 + i as f64 / 32.0).collect();
                let velocities: Vec<_> = (0..count).map(|i| i as f64 / 64.0).collect();
                let origin = ChannelReach::<R>::new(
                    client.clone(),
                    layout,
                    initial(shape, &vec![4.0; count], &depths, &velocities),
                )?;
                let before = origin.snapshot_data()?;
                let end = 1.0 / 1024.0;
                let held = forcing(0.0, end, policy);
                let host = origin.step(end, &held)?.next;
                let tensor_held = native::<R>(&client, &held, policy % 3)?;
                let current = origin.step_tensor(end, &tensor_held)?.next;
                assert_eq!(current.time().to_bits(), end.to_bits());
                let saved = current.snapshot_data()?;
                assert_same(&saved, &host.snapshot_data()?);
                assert_same(&origin.snapshot_data()?, &before);
                let receipt = current
                    .last_step()?
                    .ok_or("Missing native forcing receipt")?;
                for (endpoint, boundary) in [&held.left, &held.right].into_iter().enumerate() {
                    if let ChannelBoundary::Discharge { discharge_m3_per_s } = boundary {
                        let trace = receipt.boundary_traces[endpoint]
                            .as_ref()
                            .ok_or("Missing native discharge trace")?;
                        let sign = if endpoint == 0 { -1.0 } else { 1.0 };
                        assert_eq!(
                            trace.outward_discharge_m3_per_s.to_bits(),
                            (sign * discharge_m3_per_s).to_bits()
                        );
                        assert_eq!(
                            receipt.face_volume_m3[endpoint * count].to_bits(),
                            (discharge_m3_per_s * end).to_bits()
                        );
                    }
                }
                let checkpoint = current.snapshot();
                overwrite_aliases::<R>(&client, &tensor_held)?;
                assert_same(&current.snapshot_data()?, &saved);
                assert_same(&checkpoint.data()?, &saved);
                let trusted = origin.restore(&checkpoint)?;
                let cold = origin.import_snapshot(&saved)?;
                assert_same(&trusted.snapshot_data()?, &saved);
                assert_same(&cold.snapshot_data()?, &saved);
                let fresh = forcing(end, end * 2.0, policy);
                let tensor_fresh = native::<R>(&client, &fresh, (policy + 1) % 3)?;
                let expected = current
                    .step_tensor(end * 2.0, &tensor_fresh)?
                    .next
                    .snapshot_data()?;
                assert_same(
                    &host.step(end * 2.0, &fresh)?.next.snapshot_data()?,
                    &expected,
                );
                for restarted in [trusted, cold] {
                    assert_same(
                        &restarted
                            .step_tensor(end * 2.0, &tensor_fresh)?
                            .next
                            .snapshot_data()?,
                        &expected,
                    );
                }
                assert_same(&current.snapshot_data()?, &saved);
            }
        }
    }
    Ok(())
}

#[test]
fn native_forcing_preserves_host_packages_and_complete_continuation() -> TestResult {
    execute::<crate::runtime::Runtime>(crate::runtime::client())
}
