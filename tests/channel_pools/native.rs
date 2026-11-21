use super::fixture::{DATUM, Result, connection, definition, forcing, single};
use super::reference::CASES;
use hydr::channel::*;
use hydr::coupling::*;
use hydr::forcing::WaterAmount;
use hydr::storage::{LevelPoolCurve, LevelPoolKnot};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::CertifiedMappedBisectionAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
    CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
};

fn operand<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: f64,
    view: usize,
) -> Result<Tensor<R>> {
    Ok(match view {
        0 => Tensor::from_slice(&[value], &[1], client.device())?,
        1 => Tensor::from_slice(&[f64::NAN, value], &[2], client.device())?.narrow(0, 1, 1)?,
        _ => Tensor::from_slice(
            &[f64::NAN, value, f64::INFINITY, -999.0],
            &[2, 2],
            client.device(),
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1))
        .narrow(0, 0, 1)?,
    })
}

fn native<R: Runtime<DType = DType>>(
    client: &R::Client,
    held: &ChannelPoolForcingInterval,
    view: usize,
) -> Result<ChannelPoolTensorForcingInterval<R>> {
    let value = |v| operand::<R>(client, v, view);
    let boundary = |b: &ChannelBoundary| -> Result<ChannelTensorBoundary<R>> {
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
            ChannelBoundary::Discharge { discharge_m3_per_s } => ChannelTensorBoundary::Discharge {
                discharge_m3_per_s: value(*discharge_m3_per_s)?,
            },
            ChannelBoundary::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => ChannelTensorBoundary::QuiescentReservoir {
                elevation_m: value(*elevation_m)?,
                entrance_loss_coefficient: value(*entrance_loss_coefficient)?,
            },
            ChannelBoundary::FluxPort(port) => {
                ChannelTensorBoundary::FluxPort(Box::new(ChannelTensorFluxPort::new(
                    value(port.outward_water_rate_m3_per_s())?,
                    value(port.outward_momentum_rate_m4_per_s2())?,
                    value(port.signal_speed_m_per_s())?,
                    port.exterior_invariant_m_per_s().map(value).transpose()?,
                    port.provenance(),
                )?))
            }
        })
    };
    let amount = |a: WaterAmount| -> Result<ChannelTensorWaterAmount<R>> {
        Ok(match a {
            WaterAmount::Rate { rate_m3_per_s } => ChannelTensorWaterAmount::Rate {
                rate_m3_per_s: value(rate_m3_per_s)?,
            },
            WaterAmount::IntervalVolume { volume_m3 } => ChannelTensorWaterAmount::IntervalVolume {
                volume_m3: value(volume_m3)?,
            },
        })
    };
    let reaches = held
        .reaches
        .iter()
        .map(|r| -> Result<_> {
            let lateral_inputs = r
                .lateral_inputs
                .iter()
                .map(|i| -> Result<_> {
                    Ok(match i {
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
                .collect::<Result<Vec<_>>>()?;
            Ok(ChannelPoolTensorReachForcing {
                reach_id: r.reach_id.clone(),
                left: r.left.as_ref().map(boundary).transpose()?,
                right: r.right.as_ref().map(boundary).transpose()?,
                lateral_inputs,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ChannelPoolTensorForcingInterval {
        start_s: held.start_s,
        end_s: held.end_s,
        reaches,
    })
}

/// Frozen reference runs as `(triangular, endpoint)`, and the views of the native alias runs.
///
/// A reference run uses the triangular ideal inflow row when `triangular` is true, and the
/// first frozen row otherwise. An alias run in view `v` continues in view `(v + 1) % 3`.
pub struct Coverage {
    pub references: Vec<(bool, ChannelPoolEndpoint)>,
    pub views: Vec<usize>,
}

impl Coverage {
    /// Both rows at both endpoints, and every view.
    pub fn all() -> Self {
        let mut references = Vec::new();
        for triangular in [false, true] {
            for endpoint in [ChannelPoolEndpoint::Left, ChannelPoolEndpoint::Right] {
                references.push((triangular, endpoint));
            }
        }
        Self {
            references,
            views: vec![0, 1, 2],
        }
    }

    /// Each row at a different endpoint, and one alias run that steps in both
    /// non-contiguous views.
    pub fn device() -> Self {
        Self {
            references: vec![
                (false, ChannelPoolEndpoint::Left),
                (true, ChannelPoolEndpoint::Right),
            ],
            views: vec![1],
        }
    }
}

pub fn execute<R>(client: R::Client, destination_client: R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + NextRepresentableOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let triangle = CASES
        .iter()
        .find(|case| case.name == "triangular-inflow-ideal" && case.pool_area == 1.0)
        .copied()
        .ok_or("Missing frozen triangular runtime reference")?;
    for &(triangular, endpoint) in &coverage.references {
        let reference_case = if triangular { triangle } else { CASES[0] };
        let (reference_setup, reference_initial) = single(reference_case, endpoint)?;
        let reference_origin =
            ChannelPoolBatch::<R>::new(client.clone(), reference_setup.clone(), reference_initial)?;
        let reference_next = reference_origin
            .trial_to(0.125, &forcing(&reference_setup, 0.0, 0.125))?
            .into_next();
        let reference_step = reference_next
            .last_step()?
            .ok_or("Missing runtime finite-pool reference receipt")?;
        let reference_reach = reference_next
            .reach("reach")
            .ok_or("Missing runtime reference reach")?;
        let reference_child = reference_reach
            .last_step()?
            .ok_or("Missing runtime reference child receipt")?;
        let (endpoint_index, normal) = match endpoint {
            ChannelPoolEndpoint::Left => (0, -1.0),
            ChannelPoolEndpoint::Right => (1, 1.0),
        };
        let reference_trace = reference_child.boundary_traces[endpoint_index]
            .as_ref()
            .ok_or("Missing runtime reference trace")?;
        let reference_state = reference_reach.state()?;
        let expected = reference_case.expected;
        for (actual, frozen) in [
            (reference_trace.depth_m, expected.depth),
            (reference_trace.outward_velocity_m_per_s, expected.velocity),
            (
                reference_trace.outward_discharge_m3_per_s,
                expected.discharge,
            ),
            (
                reference_trace.momentum_flux_m4_per_s2,
                expected.momentum_flux,
            ),
            (
                reference_step.transfers[0].signed_volume_m3,
                expected.transfer,
            ),
            (
                reference_step.transfers[0].midpoint_pool_elevation_m,
                expected.head,
            ),
            (
                reference_next.pools().states()?[0].volume_m3,
                expected.pool_volume,
            ),
            (reference_state[0].volume_m3, expected.channel_volume),
            (
                reference_step.transfers[0].longitudinal_momentum_impulse_m4_per_s,
                -normal * 0.125 * expected.momentum_flux,
            ),
            (
                reference_state[0].longitudinal_momentum_m4_per_s,
                normal * expected.right_momentum,
            ),
        ] {
            assert!(
                (actual - frozen).abs() <= 1e-10 + 1e-10 * frozen.abs(),
                "{} {endpoint:?}: actual={actual:e}, independent reference={frozen:e}",
                reference_case.name
            );
        }
    }
    for &view in &coverage.views {
        let (mut setup, mut initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
        setup.reaches[0].definition = definition(false, 1.0, 1.0, 2, 16.0, 0.375);
        initial.reaches[0].volume_m3 = vec![4.8; 2];
        initial.reaches[0].longitudinal_momentum_m4_per_s = vec![0.0; 2];
        setup.reaches.push(ChannelPoolReach {
            id: "second".into(),
            definition: definition(false, 1.0, 1.0, 1, 16.0, 0.25),
        });
        initial.reaches.push(ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![4.8],
            longitudinal_momentum_m4_per_s: vec![0.0],
        });
        setup.connections.push(connection(
            "second",
            "second",
            ChannelPoolEndpoint::Left,
            "pool",
            1.0,
        ));
        setup.pools[0].curve = LevelPoolCurve::new(
            vec![
                LevelPoolKnot {
                    volume_m3: 0.0,
                    stage_m: 0.125,
                },
                LevelPoolKnot {
                    volume_m3: 3.5,
                    stage_m: 0.875,
                },
            ],
            DATUM.into(),
            "Non-knot attachment cut with V(0.375)=7/6".into(),
        )?;
        initial.pools[0].volume_m3 = 2.8;
        let origin = ChannelPoolBatch::<R>::new(client.clone(), setup.clone(), initial.clone())?;
        let destination =
            ChannelPoolBatch::<R>::new(destination_client.clone(), setup.clone(), initial)?;
        let before = origin.snapshot_data()?;
        let dt = 1.0 / 1024.0;
        let mut held = forcing(&setup, 0.0, dt);
        held.reaches[0].left = Some(ChannelBoundary::QuiescentReservoir {
            elevation_m: 0.675,
            entrance_loss_coefficient: -0.0,
        });
        held.reaches[1].right = Some(ChannelBoundary::Exterior {
            depth_m: 0.3,
            velocity_m_per_s: -0.0,
        });
        let tensor = native::<R>(&client, &held, view)?;
        let host = origin.trial_to(dt, &held)?.into_next();
        let current = origin.trial_tensor_to(dt, &tensor)?.into_next();
        let saved = current.snapshot_data()?;
        assert_eq!(saved, host.snapshot_data()?);
        let step = current
            .last_step()?
            .ok_or("Missing native finite-pool receipt")?;
        let ChannelBoundary::QuiescentReservoir {
            entrance_loss_coefficient,
            ..
        } = step.forcing_interval.reaches[0]
            .left
            .as_ref()
            .ok_or("Missing retained reservoir forcing")?
        else {
            return Err("Retained policy changes".into());
        };
        assert_eq!(entrance_loss_coefficient.to_bits(), (-0.0_f64).to_bits());
        let ChannelBoundary::Exterior {
            velocity_m_per_s, ..
        } = step.forcing_interval.reaches[1]
            .right
            .as_ref()
            .ok_or("Missing retained exterior forcing")?
        else {
            return Err("Retained exterior policy changes".into());
        };
        assert_eq!(velocity_m_per_s.to_bits(), (-0.0_f64).to_bits());
        let cut = step
            .donor_cuts
            .iter()
            .find(|c| c.bed_elevation_m == 0.375)
            .ok_or("Missing non-knot cut")?;
        assert!(cut.available_lower_m3 <= 2.8 - 7.0 / 6.0);
        assert!((cut.available_lower_m3 - (2.8 - 7.0 / 6.0)).abs() < 1e-12);
        let checkpoint = current.snapshot();
        let replacement = operand::<R>(&client, f64::NAN, 0)?;
        for record in &tensor.reaches {
            for boundary in [&record.left, &record.right].into_iter().flatten() {
                match boundary {
                    ChannelTensorBoundary::QuiescentReservoir {
                        elevation_m,
                        entrance_loss_coefficient,
                    } => {
                        client.copy_into(elevation_m, &replacement)?;
                        client.copy_into(entrance_loss_coefficient, &replacement)?;
                    }
                    ChannelTensorBoundary::Exterior {
                        depth_m,
                        velocity_m_per_s,
                    } => {
                        client.copy_into(depth_m, &replacement)?;
                        client.copy_into(velocity_m_per_s, &replacement)?;
                    }
                    _ => return Err("Native alias fixture changes its external policies".into()),
                }
            }
        }
        assert_eq!(current.snapshot_data()?, saved);
        assert_eq!(checkpoint.data()?, saved);
        let trusted = destination.restore(&checkpoint)?;
        let cold = destination.import_snapshot(&saved)?;
        assert_eq!(trusted.snapshot_data()?, saved);
        assert_eq!(cold.snapshot_data()?, saved);
        held.start_s = dt;
        held.end_s = 2.0 * dt;
        let fresh = native::<R>(&client, &held, (view + 1) % 3)?;
        let expected = current
            .trial_tensor_to(2.0 * dt, &fresh)?
            .into_next()
            .snapshot_data()?;
        assert_eq!(
            host.trial_to(2.0 * dt, &held)?
                .into_next()
                .snapshot_data()?,
            expected
        );
        for restored in [trusted, cold] {
            assert_eq!(
                restored
                    .trial_tensor_to(2.0 * dt, &fresh)?
                    .into_next()
                    .snapshot_data()?,
                expected
            );
        }
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}
