use super::boundary_common::{Shape, TestResult, definition, initial, reach};
use super::snapshot_bits::assert_same;
use super::transport::{forcing, native, operand, with_laterals};
use hydr::channel::*;
use hydr::forcing::WaterAmount;
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

type Native = ChannelTensorForcingInterval<crate::runtime::Runtime>;
type Operand = Tensor<crate::runtime::Runtime>;

fn replace_parameter(forcing: &mut Native, slot: usize, value: Operand) -> TestResult {
    let client = crate::runtime::client();
    match slot {
        0 => {
            forcing.left = ChannelTensorBoundary::Exterior {
                depth_m: value,
                velocity_m_per_s: operand(&client, 0.0, 0)?,
            }
        }
        1 => {
            forcing.left = ChannelTensorBoundary::Exterior {
                depth_m: operand(&client, 1.0, 0)?,
                velocity_m_per_s: value,
            }
        }
        2 => forcing.left = ChannelTensorBoundary::Stage { elevation_m: value },
        3 => {
            forcing.left = ChannelTensorBoundary::Discharge {
                discharge_m3_per_s: value,
            }
        }
        4 => {
            forcing.lateral_inputs[0] = ChannelTensorLateralInput::Inflow {
                boundary_id: "incoming".into(),
                amount: ChannelTensorWaterAmount::Rate {
                    rate_m3_per_s: value,
                },
                longitudinal_velocity_m_per_s: operand(&client, 0.0, 0)?,
            }
        }
        5 => {
            forcing.lateral_inputs[0] = ChannelTensorLateralInput::Inflow {
                boundary_id: "incoming".into(),
                amount: ChannelTensorWaterAmount::IntervalVolume { volume_m3: value },
                longitudinal_velocity_m_per_s: operand(&client, 0.0, 0)?,
            }
        }
        6 => {
            forcing.lateral_inputs[0] = ChannelTensorLateralInput::Inflow {
                boundary_id: "incoming".into(),
                amount: ChannelTensorWaterAmount::Rate {
                    rate_m3_per_s: operand(&client, 0.01, 0)?,
                },
                longitudinal_velocity_m_per_s: value,
            }
        }
        7 => {
            forcing.lateral_inputs[1] = ChannelTensorLateralInput::Withdrawal {
                boundary_id: "outgoing".into(),
                amount: ChannelTensorWaterAmount::Rate {
                    rate_m3_per_s: value,
                },
            }
        }
        _ => {
            forcing.lateral_inputs[1] = ChannelTensorLateralInput::Withdrawal {
                boundary_id: "outgoing".into(),
                amount: ChannelTensorWaterAmount::IntervalVolume { volume_m3: value },
            }
        }
    }
    Ok(())
}

#[test]
fn every_native_operand_rejects_invalid_layout_and_nonfinite_values_atomically() -> TestResult {
    let client = crate::runtime::client();
    let shape = Shape::Rectangle {
        gravity: 1.0,
        width: 7.0,
    };
    let origin = reach(
        with_laterals(definition(shape, 1, 4.0, 4.0, 1.0)),
        initial(shape, &[4.0], &[1.0], &[0.0]),
    )?;
    let end = 1.0 / 1024.0;
    let current = origin
        .step_tensor(end, &native(&client, &forcing(0.0, end, 2), 1)?)?
        .next;
    let before = current.snapshot_data()?;
    let malformed = vec![
        Tensor::from_slice(&[1.0_f64], &[], client.device())?,
        Tensor::from_slice(&[1.0_f64], &[1, 1], client.device())?,
        Tensor::from_slice(&[1.0_f64, 1.0], &[2], client.device())?,
        Tensor::from_slice(&[1.0_f32], &[1], client.device())?,
        Tensor::from_slice(&[1_i64], &[1], client.device())?,
        operand(&client, f64::NAN, 2)?,
        operand(&client, f64::INFINITY, 1)?,
        operand(&client, f64::NEG_INFINITY, 2)?,
    ];
    for slot in 0..9 {
        for value in &malformed {
            let mut tensor_forcing = native(&client, &forcing(end, end * 2.0, 2), 0)?;
            replace_parameter(&mut tensor_forcing, slot, value.clone())?;
            let error = current
                .step_tensor(end * 2.0, &tensor_forcing)
                .err()
                .ok_or("Invalid native operand enters an accepted interval")?;
            assert!(matches!(error, ChannelError::InvalidInput { .. }));
            assert_same(&current.snapshot_data()?, &before);
        }
    }
    Ok(())
}

#[test]
fn unsupported_native_endpoints_reject_without_changing_accepted_state() -> TestResult {
    let client = crate::runtime::client();
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
        for (depth, velocity, boundary) in [
            (0.0, 0.0, ChannelBoundary::Stage { elevation_m: 1.0 }),
            (
                0.0,
                0.0,
                ChannelBoundary::Discharge {
                    discharge_m3_per_s: 0.0,
                },
            ),
            (1.0, 1.0, ChannelBoundary::Stage { elevation_m: 1.0 }),
            (
                1.0,
                2.0,
                ChannelBoundary::Discharge {
                    discharge_m3_per_s: 0.0,
                },
            ),
            (1.0, 0.0, ChannelBoundary::Stage { elevation_m: 0.0 }),
            (1.0, 0.0, ChannelBoundary::Stage { elevation_m: 4.1 }),
            (
                1.0,
                0.0,
                ChannelBoundary::Discharge {
                    discharge_m3_per_s: 100.0,
                },
            ),
            (
                1.0,
                0.0,
                ChannelBoundary::Exterior {
                    depth_m: -1.0,
                    velocity_m_per_s: 0.0,
                },
            ),
            (
                1.0,
                0.0,
                ChannelBoundary::Exterior {
                    depth_m: 4.1,
                    velocity_m_per_s: 0.0,
                },
            ),
            (
                1.0,
                0.0,
                ChannelBoundary::Exterior {
                    depth_m: 0.0,
                    velocity_m_per_s: 0.1,
                },
            ),
        ] {
            let origin = reach(
                definition(shape, 1, 16.0, 4.0, 1.0),
                initial(shape, &[16.0], &[depth], &[velocity]),
            )?;
            let before = origin.snapshot_data()?;
            for left in [true, false] {
                let mut held = ChannelForcingInterval {
                    start_s: 0.0,
                    end_s: 1.0 / 1024.0,
                    left: ChannelBoundary::Wall,
                    right: ChannelBoundary::Wall,
                    lateral_inputs: vec![],
                };
                if left {
                    held.left = boundary.clone();
                } else {
                    held.right = boundary.clone();
                }
                assert!(
                    origin
                        .step_tensor(held.end_s, &native(&client, &held, 2)?)
                        .is_err()
                );
                assert_same(&origin.snapshot_data()?, &before);
            }
        }
    }
    Ok(())
}

#[test]
fn native_integrated_inputs_require_exact_spans_and_return_recomputed_intervals() -> TestResult {
    let client = crate::runtime::client();
    let shape = Shape::Triangle {
        gravity: 2.0,
        coefficient: 1.0,
    };
    let origin = reach(
        with_laterals(definition(shape, 1, 4.0, 4.0, 0.005)),
        initial(shape, &[4.0], &[1.0], &[0.0]),
    )?;
    let before = origin.snapshot_data()?;
    for source_rate in [0.0, 0.01] {
        let mut held = forcing(0.0, 0.01, 2);
        held.lateral_inputs = vec![
            ChannelLateralInput::Inflow {
                boundary_id: "incoming".into(),
                amount: WaterAmount::IntervalVolume {
                    volume_m3: source_rate * 0.01,
                },
                longitudinal_velocity_m_per_s: 0.0,
            },
            ChannelLateralInput::Withdrawal {
                boundary_id: "outgoing".into(),
                amount: WaterAmount::Rate { rate_m3_per_s: 0.0 },
            },
        ];
        assert!(matches!(
            origin.step_tensor(0.005, &native(&client, &held, 1)?),
            Err(ChannelError::InvalidInput { .. })
        ));
        let error = origin
            .step_tensor(0.01, &native(&client, &held, 2)?)
            .err()
            .ok_or("Integrated native input silently shortens")?;
        let ChannelError::RecomputeInterval {
            suggested_end_s, ..
        } = error
        else {
            return Err(format!("Unexpected native interval error: {error}").into());
        };
        assert!(suggested_end_s > 0.0 && suggested_end_s <= 0.005);
        assert_same(&origin.snapshot_data()?, &before);
        held.end_s = suggested_end_s;
        held.lateral_inputs[0] = ChannelLateralInput::Inflow {
            boundary_id: "incoming".into(),
            amount: WaterAmount::IntervalVolume {
                volume_m3: source_rate * suggested_end_s,
            },
            longitudinal_velocity_m_per_s: 0.0,
        };
        let native_next = origin
            .step_tensor(suggested_end_s, &native(&client, &held, 2)?)?
            .next;
        let host_next = origin.step(suggested_end_s, &held)?.next;
        assert_eq!(native_next.time().to_bits(), suggested_end_s.to_bits());
        assert_same(&native_next.snapshot_data()?, &host_next.snapshot_data()?);
    }
    for (start, end, requested) in [
        (-0.0, 0.01, 0.01),
        (0.001, 0.01, 0.01),
        (0.0, f64::INFINITY, 0.01),
        (0.0, 0.01, 0.02),
    ] {
        let mut held = forcing(start, end, 2);
        held.lateral_inputs[1] = ChannelLateralInput::Withdrawal {
            boundary_id: "outgoing".into(),
            amount: WaterAmount::Rate { rate_m3_per_s: 0.0 },
        };
        assert!(
            origin
                .step_tensor(requested, &native(&client, &held, 1)?)
                .is_err()
        );
        assert_same(&origin.snapshot_data()?, &before);
    }
    Ok(())
}

#[test]
fn native_lateral_rates_shorten_and_integrated_withdrawals_reserve_original_donor_water()
-> TestResult {
    let client = crate::runtime::client();
    let shape = Shape::Rectangle {
        gravity: 1.0,
        width: 3.0,
    };
    let origin = reach(
        with_laterals(definition(shape, 1, 1.0, 100.0, 1.0)),
        initial(shape, &[1.0], &[1.0], &[0.0]),
    )?;
    let before = origin.snapshot_data()?;
    let mut held = forcing(0.0, 0.25, 2);
    held.left = ChannelBoundary::Stage { elevation_m: 1.21 };
    held.right = ChannelBoundary::Discharge {
        discharge_m3_per_s: 0.3,
    };
    held.lateral_inputs = vec![
        ChannelLateralInput::Inflow {
            boundary_id: "incoming".into(),
            amount: WaterAmount::Rate { rate_m3_per_s: 0.0 },
            longitudinal_velocity_m_per_s: 0.0,
        },
        ChannelLateralInput::Withdrawal {
            boundary_id: "outgoing".into(),
            amount: WaterAmount::Rate {
                rate_m3_per_s: 12.0,
            },
        },
    ];
    let next = origin.step_tensor(0.25, &native(&client, &held, 2)?)?.next;
    assert!(next.time() > 0.0 && next.time() < 0.25);
    assert_same(
        &next.snapshot_data()?,
        &origin.step(0.25, &held)?.next.snapshot_data()?,
    );
    let receipt = next.last_step()?.ok_or("Missing native donor receipt")?;
    assert!(receipt.boundary_in_m3 > 0.0);
    assert!(receipt.lateral_out_m3 + receipt.boundary_out_m3 <= 3.0);
    for volume in [-1.0, 4.0] {
        held.lateral_inputs[1] = ChannelLateralInput::Withdrawal {
            boundary_id: "outgoing".into(),
            amount: WaterAmount::IntervalVolume { volume_m3: volume },
        };
        assert!(
            origin
                .step_tensor(0.25, &native(&client, &held, 1)?)
                .is_err()
        );
        assert_same(&origin.snapshot_data()?, &before);
    }
    Ok(())
}
