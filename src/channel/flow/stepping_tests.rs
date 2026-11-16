use super::*;
use crate::channel::{
    ChannelBoundary, ChannelConfig, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

#[test]
fn rounded_numerical_transfers_preserve_water_and_consume_tolerance()
-> Result<(), Box<dyn std::error::Error>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "tiny positive wet/dry transfer arithmetic fixture".into(),
        geometry: crate::channel::ChannelGeometry::Rectangular {
            cells: (0..2)
                .map(|index| RectangularChannelCell {
                    id: format!("cell-{index}"),
                    length_m: 1.0,
                    width_m: 1.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 1.0,
                    manning_n: 0.0,
                })
                .collect(),
        },
        lateral_bindings: vec![],
        config: ChannelConfig::default(),
    };
    let reach = ChannelReach::<CpuRuntime>::new(
        client,
        definition,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![1e-100, 0.0],
            longitudinal_momentum_m4_per_s: vec![0.0; 2],
        },
    )?;
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 1e-200,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: vec![],
    };
    let fields = reach
        .context
        .fields(&reach.accepted.volume.high, &reach.accepted.momentum.high)?;
    let flux = native_flux(&reach.context, &fields, &forcing.left, &forcing.right)?;
    let original_rate = flux.water.narrow(0, 1, 1)?.item::<f64>()?;
    assert!(original_rate > 0.0);
    assert_eq!(original_rate * forcing.end_s, 0.0);
    let original = reach.snapshot_data()?;
    let accepted = reach.step(forcing.end_s, &forcing)?.next;
    assert_eq!(accepted.state()?, reach.state()?);
    let receipt = accepted.last_step()?.ok_or("missing accepted interval")?;
    assert!(receipt.face_transfer_underflow_bound_m3[1] > 0.0);
    assert!(receipt.transfer_underflow_bound_m3 > 0.0);
    assert!(
        accepted.ledger().numerical_transfer_underflow_bound_m3
            >= receipt.transfer_underflow_bound_m3
    );
    assert_eq!(reach.snapshot_data()?, original);
    let mut tight_definition = reach.definition().clone();
    tight_definition.config.volume_absolute = f64::from_bits(1);
    tight_definition.config.volume_relative = 0.0;
    let tight = ChannelReach::<CpuRuntime>::new(
        reach.context.client.clone(),
        tight_definition,
        reach.context.original_initial.clone(),
    )?;
    let before = tight.snapshot_data()?;
    assert!(matches!(
        tight.step(forcing.end_s, &forcing),
        Err(ChannelError::Numerical { .. })
    ));
    assert_eq!(tight.snapshot_data()?, before);
    Ok(())
}

#[test]
fn positive_channel_motion_cannot_disappear_in_per_width_reconstruction()
-> Result<(), Box<dyn std::error::Error>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let reach = ChannelReach::<CpuRuntime>::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "wide shallow moving-water arithmetic fixture".into(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "wide-cell".into(),
                    length_m: 1.0,
                    width_m: 1e200,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 1.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![],
            config: ChannelConfig::default(),
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![1.0],
            longitudinal_momentum_m4_per_s: vec![1e-200],
        },
    )?;
    let original = reach.snapshot_data()?;
    let state = reach.state()?;
    assert!(state[0].depth_m > 0.0);
    assert!(state[0].velocity_m_per_s.is_some_and(|u| u > 0.0));
    assert!(state[0].discharge_m3_per_s > 0.0);
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 0.1,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: vec![],
    };
    assert!(matches!(
        reach.step(forcing.end_s, &forcing),
        Err(ChannelError::Numerical { .. })
    ));
    assert_eq!(reach.snapshot_data()?, original);
    Ok(())
}

#[test]
fn rounded_pressure_impulses_consume_the_configured_momentum_tolerance()
-> Result<(), Box<dyn std::error::Error>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let reach = ChannelReach::<CpuRuntime>::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "tiny finite pressure impulse arithmetic fixture".into(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: (0..2)
                    .map(|index| RectangularChannelCell {
                        id: format!("cell-{index}"),
                        length_m: 1.0,
                        width_m: 1.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 1.0,
                        manning_n: 0.0,
                    })
                    .collect(),
            },
            lateral_bindings: vec![],
            config: ChannelConfig::default(),
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![1e-100, 0.0],
            longitudinal_momentum_m4_per_s: vec![0.0; 2],
        },
    )?;
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 1e-150,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: vec![],
    };
    let fields = reach
        .context
        .fields(&reach.accepted.volume.high, &reach.accepted.momentum.high)?;
    let flux = native_flux(&reach.context, &fields, &forcing.left, &forcing.right)?;
    assert!(flux.water.narrow(0, 1, 1)?.item::<f64>()? * forcing.end_s > 0.0);
    let original_pressure = flux.momentum.narrow(0, 1, 1)?.item::<f64>()?;
    assert!(original_pressure > 0.0);
    assert_eq!(original_pressure * forcing.end_s, 0.0);
    let original = reach.snapshot_data()?;
    let accepted = reach.step(forcing.end_s, &forcing)?.next;
    assert!(accepted.state()?[1].volume_m3 > 0.0);
    let receipt = accepted.last_step()?.ok_or("missing accepted interval")?;
    assert!(receipt.max_momentum_underflow_bound_m4_per_s > 0.0);
    assert!(receipt.max_momentum_underflow_bound_m4_per_s <= reach.config().momentum_absolute);
    assert!(
        receipt
            .momentum_underflow_bound_m4_per_s
            .iter()
            .any(|bound| *bound > 0.0)
    );
    assert_eq!(reach.snapshot_data()?, original);
    let mut tight_definition = reach.definition().clone();
    tight_definition.config.momentum_absolute = f64::from_bits(1);
    tight_definition.config.momentum_relative = 0.0;
    let tight = ChannelReach::<CpuRuntime>::new(
        reach.context.client.clone(),
        tight_definition,
        reach.context.original_initial.clone(),
    )?;
    let before = tight.snapshot_data()?;
    assert!(matches!(
        tight.step(forcing.end_s, &forcing),
        Err(ChannelError::Numerical { .. })
    ));
    assert_eq!(tight.snapshot_data()?, before);
    Ok(())
}

#[test]
fn positive_wave_speed_cannot_disappear_during_width_weighting()
-> Result<(), Box<dyn std::error::Error>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let reach = ChannelReach::<CpuRuntime>::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "narrow deep still-water CFL arithmetic fixture".into(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    width_m: 1e-300,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 1e201,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![],
            config: ChannelConfig {
                gravity: 1e-300,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![1e-100],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )?;
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 0.1,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: vec![],
    };
    let fields = reach
        .context
        .fields(&reach.accepted.volume.high, &reach.accepted.momentum.high)?;
    let flux = native_flux(&reach.context, &fields, &forcing.left, &forcing.right)?;
    let speed = flux.speed.narrow(0, 0, 1)?.item::<f64>()?;
    assert!(speed > 0.0);
    assert_eq!(
        speed
            * match &reach.definition().geometry {
                crate::channel::ChannelGeometry::Rectangular { cells } => cells[0].width_m,
                _ => unreachable!(),
            },
        0.0
    );
    let original = reach.snapshot_data()?;
    assert!(matches!(
        reach.step(forcing.end_s, &forcing),
        Err(ChannelError::Numerical { .. })
    ));
    assert_eq!(reach.snapshot_data()?, original);
    Ok(())
}

fn scalar_endpoint_reach() -> Result<ChannelReach<CpuRuntime>, ChannelError> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    ChannelReach::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "scalar endpoint assembly arithmetic fixture".into(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 10.0,
                    width_m: 3.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 2.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![],
            config: ChannelConfig {
                gravity: 1.0,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![30.0],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )
}

#[test]
fn prescribed_endpoint_rates_retain_longitudinal_bits_without_width_round_trip()
-> Result<(), Box<dyn std::error::Error>> {
    let reach = scalar_endpoint_reach()?;
    let fields = reach
        .context
        .fields(&reach.accepted.volume.high, &reach.accepted.momentum.high)?;
    let target: f64 = 0.48600000000000004;
    assert_ne!((target / 3.0 * 3.0).to_bits(), target.to_bits());
    for rate in [target, -target, 0.0, -0.0] {
        let boundary = ChannelBoundary::Discharge {
            discharge_m3_per_s: rate,
        };
        let flux = native_flux(&reach.context, &fields, &boundary, &boundary)?;
        assert!(flux.riemann.is_none());
        assert_eq!(
            flux.boundary_traces
                .as_ref()
                .ok_or("missing scalar traces")?
                .endpoints,
            vec![0, 1]
        );
        for endpoint in 0..2 {
            assert_eq!(
                flux.water.narrow(0, endpoint, 1)?.item::<f64>()?.to_bits(),
                rate.to_bits()
            );
        }
    }
    Ok(())
}

#[test]
fn prescribed_endpoint_transfer_rejects_disappearing_nonzero_volume()
-> Result<(), Box<dyn std::error::Error>> {
    let reach = scalar_endpoint_reach()?;
    let fields = reach
        .context
        .fields(&reach.accepted.volume.high, &reach.accepted.momentum.high)?;
    let boundary = ChannelBoundary::Discharge {
        discharge_m3_per_s: 1e-200,
    };
    let flux = native_flux(&reach.context, &fields, &boundary, &boundary)?;
    let before = reach.snapshot_data()?;
    assert!(matches!(
        reach.context.integrate_transport(&flux, 1e-200),
        Err(ChannelError::Numerical { .. })
    ));
    assert_eq!(reach.snapshot_data()?, before);
    Ok(())
}

#[test]
fn scalar_only_rectangular_faces_preserve_full_section_motion()
-> Result<(), Box<dyn std::error::Error>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let reach = ChannelReach::<CpuRuntime>::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "scalar-only full-section motion arithmetic fixture".into(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    width_m: 1e160,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 2e-160,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![],
            config: ChannelConfig {
                gravity: 1.0,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![1.0],
            longitudinal_momentum_m4_per_s: vec![1e-170],
        },
    )?;
    let before = reach.snapshot_data()?;
    let fields = reach
        .context
        .fields(&reach.accepted.volume.high, &reach.accepted.momentum.high)?;
    assert!(fields.discharge.item::<f64>()? > 0.0);
    assert_eq!(
        fields.depth.item::<f64>()? * fields.velocity.item::<f64>()?,
        0.0
    );
    let boundary = ChannelBoundary::Stage {
        elevation_m: 1e-160,
    };
    let flux = native_flux(&reach.context, &fields, &boundary, &boundary)?;
    assert!(flux.riemann.is_none());
    assert_eq!(
        flux.boundary_traces
            .as_ref()
            .ok_or("missing scalar traces")?
            .endpoints,
        vec![0, 1]
    );
    let traces = flux
        .boundary_traces
        .as_ref()
        .ok_or("missing scalar traces")?;
    let expected_momentum: f64 = 5e-161;
    for endpoint in 0..2 {
        let momentum = flux.momentum.narrow(0, endpoint, 1)?.item::<f64>()?;
        let width = traces
            .momentum_flux_width_m4_per_s2
            .narrow(0, endpoint, 1)?
            .item::<f64>()?;
        let allowance = width + 32.0 * f64::EPSILON * expected_momentum.abs();
        assert!(
            (momentum - expected_momentum).abs() <= allowance,
            "endpoint={endpoint}, momentum={momentum:e}, expected={expected_momentum:e}, width={width:e}, allowance={allowance:e}"
        );
    }
    assert!(matches!(
        native_flux(&reach.context, &fields, &boundary, &ChannelBoundary::Wall),
        Err(ChannelError::Numerical { .. })
    ));
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 0.01,
        left: boundary.clone(),
        right: boundary,
        lateral_inputs: vec![],
    };
    let accepted = reach.step(forcing.end_s, &forcing)?.next;
    assert!(accepted.time() > reach.time());
    assert!(
        accepted
            .last_step()?
            .ok_or("missing accepted interval")?
            .boundary_traces
            .iter()
            .all(Option::is_some)
    );
    assert_eq!(reach.snapshot_data()?, before);
    Ok(())
}

fn native_flux(
    context: &super::super::context::Context<CpuRuntime>,
    fields: &super::super::admission::Fields<CpuRuntime>,
    left: &ChannelBoundary,
    right: &ChannelBoundary,
) -> Result<super::super::flux::Flux<CpuRuntime>, ChannelError> {
    use crate::channel::ChannelTensorBoundary;
    context.flux(
        fields,
        &ChannelTensorBoundary::from_host(&context.client, left)?,
        &ChannelTensorBoundary::from_host(&context.client, right)?,
    )
}
