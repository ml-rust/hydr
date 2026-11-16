use crate::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelForcingInterval, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReach,
    ChannelReachDefinition, RectangularChannelCell,
};
use crate::forcing::WaterAmount;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use std::sync::Arc;

#[test]
fn restoration_retains_the_destination_configured_client() -> Result<(), ChannelError> {
    let client = CpuClient::new(CpuDevice::new());
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "exact rectangular client fixture".into(),
        lateral_bindings: vec![ChannelLateralBinding {
            boundary_id: "inlet".into(),
            cell_id: "cell".into(),
            direction: ChannelLateralDirection::Inflow,
            ownership_assertion: "independent exact client fixture input".into(),
        }],
        geometry: crate::channel::ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "cell".into(),
                length_m: 1.0,
                width_m: 2.0,
                bed_elevation_m: 0.0,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        },
        config: ChannelConfig::default(),
    };
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    let destination = ChannelReach::<CpuRuntime>::new(
        client
            .clone()
            .with_parallelism(ParallelismConfig::new(Some(1), Some(64))),
        definition.clone(),
        initial.clone(),
    )?;
    let source = ChannelReach::<CpuRuntime>::new(
        client.with_parallelism(ParallelismConfig::new(Some(1), Some(128))),
        definition,
        initial,
    )?
    .step(
        0.01,
        &ChannelForcingInterval {
            start_s: 0.0,
            end_s: 0.01,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: vec![ChannelLateralInput::Inflow {
                boundary_id: "inlet".into(),
                amount: WaterAmount::IntervalVolume { volume_m3: 0.001 },
                longitudinal_velocity_m_per_s: 0.0,
            }],
        },
    )?
    .next;
    let original_package = source.snapshot_data()?;
    let mut altered_config = original_package.clone();
    altered_config.definition.config.boundary.velocity_relative = 2e-12;
    assert!(destination.import_snapshot(&altered_config).is_err());
    let mut forged = original_package.clone();
    let interval = forged
        .last_interval
        .as_mut()
        .ok_or_else(|| ChannelError::input("test interval absent: retain the accepted step"))?;
    interval.diagnostics.boundary_traces[0] = Some(trace());
    assert!(destination.import_snapshot(&forged).is_err());
    let mut forged_geometry = original_package.clone();
    forged_geometry
        .last_interval
        .as_mut()
        .ok_or_else(|| ChannelError::input("test interval absent: retain the accepted step"))?
        .diagnostics
        .friction_geometry = Some(
        crate::channel::flow::output::ChannelFrictionGeometryDiagnostics {
            momentum_abs_bound_m4_per_s: vec![0.0],
            max_momentum_abs_bound_m4_per_s: 0.0,
        },
    );
    assert!(destination.import_snapshot(&forged_geometry).is_err());
    let context = Arc::clone(&destination.context);
    let trusted = destination.restore(&source.snapshot())?;
    let cold = destination.import_snapshot(&source.snapshot_data()?)?;
    for restored in [&trusted, &cold] {
        assert!(Arc::ptr_eq(&restored.context, &context));
        assert_eq!(restored.context.client.parallelism().chunk_size, Some(64));
        assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
    }
    Ok(())
}

#[test]
fn cold_restore_uploads_with_a_separately_constructed_cpu_client() -> Result<(), ChannelError> {
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "exact rectangular device fixture".into(),
        lateral_bindings: Vec::new(),
        geometry: crate::channel::ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "cell".into(),
                length_m: 1.0,
                width_m: 2.0,
                bed_elevation_m: 0.0,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        },
        config: ChannelConfig::default(),
    };
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    let source = ChannelReach::<CpuRuntime>::new(
        CpuClient::new(CpuDevice::new()),
        definition.clone(),
        initial.clone(),
    )?;
    let destination =
        ChannelReach::<CpuRuntime>::new(CpuClient::new(CpuDevice::new()), definition, initial)?;
    let context = Arc::clone(&destination.context);
    let restored = destination.import_snapshot(&source.snapshot_data()?)?;
    assert!(Arc::ptr_eq(&restored.context, &context));
    assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
    Ok(())
}

#[test]
fn riemann_diagnostics_retain_every_projected_bit() -> Result<(), ChannelError> {
    use crate::channel::ChannelRiemannDiagnostics;
    let original = ChannelRiemannDiagnostics {
        surveyed_work: None,
        water_flux_width_m3_per_s: vec![0.0, 1e-12],
        momentum_flux_width_m4_per_s2: vec![2e-12, 0.0],
        max_water_flux_width_m3_per_s: 1e-12,
        max_momentum_flux_width_m4_per_s2: 2e-12,
        root_iterations: 7,
    };
    assert!(super::check::riemann(None, None));
    assert!(!super::check::riemann(Some(&original), None));
    assert!(!super::check::riemann(None, Some(&original)));
    assert!(super::check::riemann(Some(&original), Some(&original)));
    for index in 0..5 {
        let mut changed = original.clone();
        match index {
            0 => changed.water_flux_width_m3_per_s[0] = -0.0,
            1 => changed.momentum_flux_width_m4_per_s2[1] = -0.0,
            2 => changed.max_water_flux_width_m3_per_s = 0.0,
            3 => changed.max_momentum_flux_width_m4_per_s2 = 0.0,
            _ => changed.root_iterations += 1,
        }
        assert!(!super::check::riemann(Some(&original), Some(&changed)));
    }
    let mut surveyed = original.clone();
    surveyed.surveyed_work = Some(crate::channel::PrismaticRiemannWork::default());
    assert!(!super::check::riemann(Some(&original), Some(&surveyed)));
    let work = surveyed.surveyed_work.as_mut().ok_or_else(|| {
        ChannelError::input("test surveyed work absent: retain the original diagnostics")
    })?;
    work.maximum_root_iterations = 7;
    for index in 0..12 {
        let mut changed = surveyed.clone();
        let work = changed.surveyed_work.as_mut().ok_or_else(|| {
            ChannelError::input("test surveyed work absent: retain the original diagnostics")
        })?;
        match index {
            0 => work.root_solves += 1,
            1 => work.root_iterations += 1,
            2 => work.maximum_root_iterations += 1,
            3 => work.residual_points += 1,
            4 => work.simpson_value_points += 1,
            5 => work.simpson_derivative_points += 1,
            6 => work.refinements += 1,
            7 => work.maximum_simpson_panels += 1,
            8 => work.bracket_expansion_rounds += 1,
            9 => work.maximum_bracket_expansion_rounds += 1,
            10 => work.bracket_solves += 1,
            _ => changed.surveyed_work = None,
        }
        assert!(!super::check::riemann(Some(&surveyed), Some(&changed)));
    }
    Ok(())
}

#[test]
fn friction_geometry_diagnostics_retain_every_projected_bit() {
    use crate::channel::flow::output::ChannelFrictionGeometryDiagnostics;
    let original = ChannelFrictionGeometryDiagnostics {
        momentum_abs_bound_m4_per_s: vec![0.0, 1e-12],
        max_momentum_abs_bound_m4_per_s: 1e-12,
    };
    assert!(super::check::friction_geometry(None, None));
    assert!(!super::check::friction_geometry(Some(&original), None));
    assert!(!super::check::friction_geometry(None, Some(&original)));
    assert!(super::check::friction_geometry(
        Some(&original),
        Some(&original)
    ));
    let mut changed = original.clone();
    changed.momentum_abs_bound_m4_per_s[0] = -0.0;
    assert!(!super::check::friction_geometry(
        Some(&original),
        Some(&changed)
    ));
    let mut changed = original.clone();
    changed.max_momentum_abs_bound_m4_per_s = 0.0;
    assert!(!super::check::friction_geometry(
        Some(&original),
        Some(&changed)
    ));
}

#[test]
fn triangular_restoration_rejects_a_different_reconstruction() -> Result<(), ChannelError> {
    use crate::channel::{ChannelGeometry, ChannelTriangularReconstruction, TriangularChannelCell};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "triangular reconstruction continuation".into(),
        geometry: ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: 0.0,
            cells: vec![TriangularChannelCell {
                id: "cell".into(),
                length_m: 2.0,
                maximum_depth_m: 3.0,
                manning_n: 0.0,
            }],
        },
        lateral_bindings: vec![],
        config: ChannelConfig::default(),
    };
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    let source =
        ChannelReach::<CpuRuntime>::new(client.clone(), definition.clone(), initial.clone())?;
    let mut changed = definition;
    changed.config.triangular_reconstruction = ChannelTriangularReconstruction::FirstOrderReference;
    assert!(!source.definition().same_bits(&changed));
    let destination = ChannelReach::<CpuRuntime>::new(client, changed, initial)?;
    assert!(destination.restore(&source.snapshot()).is_err());
    assert!(
        destination
            .import_snapshot(&source.snapshot_data()?)
            .is_err()
    );
    assert!(source.restore(&destination.snapshot()).is_err());
    assert!(
        source
            .import_snapshot(&destination.snapshot_data()?)
            .is_err()
    );
    Ok(())
}

fn trace() -> crate::channel::ChannelBoundaryTrace {
    crate::channel::ChannelBoundaryTrace {
        depth_m: 0.0,
        area_m2: 0.0,
        outward_velocity_m_per_s: 0.0,
        outward_discharge_m3_per_s: 0.0,
        momentum_flux_m4_per_s2: 0.0,
        signal_speed_bound_m_per_s: 0.0,
        velocity_compatibility_abs_bound_m_per_s: 0.0,
        water_compatibility_abs_bound_m3_per_s: 0.0,
        momentum_compatibility_abs_bound_m4_per_s2: 0.0,
        water_flux_width_m3_per_s: 0.0,
        momentum_flux_width_m4_per_s2: 0.0,
        cap_compatibility_approximation: false,
        reservoir: None,
        root_iterations: 3,
    }
}

#[test]
fn boundary_receipts_retain_every_field_and_optional_entry() {
    let original = trace();
    assert!(super::check::boundary_trace(None, None));
    assert!(super::check::boundary_trace(
        Some(&original),
        Some(&original)
    ));
    assert!(!super::check::boundary_trace(None, Some(&original)));
    assert!(!super::check::boundary_trace(Some(&original), None));
    for index in 0..13 {
        let mut changed = original.clone();
        match index {
            0 => changed.depth_m = -0.0,
            1 => changed.area_m2 = -0.0,
            2 => changed.outward_velocity_m_per_s = -0.0,
            3 => changed.outward_discharge_m3_per_s = -0.0,
            4 => changed.momentum_flux_m4_per_s2 = -0.0,
            5 => changed.signal_speed_bound_m_per_s = -0.0,
            6 => changed.velocity_compatibility_abs_bound_m_per_s = -0.0,
            7 => changed.water_compatibility_abs_bound_m3_per_s = -0.0,
            8 => changed.momentum_compatibility_abs_bound_m4_per_s2 = -0.0,
            9 => changed.water_flux_width_m3_per_s = -0.0,
            10 => changed.momentum_flux_width_m4_per_s2 = -0.0,
            11 => changed.cap_compatibility_approximation = true,
            _ => changed.root_iterations += 1,
        }
        assert!(!super::check::boundary_trace(
            Some(&original),
            Some(&changed)
        ));
    }
}

#[test]
fn scalar_forcing_identity_retains_policy_target_bits() {
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 1.0,
        left: ChannelBoundary::Stage { elevation_m: 0.0 },
        right: ChannelBoundary::Discharge {
            discharge_m3_per_s: 0.0,
        },
        lateral_inputs: vec![],
    };
    assert!(super::check::forcing(&forcing, &forcing));
    for index in 0..4 {
        let mut changed = forcing.clone();
        match index {
            0 => changed.left = ChannelBoundary::Stage { elevation_m: -0.0 },
            1 => {
                changed.right = ChannelBoundary::Discharge {
                    discharge_m3_per_s: -0.0,
                }
            }
            2 => {
                changed.left = ChannelBoundary::Discharge {
                    discharge_m3_per_s: 0.0,
                }
            }
            _ => changed.right = ChannelBoundary::Stage { elevation_m: 0.0 },
        }
        assert!(!super::check::forcing(&forcing, &changed));
    }
}

#[test]
fn reservoir_receipts_retain_each_bound_bit_and_policy_presence() -> Result<(), ChannelError> {
    use crate::channel::ChannelReservoirTrace;
    let mut original = trace();
    original.reservoir = Some(ChannelReservoirTrace {
        depth_lower_m: 0.0,
        depth_upper_m: 0.0,
        head_compatibility_abs_bound_m: 0.0,
        dissipation_head_lower_m: 0.0,
        dissipation_head_upper_m: 0.0,
    });
    assert!(original.same_bits(&original));
    assert!(!original.same_bits(&trace()));
    for index in 0..5 {
        let mut changed = original.clone();
        let receipt = changed.reservoir.as_mut().ok_or_else(|| {
            ChannelError::input("test reservoir receipt absent: retain the original certificate")
        })?;
        match index {
            0 => receipt.depth_lower_m = -0.0,
            1 => receipt.depth_upper_m = -0.0,
            2 => receipt.head_compatibility_abs_bound_m = -0.0,
            3 => receipt.dissipation_head_lower_m = -0.0,
            _ => receipt.dissipation_head_upper_m = -0.0,
        }
        assert!(!original.same_bits(&changed));
    }
    Ok(())
}

#[test]
fn reservoir_forcing_identity_retains_head_coefficient_and_policy_bits() {
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 1.0,
        left: ChannelBoundary::QuiescentReservoir {
            elevation_m: 0.0,
            entrance_loss_coefficient: 0.0,
        },
        right: ChannelBoundary::Wall,
        lateral_inputs: vec![],
    };
    assert!(super::check::forcing(&forcing, &forcing));
    for index in 0..3 {
        let mut changed = forcing.clone();
        changed.left = match index {
            0 => ChannelBoundary::QuiescentReservoir {
                elevation_m: -0.0,
                entrance_loss_coefficient: 0.0,
            },
            1 => ChannelBoundary::QuiescentReservoir {
                elevation_m: 0.0,
                entrance_loss_coefficient: -0.0,
            },
            _ => ChannelBoundary::Stage { elevation_m: 0.0 },
        };
        assert!(!super::check::forcing(&forcing, &changed));
    }
}
