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
fn riemann_diagnostics_retain_every_projected_bit() {
    use crate::channel::ChannelRiemannDiagnostics;
    let original = ChannelRiemannDiagnostics {
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
