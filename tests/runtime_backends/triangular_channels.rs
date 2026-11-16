use super::{Result, runtime};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelForcingInterval, ChannelGeometry,
    ChannelInitialState, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
    ChannelReach, ChannelReachDefinition, ChannelTriangularReconstruction, TriangularChannelCell,
};
use hydr::forcing::WaterAmount;
use numr::dtype::DType;
use numr::ops::NextRepresentableOps;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedMappedBisectionAlgorithms;
use solvr::optimize::roots::{
    BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
};

fn definition(sources: bool) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: "Synthetic runtime datum".into(),
        provenance: "Explicit triangular runtime geometry".into(),
        geometry: ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 0.0,
            right_bank_slope_h_per_v: 2.0,
            bed_elevation_m: 0.0,
            cells: vec![TriangularChannelCell {
                id: "cell".into(),
                length_m: 8.0,
                maximum_depth_m: 10.0,
                manning_n: 0.0,
            }],
        },
        lateral_bindings: if sources {
            [
                ("in", ChannelLateralDirection::Inflow),
                ("out", ChannelLateralDirection::Withdrawal),
            ]
            .into_iter()
            .map(|(id, direction)| ChannelLateralBinding {
                boundary_id: id.into(),
                cell_id: "cell".into(),
                direction,
                ownership_assertion: format!("Independent synthetic runtime source {id}"),
            })
            .collect()
        } else {
            Vec::new()
        },
        config: ChannelConfig::default(),
    }
}

fn initial(velocity: f64) -> ChannelInitialState {
    ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![8.0],
        longitudinal_momentum_m4_per_s: vec![8.0 * velocity],
    }
}

fn interval(start_s: f64, end_s: f64, velocity: f64, sources: bool) -> ChannelForcingInterval {
    let exterior = ChannelBoundary::Exterior {
        depth_m: 1.0,
        velocity_m_per_s: velocity,
    };
    ChannelForcingInterval {
        start_s,
        end_s,
        left: exterior.clone(),
        right: exterior,
        lateral_inputs: if sources {
            vec![
                ChannelLateralInput::Inflow {
                    boundary_id: "in".into(),
                    amount: WaterAmount::IntervalVolume { volume_m3: 2.0 },
                    longitudinal_velocity_m_per_s: -1.0,
                },
                ChannelLateralInput::Withdrawal {
                    boundary_id: "out".into(),
                    amount: WaterAmount::Rate {
                        rate_m3_per_s: 200.0,
                    },
                },
            ]
        } else {
            Vec::new()
        },
    }
}

fn execute<R>(client: R::Client) -> Result
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
        + NextRepresentableOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    for velocity in [-1.0, 1.0] {
        let origin = ChannelReach::<R>::new(client.clone(), definition(false), initial(velocity))?;
        let before = origin.snapshot_data()?;
        let next = origin
            .step(0.01, &interval(0.0, 0.01, velocity, false))?
            .next;
        assert_eq!(next.time(), 0.01);
        let state = next.state()?;
        assert!((state[0].depth_m - 1.0).abs() <= 1e-12);
        assert!((state[0].discharge_m3_per_s - velocity).abs() <= 1e-11);
        let receipt = next
            .last_step()?
            .ok_or("Missing triangular runtime receipt")?;
        assert!((receipt.boundary_in_m3 - 0.01).abs() <= 1e-12);
        assert!((receipt.boundary_out_m3 - 0.01).abs() <= 1e-12);
        assert_eq!(origin.snapshot_data()?, before);
    }
    let origin = ChannelReach::<R>::new(client.clone(), definition(true), initial(3.0))?;
    let forcing = interval(0.0, 0.01, 3.0, true);
    let current = origin.step(0.01, &forcing)?.next;
    let state = current.state()?;
    assert!((state[0].volume_m3 - 8.0).abs() <= 1e-12);
    assert!((state[0].longitudinal_momentum_m4_per_s - 16.0).abs() <= 1e-12);
    let receipt = current
        .last_step()?
        .ok_or("Missing triangular source receipt")?;
    assert_eq!(receipt.retries, 0);
    assert_eq!(receipt.lateral_receipts.len(), 2);
    assert_eq!(receipt.lateral_receipts[0].input, forcing.lateral_inputs[0]);
    assert_eq!(receipt.lateral_receipts[1].input, forcing.lateral_inputs[1]);
    let saved = current.snapshot_data()?;
    let trusted = origin.restore(&current.snapshot())?;
    let cold = origin.import_snapshot(&saved)?;
    assert_eq!(trusted.snapshot_data()?, saved);
    assert_eq!(cold.snapshot_data()?, saved);
    let continuation = interval(0.01, 0.02, 2.0, true);
    let expected = current.step(0.02, &continuation)?.next.snapshot_data()?;
    assert_eq!(
        trusted.step(0.02, &continuation)?.next.snapshot_data()?,
        expected
    );
    assert_eq!(
        cold.step(0.02, &continuation)?.next.snapshot_data()?,
        expected
    );
    assert_eq!(current.snapshot_data()?, saved);

    nonuniform_reconstruction::<R>(client.clone())?;

    let mut short_definition = definition(true);
    short_definition.config.max_step = 0.005;
    let short = ChannelReach::<R>::new(client, short_definition, initial(3.0))?;
    let before = short.snapshot_data()?;
    let error = short
        .step(0.01, &forcing)
        .err()
        .ok_or("Exact triangular runtime source shortened")?;
    assert!(matches!(error, ChannelError::RecomputeInterval { .. }));
    assert_eq!(short.snapshot_data()?, before);
    Ok(())
}

fn nonuniform_reconstruction<R>(client: R::Client) -> Result
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
        + NextRepresentableOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let mut layout = definition(false);
    layout.config.triangular_reconstruction = ChannelTriangularReconstruction::LimitedLinear;
    if let ChannelGeometry::PrismaticTriangular { cells, .. } = &mut layout.geometry {
        *cells = (0..5)
            .map(|index| TriangularChannelCell {
                id: format!("cell-{index}"),
                length_m: 4.0,
                maximum_depth_m: 10.0,
                manning_n: 0.0,
            })
            .collect();
    }
    // Both conserved fields increase through interior cells, giving nonzero limited slopes.
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0, 3.0, 4.0, 5.0, 6.0],
        longitudinal_momentum_m4_per_s: vec![0.25, 0.5, 0.75, 1.0, 1.25],
    };
    let mut reference_layout = layout.clone();
    reference_layout.config.triangular_reconstruction =
        ChannelTriangularReconstruction::FirstOrderReference;
    let reference = ChannelReach::<R>::new(client.clone(), reference_layout, initial.clone())?;
    let origin = ChannelReach::<R>::new(client, layout, initial)?;
    let unchanged = origin.snapshot_data()?;
    let first_end = 1.0 / 1024.0;
    let second_end = 2.0 / 1024.0;
    let forcing = |start_s, end_s| ChannelForcingInterval {
        start_s,
        end_s,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: Vec::new(),
    };
    let current = origin.step(first_end, &forcing(0.0, first_end))?.next;
    assert_eq!(current.time().to_bits(), first_end.to_bits());
    let reference = reference.step(first_end, &forcing(0.0, first_end))?.next;
    assert_ne!(current.state()?, reference.state()?);
    let saved = current.snapshot_data()?;
    let trusted = origin.restore(&current.snapshot())?;
    let cold = origin.import_snapshot(&saved)?;
    assert_eq!(trusted.snapshot_data()?, saved);
    assert_eq!(cold.snapshot_data()?, saved);
    let continued = current
        .step(second_end, &forcing(first_end, second_end))?
        .next;
    assert_eq!(continued.time().to_bits(), second_end.to_bits());
    let expected = continued.snapshot_data()?;
    assert_eq!(
        trusted
            .step(second_end, &forcing(first_end, second_end))?
            .next
            .snapshot_data()?,
        expected
    );
    assert_eq!(
        cold.step(second_end, &forcing(first_end, second_end))?
            .next
            .snapshot_data()?,
        expected
    );
    assert_eq!(origin.snapshot_data()?, unchanged);
    assert_eq!(current.snapshot_data()?, saved);
    for accepted in [&current, &continued] {
        let state = accepted.state()?;
        assert_eq!(state.len(), 5);
        for cell in &state {
            assert!(cell.volume_m3.is_finite() && cell.volume_m3 > 0.0);
            assert!(cell.depth_m.is_finite() && cell.depth_m > 0.0);
            assert!(cell.discharge_m3_per_s.is_finite());
            assert!(cell.longitudinal_momentum_m4_per_s.is_finite());
            assert!(!cell.dry);
        }
        let ledger = accepted.ledger();
        assert!((ledger.storage_m3 - 20.0).abs() <= 1e-9 + 20e-12);
        assert_eq!(ledger.boundary_in_m3, 0.0);
        assert_eq!(ledger.boundary_out_m3, 0.0);
        assert_eq!(ledger.lateral_in_m3, 0.0);
        assert_eq!(ledger.lateral_out_m3, 0.0);
        assert!(
            ledger.residual_m3.abs()
                + ledger.numerical_transfer_underflow_bound_m3
                + ledger.lateral_counter_abs_bound_m3
                + ledger.lateral_assembly_abs_bound_m3
                <= 1e-9 + 20e-12
        );
        let receipt = accepted
            .last_step()?
            .ok_or("Missing nonuniform runtime receipt")?;
        let dt = receipt.end_s - receipt.start_s;
        assert!(dt <= receipt.cfl_limit_s);
        assert!(dt <= receipt.accepted_start_cfl_limit_s);
        assert!(dt <= receipt.depletion_limit_s);
        assert_eq!(receipt.face_volume_m3[0], 0.0);
        assert_eq!(receipt.face_volume_m3[5], 0.0);
        let certificate = receipt
            .riemann
            .as_ref()
            .ok_or("Missing nonuniform flux certificate")?;
        for (widths, transfers, absolute) in [
            (
                &certificate.water_flux_width_m3_per_s,
                &receipt.face_volume_m3,
                accepted.config().riemann_water_flux_absolute_m3_per_s,
            ),
            (
                &certificate.momentum_flux_width_m4_per_s2,
                &receipt.face_momentum_m4_per_s,
                accepted.config().riemann_momentum_flux_absolute_m4_per_s2,
            ),
        ] {
            assert_eq!(widths.len(), 6);
            for (&width, &transfer) in widths.iter().zip(transfers) {
                assert!(width.is_finite() && width >= 0.0);
                let scale = (transfer / dt).abs() + width;
                assert!(width <= absolute + accepted.config().riemann_flux_relative * scale);
            }
        }
        assert_eq!(
            certificate.max_water_flux_width_m3_per_s,
            certificate
                .water_flux_width_m3_per_s
                .iter()
                .copied()
                .fold(0.0, f64::max)
        );
        assert_eq!(
            certificate.max_momentum_flux_width_m4_per_s2,
            certificate
                .momentum_flux_width_m4_per_s2
                .iter()
                .copied()
                .fold(0.0, f64::max)
        );
        assert!(certificate.root_iterations <= accepted.config().root_iterations);
    }
    Ok(())
}

#[test]
fn selected_cpu_client_advances_triangular_channels_sources_and_complete_restart() -> Result {
    execute::<runtime::Runtime>(runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_advances_triangular_channels_sources_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    execute::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_client_rejects_triangular_f64_without_fallback() -> Result {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    let error = ChannelReach::<WgpuRuntime>::new(
        WgpuClient::new(WgpuDevice::new(0))?,
        definition(false),
        initial(0.0),
    )
    .err()
    .ok_or("WGPU accepted triangular F64")?;
    assert!(matches!(error, ChannelError::InvalidInput { .. }));
    assert!(error.to_string().contains("F64"));
    Ok(())
}
