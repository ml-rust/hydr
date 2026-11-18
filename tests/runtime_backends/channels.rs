use super::{Result, runtime};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelForcingInterval, ChannelGeometry,
    ChannelInitialState, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
    ChannelReach, ChannelReachDefinition, RectangularChannelCell,
};
use hydr::forcing::WaterAmount;
use numr::dtype::DType;
use numr::ops::NextRepresentableOps;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, HypotOps, IndexingOps, MidpointOps,
    ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedMappedBisectionAlgorithms;
use solvr::optimize::roots::{
    BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
};

fn reach_definition(widths: &[f64], beds: &[f64]) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: "Synthetic runtime datum".into(),
        provenance: "Explicit rectangular hydraulic runtime fixture".into(),
        geometry: ChannelGeometry::Rectangular {
            cells: widths
                .iter()
                .zip(beds)
                .enumerate()
                .map(
                    |(index, (&width_m, &bed_elevation_m))| RectangularChannelCell {
                        id: format!("cell-{index}"),
                        length_m: 1.0,
                        width_m,
                        bed_elevation_m,
                        maximum_depth_m: 8.0,
                        manning_n: 0.0,
                    },
                )
                .collect(),
        },
        lateral_bindings: Vec::new(),
        config: ChannelConfig::default(),
    }
}

fn initial_state(
    definition: &ChannelReachDefinition,
    depths: &[f64],
    velocities: &[f64],
) -> ChannelInitialState {
    let ChannelGeometry::Rectangular { cells } = &definition.geometry else {
        unreachable!()
    };
    let volume_m3: Vec<_> = cells
        .iter()
        .zip(depths)
        .map(|(cell, depth)| cell.width_m * cell.length_m * depth)
        .collect();
    let longitudinal_momentum_m4_per_s = volume_m3
        .iter()
        .zip(velocities)
        .map(|(volume, velocity)| volume * velocity)
        .collect();
    ChannelInitialState {
        time_s: 0.0,
        volume_m3,
        longitudinal_momentum_m4_per_s,
    }
}

fn forcing(start_s: f64, end_s: f64) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s,
        end_s,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: Vec::new(),
    }
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11,
        "actual={actual}, expected={expected}"
    );
}

fn channel_execution<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
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
        let definition = reach_definition(&[2.0; 4], &[0.0; 4]);
        let initial = initial_state(&definition, &[1.0; 4], &[velocity; 4]);
        let reach = ChannelReach::<R>::new(client.clone(), definition, initial)?;
        let original = reach.snapshot_data()?;
        let exterior = ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: velocity,
        };
        let interval = ChannelForcingInterval {
            left: exterior.clone(),
            right: exterior,
            ..forcing(0.0, 0.01)
        };
        let next = reach.step(0.01, &interval)?.next;
        assert_eq!(reach.snapshot_data()?, original);
        assert_eq!(next.time(), 0.01);
        for cell in next.state()? {
            near(cell.depth_m, 1.0);
            near(cell.discharge_m3_per_s, 2.0 * velocity);
            near(
                cell.velocity_m_per_s.ok_or("Missing wet velocity")?,
                velocity,
            );
        }
        let receipt = next.last_step()?.ok_or("Missing uniform-flow receipt")?;
        near(receipt.boundary_in_m3, 0.02);
        near(receipt.boundary_out_m3, 0.02);
        near(receipt.residual_m3, 0.0);
    }

    let definition = reach_definition(&[2.0, 0.5, 3.0, 1.0], &[0.0, 0.5, 2.0, 0.0]);
    let depths = [1.5, 1.0, 0.0, 1.5];
    let initial = initial_state(&definition, &depths, &[0.0; 4]);
    let reach = ChannelReach::<R>::new(client.clone(), definition, initial.clone())?;
    let next = reach.step(0.01, &forcing(0.0, 0.01))?.next;
    assert_eq!(next.time(), 0.01);
    for ((cell, depth), volume) in next.state()?.iter().zip(depths).zip(initial.volume_m3) {
        near(cell.depth_m, depth);
        near(cell.volume_m3, volume);
        near(cell.longitudinal_momentum_m4_per_s, 0.0);
    }
    assert!(next.state()?[2].dry);
    near(next.ledger().boundary_in_m3, 0.0);
    near(next.ledger().boundary_out_m3, 0.0);

    let definition = reach_definition(&[0.01, 100.0], &[0.0; 2]);
    let initial = initial_state(&definition, &[2.0, 0.5], &[0.0; 2]);
    let reach = ChannelReach::<R>::new(client.clone(), definition, initial.clone())?;
    let next = reach.step(0.01, &forcing(0.0, 0.01))?.next;
    let receipt = next.last_step()?.ok_or("Missing width-ratio receipt")?;
    assert!(next.time() > 0.0 && next.time() < 0.01);
    assert!(receipt.cfl_limit_s.is_finite());
    assert!(next.time() <= receipt.cfl_limit_s);
    assert!(next.time() <= receipt.depletion_limit_s);
    assert_eq!(receipt.face_volume_m3.len(), 3);
    assert_eq!(receipt.face_volume_m3[0], 0.0);
    assert_eq!(receipt.face_volume_m3[2], 0.0);
    assert!(receipt.face_volume_m3[1] > 0.0);
    assert!(receipt.face_volume_m3[1] <= initial.volume_m3[0]);
    for cell in next.state()? {
        assert!(cell.volume_m3 >= 0.0);
        assert!(cell.longitudinal_momentum_m4_per_s.is_finite());
    }
    near(next.ledger().storage_m3, initial.volume_m3.iter().sum());
    near(next.ledger().residual_m3, 0.0);

    let mut definition = reach_definition(&[2.0; 4], &[0.0; 4]);
    let ChannelGeometry::Rectangular { cells } = &mut definition.geometry else {
        unreachable!()
    };
    for cell in cells {
        cell.manning_n = 0.03;
    }
    let initial = initial_state(&definition, &[1.0, 0.5, 0.25, 0.75], &[0.2, -0.1, 0.0, 0.1]);
    let destination = ChannelReach::<R>::new(client.clone(), definition, initial)?;
    let first = destination.step(0.005, &forcing(0.0, 0.005))?.next;
    assert_eq!(first.time(), 0.005);
    let checkpoint = first.snapshot();
    let checkpoint_data = first.snapshot_data()?;
    let interval = forcing(0.005, 0.01);
    let continued = first.step(0.01, &interval)?.next;
    assert_eq!(continued.time(), 0.01);
    let continued_data = continued.snapshot_data()?;
    let trusted = destination.restore(&checkpoint)?;
    let cold = destination.import_snapshot(&checkpoint_data)?;
    assert_eq!(trusted.snapshot_data()?, checkpoint_data);
    assert_eq!(cold.snapshot_data()?, checkpoint_data);
    assert_eq!(
        trusted.step(0.01, &interval)?.next.snapshot_data()?,
        continued_data
    );
    assert_eq!(
        cold.step(0.01, &interval)?.next.snapshot_data()?,
        continued_data
    );
    assert_eq!(first.snapshot_data()?, checkpoint_data);

    let mut definition = reach_definition(&[1.0; 2], &[0.0; 2]);
    let initial = initial_state(&definition, &[1e-100, 0.0], &[0.0; 2]);
    let reach = ChannelReach::<R>::new(client.clone(), definition.clone(), initial.clone())?;
    let original = reach.snapshot_data()?;
    let next = reach.step(1e-200, &forcing(0.0, 1e-200))?.next;
    assert_eq!(next.time(), 1e-200);
    let receipt = next
        .last_step()?
        .ok_or("Missing bounded-underflow receipt")?;
    assert!(receipt.transfer_underflow_bound_m3 > 0.0);
    assert_eq!(receipt.face_transfer_underflow_bound_m3.len(), 3);
    assert!(receipt.face_transfer_underflow_bound_m3[1] > 0.0);
    assert!(next.ledger().numerical_transfer_underflow_bound_m3 > 0.0);
    assert!(
        next.ledger().numerical_transfer_underflow_bound_m3 >= receipt.transfer_underflow_bound_m3
    );
    assert_eq!(reach.snapshot_data()?, original);

    definition.config.volume_absolute = f64::from_bits(1);
    definition.config.volume_relative = 0.0;
    let reach = ChannelReach::<R>::new(client.clone(), definition, initial)?;
    let original = reach.snapshot_data()?;
    let error = reach
        .step(1e-200, &forcing(0.0, 1e-200))
        .err()
        .ok_or("Channel accepted a transfer-underflow charge exceeding its water tolerance")?;
    assert!(matches!(
        error,
        hydr::channel::ChannelError::Numerical { .. }
    ));
    assert_eq!(reach.snapshot_data()?, original);

    let definition = reach_definition(&[1e200], &[0.0]);
    let initial = initial_state(&definition, &[1e-200], &[1e-200]);
    assert_eq!(initial.volume_m3, vec![1.0]);
    assert_eq!(initial.longitudinal_momentum_m4_per_s, vec![1e-200]);
    let reach = ChannelReach::<R>::new(client.clone(), definition, initial)?;
    let original = reach.snapshot_data()?;
    let error = reach
        .step(1.0, &forcing(0.0, 1.0))
        .err()
        .ok_or("Channel accepted underflowed reconstructed discharge")?;
    assert!(matches!(
        error,
        hydr::channel::ChannelError::Numerical { .. }
    ));
    assert_eq!(reach.snapshot_data()?, original);
    lateral_execution::<R>(client)?;
    Ok(())
}

fn lateral_execution<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
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
    let binding = |id: &str, direction| ChannelLateralBinding {
        boundary_id: id.into(),
        cell_id: "cell-0".into(),
        direction,
        ownership_assertion: format!("Independent synthetic runtime source {id}"),
    };
    let input = |amount| ChannelLateralInput::Inflow {
        boundary_id: "in".into(),
        amount,
        longitudinal_velocity_m_per_s: -1.0,
    };
    let mut definition = reach_definition(&[2.0], &[0.0]);
    let ChannelGeometry::Rectangular { cells } = &mut definition.geometry else {
        unreachable!()
    };
    cells[0].length_m = 4.0;
    definition.lateral_bindings = vec![
        binding("in", ChannelLateralDirection::Inflow),
        binding("out", ChannelLateralDirection::Withdrawal),
    ];
    let initial = initial_state(&definition, &[1.0], &[3.0]);
    let destination = ChannelReach::<R>::new(client.clone(), definition, initial)?;
    let exterior = ChannelBoundary::Exterior {
        depth_m: 1.0,
        velocity_m_per_s: 3.0,
    };
    let interval = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 0.01,
        left: exterior.clone(),
        right: exterior,
        lateral_inputs: vec![
            input(WaterAmount::IntervalVolume { volume_m3: 2.0 }),
            ChannelLateralInput::Withdrawal {
                boundary_id: "out".into(),
                amount: WaterAmount::Rate {
                    rate_m3_per_s: 200.0,
                },
            },
        ],
    };
    let first = destination.step(0.01, &interval)?.next;
    let state = first.state()?;
    assert!((state[0].volume_m3 - 8.0).abs() <= 1e-12);
    assert!((state[0].longitudinal_momentum_m4_per_s - 16.0).abs() <= 1e-12);
    let receipt = first
        .last_step()?
        .ok_or("Missing lateral runtime receipt")?;
    assert_eq!(receipt.retries, 0);
    assert_eq!(receipt.lateral_receipts.len(), 2);
    assert_eq!(
        receipt.lateral_receipts[0].input,
        interval.lateral_inputs[0]
    );
    let checkpoint = first.snapshot();
    let data = first.snapshot_data()?;
    let continuation = ChannelForcingInterval {
        start_s: 0.01,
        end_s: 0.02,
        ..interval.clone()
    };
    let expected = first.step(0.02, &continuation)?.next.snapshot_data()?;
    assert_eq!(
        destination
            .restore(&checkpoint)?
            .step(0.02, &continuation)?
            .next
            .snapshot_data()?,
        expected
    );
    assert_eq!(
        destination
            .import_snapshot(&data)?
            .step(0.02, &continuation)?
            .next
            .snapshot_data()?,
        expected
    );
    assert_eq!(first.snapshot_data()?, data);

    let mut definition = reach_definition(&[2.0], &[0.0]);
    let ChannelGeometry::Rectangular { cells } = &mut definition.geometry else {
        unreachable!()
    };
    cells[0].maximum_depth_m = 10.0;
    definition.lateral_bindings = vec![binding("in", ChannelLateralDirection::Inflow)];
    let initial = initial_state(&definition, &[0.0], &[0.0]);
    let dry = ChannelReach::<R>::new(client, definition, initial)?;
    let before = dry.snapshot_data()?;
    let injection = |amount| ChannelForcingInterval {
        lateral_inputs: vec![ChannelLateralInput::Inflow {
            boundary_id: "in".into(),
            amount,
            longitudinal_velocity_m_per_s: 20.0,
        }],
        ..forcing(0.0, 1.0)
    };
    let next = dry
        .step(1.0, &injection(WaterAmount::Rate { rate_m3_per_s: 2.0 }))?
        .next;
    // The injected 20 m/s stream runs into the wall. The rightward signal speed at the left face is
    // at least 20 m/s, so the cell frequency is at least 10 /s. The CFL limit stays below
    // cfl / 10 = 0.045 s, far under max_step = 1.0 s.
    assert!(next.time() > 0.0 && next.time() < 0.045);
    let receipt = next
        .last_step()?
        .ok_or("Missing candidate-CFL runtime receipt")?;
    assert_eq!(receipt.accepted_start_cfl_limit_s, f64::INFINITY);
    let candidate_limit = receipt
        .candidate_cfl_limit_s
        .ok_or("Missing source-created wave limit")?;
    assert!(candidate_limit < 0.045 && next.time() <= candidate_limit);
    let error = dry
        .step(
            1.0,
            &injection(WaterAmount::IntervalVolume { volume_m3: 2.0 }),
        )
        .err()
        .ok_or("Exact runtime injection silently shortened")?;
    assert!(matches!(error, ChannelError::RecomputeInterval { .. }));
    assert_eq!(dry.snapshot_data()?, before);
    Ok(())
}

#[test]
fn selected_cpu_client_advances_rectangular_channels_and_complete_restart() -> Result {
    channel_execution::<runtime::Runtime>(runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_advances_rectangular_channels_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    channel_execution::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_client_rejects_channel_f64_without_fallback() -> Result {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

    let definition = reach_definition(&[2.0], &[0.0]);
    let initial = initial_state(&definition, &[1.0], &[0.0]);
    let error =
        ChannelReach::<WgpuRuntime>::new(WgpuClient::new(WgpuDevice::new(0))?, definition, initial)
            .err()
            .ok_or("WGPU accepted the unsupported channel F64 profile")?;
    assert!(matches!(
        error,
        hydr::channel::ChannelError::InvalidInput { .. }
    ));
    assert!(error.to_string().contains("F64"));
    Ok(())
}
