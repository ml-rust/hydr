use super::{Result, runtime};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelFluxPort, ChannelForcingInterval,
    ChannelGeometry, ChannelInitialState, ChannelReach, ChannelReachDefinition,
    RectangularChannelCell, TriangularChannelCell,
};
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

fn definition(triangle: bool) -> ChannelReachDefinition {
    let geometry = if triangle {
        ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: 0.0,
            cells: (0..4)
                .map(|i| TriangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 4.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        }
    } else {
        ChannelGeometry::Rectangular {
            cells: (0..4)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 4.0,
                    width_m: 7.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        }
    };
    ChannelReachDefinition {
        vertical_datum: "Synthetic runtime datum".into(),
        provenance: "Explicit nonuniform scalar boundary runtime fixture".into(),
        geometry,
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            gravity: if triangle { 2.0 } else { 1.0 },
            ..ChannelConfig::default()
        },
    }
}

fn initial(triangle: bool) -> ChannelInitialState {
    let depths = [1.0, 1.03125, 1.0625, 1.09375];
    let volume_m3: Vec<_> = depths
        .iter()
        .map(|h| 4.0 * if triangle { h * h } else { 7.0 * h })
        .collect();
    let longitudinal_momentum_m4_per_s = volume_m3
        .iter()
        .zip([0.0, 0.015625, 0.03125, 0.046875])
        .map(|(v, u)| v * u)
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
        left: ChannelBoundary::Stage { elevation_m: 1.0 },
        right: ChannelBoundary::Discharge {
            discharge_m3_per_s: 0.03,
        },
        lateral_inputs: Vec::new(),
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
    for triangle in [false, true] {
        let origin =
            ChannelReach::<R>::new(client.clone(), definition(triangle), initial(triangle))?;
        let before = origin.snapshot_data()?;
        let first_end = 1.0 / 1024.0;
        let second_end = 2.0 / 1024.0;
        let current = origin.step(first_end, &forcing(0.0, first_end))?.next;
        assert_eq!(current.time(), first_end);
        let saved = current.snapshot_data()?;
        let trusted = origin.restore(&current.snapshot())?;
        let cold = origin.import_snapshot(&saved)?;
        assert_eq!(trusted.snapshot_data()?, saved);
        assert_eq!(cold.snapshot_data()?, saved);
        let continued = current
            .step(second_end, &forcing(first_end, second_end))?
            .next;
        assert_eq!(continued.time(), second_end);
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
        assert_eq!(current.snapshot_data()?, saved);
        assert_eq!(origin.snapshot_data()?, before);
        for accepted in [&current, &continued] {
            let state = accepted.state()?;
            assert_eq!(state.len(), 4);
            for cell in state {
                assert!(cell.volume_m3.is_finite() && cell.volume_m3 > 0.0);
                assert!(cell.depth_m.is_finite() && cell.depth_m > 0.0);
                assert!(cell.longitudinal_momentum_m4_per_s.is_finite());
            }
            let receipt = accepted
                .last_step()?
                .ok_or("Missing runtime scalar receipt")?;
            let dt = receipt.end_s - receipt.start_s;
            assert_eq!(receipt.face_volume_m3[4].to_bits(), (0.03 * dt).to_bits());
            assert!(dt <= receipt.cfl_limit_s && dt <= receipt.accepted_start_cfl_limit_s);
            assert!(
                dt <= receipt
                    .candidate_cfl_limit_s
                    .ok_or("Missing final scalar runtime CFL")?
            );
            for trace in &receipt.boundary_traces {
                let trace = trace
                    .as_ref()
                    .ok_or("Missing actual runtime scalar trace")?;
                assert!(trace.depth_m.is_finite() && trace.depth_m > 0.0);
                assert!(trace.outward_velocity_m_per_s.is_finite());
                assert!(trace.velocity_compatibility_abs_bound_m_per_s.is_finite());
                assert!(trace.water_compatibility_abs_bound_m3_per_s.is_finite());
                assert!(trace.momentum_compatibility_abs_bound_m4_per_s2.is_finite());
                assert!(trace.root_iterations <= accepted.config().root_iterations);
            }
            let discharge = receipt.boundary_traces[1]
                .as_ref()
                .ok_or("Missing runtime prescribed discharge")?;
            assert_eq!(
                discharge.outward_discharge_m3_per_s.to_bits(),
                0.03_f64.to_bits()
            );
            assert_eq!(
                discharge.water_flux_width_m3_per_s.to_bits(),
                0.0_f64.to_bits()
            );
            let data = accepted.snapshot_data()?;
            let ChannelBoundary::Discharge { discharge_m3_per_s } = &data
                .last_interval
                .as_ref()
                .ok_or("Missing runtime scalar interval")?
                .forcing
                .right
            else {
                return Err("Runtime persistence changes discharge policy".into());
            };
            assert_eq!(discharge_m3_per_s.to_bits(), 0.03_f64.to_bits());
            let ledger = accepted.ledger();
            let scale = ledger.initial_storage_m3.max(ledger.storage_m3).max(1.0);
            assert!(
                ledger.residual_m3.abs()
                    + ledger.numerical_transfer_underflow_bound_m3
                    + ledger.lateral_counter_abs_bound_m3
                    + ledger.lateral_assembly_abs_bound_m3
                    <= accepted.config().volume_absolute
                        + accepted.config().volume_relative * scale
            );
        }
    }
    Ok(())
}

/// Ports that supply the exact flux of uniform flow at unit depth and Froude number 0.5.
fn uniform_ports(triangle: bool) -> std::result::Result<[ChannelBoundary; 2], ChannelError> {
    let definition = definition(triangle);
    let gravity = definition.config.gravity;
    let h = 1.0_f64;
    // The triangle has unit bank slopes: A = h², c = sqrt(g h / 2), p = g h³ / 3, φ = 4c.
    let (area, sound, pressure, phi_factor) = if triangle {
        (
            h * h,
            (gravity * h / 2.0).sqrt(),
            gravity * h * h * h / 3.0,
            4.0,
        )
    } else {
        (
            7.0 * h,
            (gravity * h).sqrt(),
            gravity * 7.0 * h * h / 2.0,
            2.0,
        )
    };
    let u = 0.5 * sound;
    let q = area * u;
    let momentum = q * u + pressure;
    let speed = u + sound;
    let phi = phi_factor * sound;
    let port = |water, invariant| {
        ChannelFluxPort::new(
            water,
            momentum,
            speed,
            Some(invariant),
            "runtime port fixture",
        )
        .map(ChannelBoundary::FluxPort)
    };
    Ok([port(-q, u + phi)?, port(q, phi - u)?])
}

/// Advances uniform flow between flux ports, restarts it, and returns every accepted state.
fn execute_ports<R>(client: R::Client) -> Result<Vec<String>>
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
    let mut states = Vec::new();
    for triangle in [false, true] {
        let [left, right] = uniform_ports(triangle)?;
        let forcing = |start_s, end_s| ChannelForcingInterval {
            start_s,
            end_s,
            left: left.clone(),
            right: right.clone(),
            lateral_inputs: Vec::new(),
        };
        let mut layout = definition(triangle);
        layout.provenance = "Explicit uniform flux port runtime fixture".into();
        // Unit depth: the triangle holds 1 m² and the 7 m rectangle 7 m² in 4 m cells.
        let area = if triangle { 1.0 } else { 7.0 };
        let gravity = layout.config.gravity;
        let sound = if triangle {
            (gravity / 2.0).sqrt()
        } else {
            gravity.sqrt()
        };
        let volume = 4.0 * area;
        let initial = ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![volume; 4],
            longitudinal_momentum_m4_per_s: vec![volume * 0.5 * sound; 4],
        };
        let origin = ChannelReach::<R>::new(client.clone(), layout, initial)?;
        let (first, second) = (1.0 / 64.0, 2.0 / 64.0);
        let current = origin.step(first, &forcing(0.0, first))?.next;
        let saved = current.snapshot_data()?;
        let cold = origin.import_snapshot(&saved)?;
        assert_eq!(cold.snapshot_data()?, saved);
        let continued = current.step(second, &forcing(first, second))?.next;
        assert_eq!(
            cold.step(second, &forcing(first, second))?
                .next
                .snapshot_data()?,
            continued.snapshot_data()?
        );
        let receipt = continued
            .last_step()?
            .ok_or("Missing runtime flux port receipt")?;
        assert_eq!(receipt.boundary_traces, [None, None]);
        let dt = receipt.end_s - receipt.start_s;
        let ChannelBoundary::FluxPort(port) = &right else {
            return Err("Runtime fixture lost its port".into());
        };
        assert_eq!(
            receipt.face_volume_m3[4].to_bits(),
            (port.outward_water_rate_m3_per_s() * dt).to_bits()
        );
        for accepted in [&current, &continued] {
            states.push(format!("{:?}", accepted.state()?));
        }
    }
    Ok(states)
}

#[test]
fn selected_cpu_executes_flux_port_endpoints_and_complete_restart() -> Result {
    execute_ports::<runtime::Runtime>(runtime::client()).map(|_| ())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_executes_flux_port_endpoints_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = execute_ports::<runtime::Runtime>(runtime::client())?;
    let cuda = execute_ports::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)?;
    super::same_bits(&cpu, &cuda)
}

#[test]
fn selected_cpu_executes_nonuniform_scalar_boundaries_and_complete_restart() -> Result {
    execute::<runtime::Runtime>(runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_executes_nonuniform_scalar_boundaries_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    execute::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_scalar_channel_f64_without_fallback() -> Result {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    let error = ChannelReach::<WgpuRuntime>::new(
        WgpuClient::new(WgpuDevice::new(0))?,
        definition(true),
        initial(true),
    )
    .err()
    .ok_or("WGPU accepts unsupported scalar channel F64")?;
    assert!(matches!(error, ChannelError::InvalidInput { .. }));
    assert!(error.to_string().contains("F64"));
    Ok(())
}
