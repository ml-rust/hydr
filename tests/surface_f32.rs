//! Tests for the opt-in `F32` 2D shallow-water solver (`SurfaceSolverF32`) on CPU, CUDA, and WGPU.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "support/impervious.rs"]
mod impervious;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, ForcingInterval, ForcingSchedule, SolverConfig,
    SurfaceSolverF32, SurfaceSolverF32Config,
};
use impervious::impervious_solver;
use runtime::Runtime;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn make_basin(
    nx: usize,
    ny: usize,
    bed: Vec<f64>,
    depths: Vec<f64>,
) -> Result<(Domain, FlowState), Box<dyn std::error::Error>> {
    let grid = CartesianGrid::new(nx, ny, 10.0, 10.0)?;
    let n = grid.cell_count();
    let reference = SpatialReference::new("metric".into(), "NAVD88".into(), [0.0, 0.0])?;
    let domain = Domain::new(grid, bed, vec![0.025; n], vec![true; n], reference)?;
    let cells = depths
        .into_iter()
        .map(|h| CellState {
            h,
            qx: 0.0,
            qy: 0.0,
        })
        .collect();
    let flow = FlowState::new(&domain, cells)?;
    Ok((domain, flow))
}

#[test]
fn surface_solver_f32_cpu_lake_at_rest_rain_and_dambreak() -> TestResult {
    // 1. Well-balanced lake-at-rest across stepped bed (WSE = 2.0 m).
    let bed = vec![
        0.0, 0.4, 0.8, 0.2, 0.0, 0.3, 0.9, 0.5, 0.1, 0.0, 0.2, 0.6, 0.4, 0.1, 0.0, 0.0,
    ];
    let depths: Vec<f64> = bed.iter().map(|&z| 2.0 - z).collect();
    let (domain, flow) = make_basin(4, 4, bed, depths)?;
    let mut lake = SurfaceSolverF32::<Runtime>::new(
        runtime::client(),
        domain,
        flow,
        SurfaceSolverF32Config::default(),
    )?;
    let zeros = vec![0.0_f32; 16];
    lake.advance_to(10.0, &zeros, &zeros, &zeros, 5_000)?;
    for cell in lake.state()? {
        assert!(
            cell.qx.abs() < 1e-5 && cell.qy.abs() < 1e-5,
            "F32 lake-at-rest velocity drift too large: qx={}, qy={}",
            cell.qx,
            cell.qy
        );
    }

    // 2. Rain-on-basin mass conservation + bit-identical F32 snapshot/restore.
    let (domain_r, flow_r) = make_basin(4, 4, vec![0.0; 16], vec![0.0; 16])?;
    let mut rain_ref = SurfaceSolverF32::<Runtime>::new(
        runtime::client(),
        domain_r.clone(),
        flow_r.clone(),
        SurfaceSolverF32Config::default(),
    )?;
    let rain = vec![1e-3_f32; 16];
    rain_ref.advance_to(10.0, &rain, &zeros, &zeros, 5_000)?;
    let snap = rain_ref.snapshot_data()?;
    rain_ref.advance_to(20.0, &rain, &zeros, &zeros, 5_000)?;
    assert!(
        rain_ref.ledger().residual_m3.abs() < 1e-3,
        "F32 rain mass residual too large: {}",
        rain_ref.ledger().residual_m3
    );

    let mut rain_restored = SurfaceSolverF32::<Runtime>::new(
        runtime::client(),
        domain_r,
        flow_r,
        SurfaceSolverF32Config::default(),
    )?;
    rain_restored.restore_data(&snap)?;
    rain_restored.advance_to(20.0, &rain, &zeros, &zeros, 5_000)?;
    assert_eq!(rain_restored.state()?, rain_ref.state()?);

    // 3. 2D dam-break: compare F32 solver against certified F64 `Solver`.
    let mut init_h = vec![0.2; 25];
    for row in 0..5 {
        for col in 0..2 {
            init_h[row * 5 + col] = 1.0;
        }
    }
    let (domain_d, flow_d) = make_basin(5, 5, vec![0.0; 25], init_h)?;
    let mut f32_dam = SurfaceSolverF32::<Runtime>::new(
        runtime::client(),
        domain_d.clone(),
        flow_d.clone(),
        SurfaceSolverF32Config {
            max_step_s: 0.5,
            ..SurfaceSolverF32Config::default()
        },
    )?;
    let z25 = vec![0.0_f32; 25];
    f32_dam.advance_to(5.0, &z25, &z25, &z25, 10_000)?;

    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain_d,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 6.0,
            potential_evaporation: vec![0.0; 25],
            rain: vec![0.0; 25],
            withdrawal: vec![0.0; 25],
            boundaries: Vec::new(),
        }],
    )?;
    let mut f64_dam = impervious_solver(
        domain_d,
        flow_d,
        boundaries,
        forcing,
        SolverConfig {
            gravity: 9.81,
            max_step: 0.5,
            ..SolverConfig::default()
        },
        0.0,
    )?;
    assert_eq!(
        f64_dam.advance_to(5.0, 10_000)?.status,
        AdvanceStatus::ReachedTarget
    );

    let max_depth_diff = f32_dam
        .state()?
        .iter()
        .zip(f64_dam.state()?.cells())
        .map(|(a, b)| (a.h - b.h).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_depth_diff < 5e-3,
        "F32 dam-break diverged from F64 reference: max |dh| = {max_depth_diff}"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_surface_solver_f32() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    let mut init_h = vec![0.2; 25];
    for row in 0..5 {
        for col in 0..2 {
            init_h[row * 5 + col] = 1.0;
        }
    }
    let (domain, flow) = make_basin(5, 5, vec![0.0; 25], init_h)?;
    let cfg = SurfaceSolverF32Config {
        max_step_s: 0.5,
        ..SurfaceSolverF32Config::default()
    };
    let z = vec![0.0_f32; 25];

    let mut cpu =
        SurfaceSolverF32::<Runtime>::new(runtime::client(), domain.clone(), flow.clone(), cfg)?;
    cpu.advance_to(5.0, &z, &z, &z, 10_000)?;

    let cuda_client = CudaClient::new(CudaDevice::new(0))?;
    let mut cuda = SurfaceSolverF32::<CudaRuntime>::new(cuda_client, domain, flow, cfg)?;
    cuda.advance_to(5.0, &z, &z, &z, 10_000)?;

    let max_diff = cpu
        .state()?
        .iter()
        .zip(cuda.state()?)
        .map(|(a, b)| (a.h - b.h).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-5,
        "CUDA F32 diverged from CPU F32: max |dh| = {max_diff}"
    );
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_client_executes_surface_solver_f32() -> TestResult {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

    let Ok(wgpu_client) = WgpuClient::new(WgpuDevice::new(0)) else {
        eprintln!("skipping WGPU F32 execution: no WebGPU adapter available on host");
        return Ok(());
    };

    let mut init_h = vec![0.2; 25];
    for row in 0..5 {
        for col in 0..2 {
            init_h[row * 5 + col] = 1.0;
        }
    }
    let (domain, flow) = make_basin(5, 5, vec![0.0; 25], init_h)?;
    let cfg = SurfaceSolverF32Config {
        max_step_s: 0.5,
        ..SurfaceSolverF32Config::default()
    };
    let z = vec![0.0_f32; 25];

    let mut cpu =
        SurfaceSolverF32::<Runtime>::new(runtime::client(), domain.clone(), flow.clone(), cfg)?;
    cpu.advance_to(5.0, &z, &z, &z, 10_000)?;

    let mut wgpu = SurfaceSolverF32::<WgpuRuntime>::new(wgpu_client, domain, flow, cfg)?;
    wgpu.advance_to(5.0, &z, &z, &z, 10_000)?;

    let max_diff = cpu
        .state()?
        .iter()
        .zip(wgpu.state()?)
        .map(|(a, b)| (a.h - b.h).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-5,
        "WGPU F32 diverged from CPU F32: max |dh| = {max_diff}"
    );
    Ok(())
}
