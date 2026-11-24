//! Integration tests for Track B (2nd-order MUSCL reconstruction + Subgrid terrain tables)
//! and Track C (coastal wind stress, Coriolis rotation, and inverse-barometer forcing on F64 Solver).

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "support/impervious.rs"]
mod impervious;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, CoastalForcing, ForcingInterval, ForcingSchedule,
    SolverConfig, SubgridCellCurve, SubgridTerrain, SurfaceF32SpatialOrder, SurfaceSolverF32,
    SurfaceSolverF32Config,
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
    let domain = Domain::new(grid, bed, vec![0.0; n], vec![true; n], reference)?;
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
fn muscl_second_order_preserves_lake_at_rest_and_sharpens_dambreak() -> TestResult {
    // 1. Lake-at-rest over stepped bed stays exact with SecondOrderMuscl.
    let bed = vec![
        0.0, 0.4, 0.8, 0.2, 0.0, 0.3, 0.9, 0.5, 0.1, 0.0, 0.2, 0.6, 0.4, 0.1, 0.0, 0.0,
    ];
    let depths: Vec<f64> = bed.iter().map(|&z| 2.0 - z).collect();
    let (domain, flow) = make_basin(4, 4, bed, depths)?;
    let mut lake = SurfaceSolverF32::<Runtime>::new(
        runtime::client(),
        domain,
        flow,
        SurfaceSolverF32Config {
            spatial_order: SurfaceF32SpatialOrder::SecondOrderMuscl,
            ..SurfaceSolverF32Config::default()
        },
    )?;
    let z16 = vec![0.0_f32; 16];
    lake.advance_to(8.0, &z16, &z16, &z16, 5_000)?;
    for cell in lake.state()? {
        assert!(
            cell.qx.abs() < 1e-5 && cell.qy.abs() < 1e-5,
            "MUSCL lake-at-rest drift too large: qx={}, qy={}",
            cell.qx,
            cell.qy
        );
    }

    // 2. 1D dam-break along x (20x1 grid): MUSCL 2nd-order produces a sharper front (larger max |dh/dx|).
    let nx = 20;
    let init_h: Vec<f64> = (0..nx).map(|i| if i < 10 { 1.5 } else { 0.5 }).collect();
    let (d1, f1) = make_basin(nx, 1, vec![0.0; nx], init_h)?;
    let z20 = vec![0.0_f32; nx];

    let mut s1 = SurfaceSolverF32::<Runtime>::new(
        runtime::client(),
        d1.clone(),
        f1.clone(),
        SurfaceSolverF32Config {
            spatial_order: SurfaceF32SpatialOrder::FirstOrder,
            max_step_s: 0.25,
            ..SurfaceSolverF32Config::default()
        },
    )?;
    let mut s2 = SurfaceSolverF32::<Runtime>::new(
        runtime::client(),
        d1,
        f1,
        SurfaceSolverF32Config {
            spatial_order: SurfaceF32SpatialOrder::SecondOrderMuscl,
            max_step_s: 0.25,
            ..SurfaceSolverF32Config::default()
        },
    )?;
    s1.advance_to(10.0, &z20, &z20, &z20, 5_000)?;
    s2.advance_to(10.0, &z20, &z20, &z20, 5_000)?;

    assert!(s2.ledger().residual_m3.abs() < 1e-2);
    let max_grad = |cells: &[CellState]| -> f64 {
        cells
            .windows(2)
            .map(|w| (w[1].h - w[0].h).abs())
            .fold(0.0_f64, f64::max)
    };
    let g1 = max_grad(&s1.state()?);
    let g2 = max_grad(&s2.state()?);
    assert!(
        g2 > g1,
        "expected MUSCL 2nd-order front gradient ({g2}) > 1st-order ({g1})"
    );
    Ok(())
}

#[test]
fn subgrid_terrain_inverts_stage_volume_and_wet_fraction_exactly() -> TestResult {
    // 4 sub-cells inside a 100 m² coarse cell (25 m² each) with beds [0.0, 1.0, 2.0, 3.0].
    let curve = SubgridCellCurve::from_subcell_dem(&[2.0, 0.0, 3.0, 1.0], 100.0)?;
    assert_eq!(curve.min_bed_m(), 0.0);
    assert_eq!(curve.max_bed_m(), 3.0);

    // At stage = 1.5 m: sub-cell 0 has depth 1.5, sub-cell 1 has depth 0.5 -> V = 25*(1.5 + 0.5) = 50 m³.
    let v = curve.volume_at_stage(1.5);
    assert!((v - 50.0).abs() < 1e-12);
    assert!((curve.stage_at_volume(v) - 1.5).abs() < 1e-12);
    assert!((curve.wet_fraction_at_stage(1.5) - 0.5).abs() < 1e-12);

    // Above max bed (stage = 4.0 m): V = 25*(4 + 3 + 2 + 1) = 250 m³, wet_fraction = 1.0.
    assert!((curve.volume_at_stage(4.0) - 250.0).abs() < 1e-12);
    assert!((curve.stage_at_volume(250.0) - 4.0).abs() < 1e-12);
    assert_eq!(curve.wet_fraction_at_stage(4.0), 1.0);

    // Grid-wide 2x1 coarse grid from 4x2 fine DEM.
    let coarse = CartesianGrid::new(2, 1, 10.0, 10.0)?;
    let fine_bed = vec![0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0];
    let terrain = SubgridTerrain::from_fine_dem(coarse, 4, 2, &fine_bed)?;
    let eff_depths = terrain.effective_depth_field(&[1.0, 2.5]);
    let stages_back = terrain.stage_field(&eff_depths);
    assert!((stages_back[0] - 1.0).abs() < 1e-12);
    assert!((stages_back[1] - 2.5).abs() < 1e-12);
    let wets = terrain.wet_fraction_field(&eff_depths);
    assert!((wets[0] - 0.5).abs() < 1e-12);
    assert!((wets[1] - 1.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn f64_solver_coastal_wind_coriolis_and_inverse_barometer() -> TestResult {
    // Closed 4x4 frictionless flat basin (depth = 2.0 m) under eastward wind + Northern Hemisphere Coriolis.
    let (domain, flow) = make_basin(4, 4, vec![0.0; 16], vec![2.0; 16])?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 20.0,
            potential_evaporation: vec![0.0; 16],
            rain: vec![0.0; 16],
            withdrawal: vec![0.0; 16],
            boundaries: Vec::new(),
        }],
    )?;
    let mut solver = impervious_solver(
        domain.clone(),
        flow.clone(),
        boundaries.clone(),
        forcing.clone(),
        SolverConfig {
            gravity: 9.81,
            max_step: 0.5,
            ..SolverConfig::default()
        },
        0.0,
    )?
    .with_coastal_forcing(CoastalForcing {
        wind_x_m_per_s: 20.0,
        wind_y_m_per_s: 0.0,
        coriolis_f_per_s: 1e-3,
        ..CoastalForcing::default()
    })?;

    assert_eq!(
        solver.advance_to(10.0, 5_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    let center = solver.state()?.cells()[5];
    assert!(
        center.qx > 1e-3 && center.qy < -1e-6,
        "F64 Solver expected eastward wind + southward Coriolis deflection, got qx={}, qy={}",
        center.qx,
        center.qy
    );
    assert!(solver.ledger().residual.abs() < 1e-9);

    // Inverse barometer: positive dP/dx pushes water westward (qx < 0).
    let mut baro = impervious_solver(
        domain,
        flow,
        boundaries,
        forcing,
        SolverConfig {
            gravity: 9.81,
            max_step: 0.5,
            ..SolverConfig::default()
        },
        0.0,
    )?
    .with_coastal_forcing(CoastalForcing {
        dp_dx_pa_per_m: 10.0,
        ..CoastalForcing::default()
    })?;
    assert_eq!(
        baro.advance_to(5.0, 5_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    let b_center = baro.state()?.cells()[5];
    assert!(
        b_center.qx < -1e-2,
        "F64 Solver expected westward inverse-barometer acceleration, got qx={}",
        b_center.qx
    );
    Ok(())
}
