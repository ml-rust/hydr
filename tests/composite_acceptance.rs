//! §7 Composite acceptance: rainfall-soil-surface-drainage refinement, 2D terrain, and restart.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

use hydr::coupling::{
    SurfaceDrainageBatch, SurfaceDrainageConfig, SurfaceDrainageInlet, SurfaceDrainageSetup,
};
use hydr::drainage::{
    SwmmControlAuthority, SwmmDrainage, SwmmDrainageDefinition, SwmmSurchargeMethod,
};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandCell, LandModel, ModifiedGreenAmptParameters, SoilState};
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, ExteriorBoundary,
    ExteriorState, ForcingInterval, ForcingSchedule, InitialState, Solver, SolverConfig,
    SolverSetup,
};
use runtime::Runtime;
use std::path::PathBuf;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const PROVENANCE: &str = "EPA SWMM v5.2.4 commit 7952ca837988b1c32f791812eccc9fd64547e093";
const SURCHARGE_EVIDENCE: &str = "Composite §7 acceptance under EXTRAN";

fn worker_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resource/reference-engines/swmm/build-drainage-worker/swmm-drainage-worker")
}

fn composite_drainage_inp() -> String {
    r#"[OPTIONS]
FLOW_UNITS           CFS
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
END_DATE             01/01/2025
END_TIME             01:00:00
ROUTING_STEP         00:00:01
MINIMUM_STEP         0.25
VARIABLE_STEP        0.75
SURCHARGE_METHOD     EXTRAN
[JUNCTIONS]
J1  8.0  10.0  0.0  10.0  0.0
J2  6.0  10.0  0.0  10.0  0.0
[OUTFALLS]
OUT1  4.0  FREE  NO
[CONDUITS]
C1  J1  J2    200.0  0.013  0.0  0.0  0.0  0.0
C2  J2  OUT1  200.0  0.013  0.0  0.0  0.0  0.0
[ORIFICES]
OR1  J1  J2  SIDE  0.5  0.65  NO  0.05
[XSECTIONS]
C1   CIRCULAR  2.0  0  0  0  1
C2   CIRCULAR  2.0  0  0  0  1
OR1  CIRCULAR  1.0  0  0  0
[CONTROLS]
RULE R_OR1
IF NODE J1 DEPTH > 0.8
THEN ORIFICE OR1 SETTING = 1.0
ELSE ORIFICE OR1 SETTING = 0.5
"#
    .to_string()
}

fn open_drainage() -> Result<SwmmDrainage, Box<dyn std::error::Error>> {
    let bin = worker_binary();
    assert!(
        bin.is_file(),
        "pinned SWMM worker missing at {}",
        bin.display()
    );
    let def = SwmmDrainageDefinition::new(
        composite_drainage_inp(),
        "NAVD88",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::NativeSwmm,
        bin,
        PROVENANCE,
        SURCHARGE_EVIDENCE,
    )?;
    Ok(SwmmDrainage::open(def)?)
}

/// Builds a symmetric 2D V-shaped urban valley (`90 m x 90 m`) with Modified Green-Ampt pervious
/// hillslopes, impervious central drainage trough, and constant rainfall.
fn build_valley_surface(
    n_side: usize,
    max_step: f64,
    rain_m_per_s: f64,
    end_s: f64,
) -> Result<(Solver<Runtime>, usize, f64), Box<dyn std::error::Error>> {
    assert!(
        n_side % 2 == 1,
        "use odd grid dimension for exact center cell"
    );
    let length = 90.0;
    let dx = length / n_side as f64;
    let grid = CartesianGrid::new(n_side, n_side, dx, dx)?;
    let count = grid.cell_count();
    let mid = n_side / 2;
    let center_cell = mid * n_side + mid;

    let mut bed = Vec::with_capacity(count);
    let mut land_cells = Vec::with_capacity(count);
    let soil_params = ModifiedGreenAmptParameters::from_native(0.35, 2.0e-5, 0.25)?;
    for row in 0..n_side {
        let y = (row as f64 + 0.5) * dx;
        for col in 0..n_side {
            let x = (col as f64 + 0.5) * dx;
            let z = 4.0 + 0.01 * (x - 45.0).abs() + 0.004 * (y - 45.0).abs();
            bed.push(z);
            if col == mid {
                land_cells.push(LandCell::Impervious);
            } else {
                land_cells.push(LandCell::ModifiedGreenAmpt(soil_params));
            }
        }
    }
    let center_bed = bed[center_cell];

    let reference = SpatialReference::new(
        "synthetic projected metres".into(),
        "NAVD88".into(),
        [0.0, 0.0],
    )?;
    let domain = Domain::new(grid, bed, vec![0.035; count], vec![true; count], reference)?;
    let land = LandModel::new(&domain, land_cells)?;
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 0.0,
                qx: 0.0,
                qy: 0.0,
            };
            count
        ],
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: end_s + 1.0,
            potential_evaporation: vec![0.0; count],
            rain: vec![rain_m_per_s; count],
            withdrawal: vec![0.0; count],
            boundaries: Vec::new(),
        }],
    )?;
    let setup = SolverSetup {
        domain,
        land,
        boundaries,
        forcing,
        config: SolverConfig {
            gravity: 9.81,
            max_step,
            ..SolverConfig::default()
        },
    };
    let initial = InitialState {
        time: 0.0,
        flow,
        soil,
    };
    let solver = Solver::new(runtime::client(), setup, initial)?;
    Ok((solver, center_cell, center_bed))
}

struct ValleyMetrics {
    surface_storage_m3: f64,
    infiltration_m3: f64,
    drainage_capture_m3: f64,
    peak_center_depth_m: f64,
    max_symmetry_error_m: f64,
}

fn valley_setup(center_cell: usize, center_bed: f64, max_step: f64) -> SurfaceDrainageSetup {
    SurfaceDrainageSetup {
        vertical_datum: "NAVD88".into(),
        config: SurfaceDrainageConfig {
            max_step_s: max_step,
            min_step_s: 0.05,
            max_retries: 8,
            max_iterations: 20,
            head_tolerance_m: 1e-4,
            volume_tolerance_m3: 1e-4,
            gravity: 9.81,
        },
        inlets: vec![SurfaceDrainageInlet {
            id: "IN_VALLEY".into(),
            surface_cell: center_cell,
            drainage_node_id: "J1".into(),
            rim_elevation_m: center_bed,
            weir_crest_length_m: 3.0,
            weir_cd: 1.7,
            orifice_area_m2: 0.4,
            orifice_cd: 0.62,
            allow_surcharge_to_surface: true,
        }],
        control_claims: Vec::new(),
    }
}

fn run_valley_composite(
    n_side: usize,
    max_step: f64,
    end_s: f64,
) -> Result<ValleyMetrics, Box<dyn std::error::Error>> {
    let rain_m_per_s = 5.0e-4;
    let (surface, center_cell, center_bed) =
        build_valley_surface(n_side, max_step, rain_m_per_s, end_s)?;
    let drainage = open_drainage()?;
    let setup = valley_setup(center_cell, center_bed, max_step);
    let mut batch = SurfaceDrainageBatch::new(surface, drainage, setup)?;
    let mut peak_center_depth_m = 0.0_f64;
    while batch.time_s() < end_s - 1e-12 {
        let trial = batch.trial_to(end_s)?;
        trial.accept();
        let h_center = batch.surface().state()?.cells()[center_cell].h;
        peak_center_depth_m = peak_center_depth_m.max(h_center);
    }

    let area = batch.surface().domain().grid().cell_area();
    let cells = batch.surface().state()?.cells().to_vec();
    let surface_storage_m3: f64 = cells.iter().map(|c| c.h * area).sum();
    let ledger = batch.surface().ledger();
    let infiltration_m3 = ledger.infiltration;
    let drainage_capture_m3 = batch.cumulative_inlet_transfers_m3()[0];

    // Check exact composite water balance: rain = surface_storage + infiltration + drainage_capture.
    let expected_rain_m3 = rain_m_per_s * 90.0 * 90.0 * end_s;
    let accounted_m3 = surface_storage_m3 + infiltration_m3 + drainage_capture_m3;
    assert!(
        (accounted_m3 - expected_rain_m3).abs() < 1e-5,
        "composite balance mismatch at {n_side}x{n_side}, dt={max_step}: accounted={accounted_m3}, rain={expected_rain_m3}"
    );
    assert!(
        infiltration_m3 > 0.1 && drainage_capture_m3 > 0.1,
        "both soil infiltration ({infiltration_m3} m³) and drainage capture ({drainage_capture_m3} m³) must be active"
    );

    // Genuine 2D transverse symmetry: h(row, col) == h(row, n_side - 1 - col).
    let mut max_symmetry_error_m = 0.0_f64;
    for row in 0..n_side {
        for col in 0..n_side {
            let mirror = n_side - 1 - col;
            let diff = (cells[row * n_side + col].h - cells[row * n_side + mirror].h).abs();
            max_symmetry_error_m = max_symmetry_error_m.max(diff);
        }
    }

    Ok(ValleyMetrics {
        surface_storage_m3,
        infiltration_m3,
        drainage_capture_m3,
        peak_center_depth_m,
        max_symmetry_error_m,
    })
}

#[test]
fn composite_rainfall_soil_surface_drainage_refines_in_space_and_time() -> TestResult {
    if !worker_binary().is_file() {
        return Ok(());
    }
    // Independent spatial refinement (3x3 -> 5x5 -> 9x9) at fixed max_step = 2.0 s.
    let m3 = run_valley_composite(3, 2.0, 30.0)?;
    let m5 = run_valley_composite(5, 2.0, 30.0)?;
    let m9 = run_valley_composite(9, 2.0, 30.0)?;
    for m in [&m3, &m5, &m9] {
        assert!(
            m.max_symmetry_error_m < 1e-12,
            "2D transverse valley symmetry violated: {}",
            m.max_symmetry_error_m
        );
        assert!(m.peak_center_depth_m > 0.005 && m.infiltration_m3 > 0.1);
    }
    let coarse_storage_diff = (m5.surface_storage_m3 - m3.surface_storage_m3).abs();
    let fine_storage_diff = (m9.surface_storage_m3 - m5.surface_storage_m3).abs();
    assert!(
        fine_storage_diff < coarse_storage_diff,
        "spatial refinement must reduce storage sensitivity: coarse={coarse_storage_diff:.4}, fine={fine_storage_diff:.4}"
    );

    // Independent timestep refinement (max_step = 4.0 -> 2.0 -> 1.0 s) on fixed 5x5 mesh.
    let dt4 = run_valley_composite(5, 4.0, 30.0)?;
    let dt2 = m5;
    let dt1 = run_valley_composite(5, 1.0, 30.0)?;
    let coarse_dt_diff = (dt2.drainage_capture_m3 - dt4.drainage_capture_m3).abs();
    let fine_dt_diff = (dt1.drainage_capture_m3 - dt2.drainage_capture_m3).abs();
    assert!(
        fine_dt_diff < coarse_dt_diff,
        "timestep refinement must reduce capture sensitivity: coarse={coarse_dt_diff:.6}, fine={fine_dt_diff:.6}"
    );
    Ok(())
}

/// Runs a genuine 2D non-separable terrain with rainfall pulse (0..30 s) and dry recession (30..90 s).
fn run_2d_terrain_recession(
    max_step: f64,
) -> Result<(f64, f64, f64, f64), Box<dyn std::error::Error>> {
    let n = 10usize;
    let dx = 10.0;
    let grid = CartesianGrid::new(n, n, dx, dx)?;
    let count = grid.cell_count();
    let soil_params = ModifiedGreenAmptParameters::from_native(0.3, 1.5e-5, 0.2)?;
    let mut bed = Vec::with_capacity(count);
    for row in 0..n {
        let y = (row as f64 + 0.5) * dx;
        for col in 0..n {
            let x = (col as f64 + 0.5) * dx;
            let z = 2.0
                + 0.008 * ((x - 50.0).powi(2) / 100.0 + (y - 50.0).powi(2) / 100.0)
                + 0.12
                    * (std::f64::consts::PI * x / 100.0).sin()
                    * (std::f64::consts::PI * y / 100.0).sin();
            bed.push(z);
        }
    }
    let reference = SpatialReference::new(
        "synthetic projected metres".into(),
        "NAVD88".into(),
        [0.0, 0.0],
    )?;
    let domain = Domain::new(grid, bed, vec![0.03; count], vec![true; count], reference)?;
    let land = LandModel::new(
        &domain,
        vec![LandCell::ModifiedGreenAmpt(soil_params); count],
    )?;
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 0.0,
                qx: 0.0,
                qy: 0.0,
            };
            count
        ],
    )?;
    // Open low-elevation exterior boundary along the right edge (column = n).
    let open_faces: Vec<BoundaryFace> = (0..n)
        .map(|row| BoundaryFace {
            axis: Axis::X,
            column: n,
            row,
        })
        .collect();
    let boundaries = BoundaryConditions::with_open_faces(&domain, open_faces.clone())?;
    let ext_boundaries: Vec<_> = open_faces
        .iter()
        .map(|&face| {
            BoundaryForcing::Exterior(ExteriorBoundary {
                face,
                state: ExteriorState::Dry,
            })
        })
        .collect();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![
            ForcingInterval {
                start: 0.0,
                end: 30.0,
                potential_evaporation: vec![0.0; count],
                rain: vec![4.0e-4; count],
                withdrawal: vec![0.0; count],
                boundaries: ext_boundaries.clone(),
            },
            ForcingInterval {
                start: 30.0,
                end: 90.0,
                potential_evaporation: vec![0.0; count],
                rain: vec![0.0; count],
                withdrawal: vec![0.0; count],
                boundaries: ext_boundaries,
            },
        ],
    )?;
    let mut solver = Solver::<Runtime>::new(
        runtime::client(),
        SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config: SolverConfig {
                gravity: 9.81,
                max_step,
                ..SolverConfig::default()
            },
        },
        InitialState {
            time: 0.0,
            flow,
            soil,
        },
    )?;

    assert_eq!(
        solver.advance_to(30.0, 100_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    let area = solver.domain().grid().cell_area();
    let cells_30 = solver.state()?.cells().to_vec();
    let storage_30: f64 = cells_30.iter().map(|c| c.h * area).sum();
    let max_ux = cells_30
        .iter()
        .filter(|c| c.h > 1e-6)
        .map(|c| (c.qx / c.h).abs())
        .fold(0.0_f64, f64::max);
    let max_uy = cells_30
        .iter()
        .filter(|c| c.h > 1e-6)
        .map(|c| (c.qy / c.h).abs())
        .fold(0.0_f64, f64::max);

    assert_eq!(
        solver.advance_to(90.0, 100_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    let storage_90: f64 = solver.state()?.cells().iter().map(|c| c.h * area).sum();
    assert!(
        solver.ledger().residual.abs() < 1e-8,
        "2D terrain mass balance residual too large: {}",
        solver.ledger().residual
    );
    Ok((storage_30, storage_90, max_ux, max_uy))
}

#[test]
fn composite_2d_terrain_transverse_wave_and_recession_characterization() -> TestResult {
    let (s30_2, s90_2, ux_2, uy_2) = run_2d_terrain_recession(2.0)?;
    let (s30_1, s90_1, ux_1, uy_1) = run_2d_terrain_recession(1.0)?;

    // Genuine 2D flow: both x and y velocities are strictly positive.
    assert!(ux_2 > 1e-3 && uy_2 > 1e-3 && ux_1 > 1e-3 && uy_1 > 1e-3);
    // Arrival & recession: storage peaks at 30 s and recedes by 90 s.
    assert!(s30_1 > 10.0 && s90_1 < s30_1);
    // Timestep refinement consistency across max_step = 2 s vs 1 s.
    assert!(
        (s30_2 - s30_1).abs() / s30_1 < 0.05 && (s90_2 - s90_1).abs() / s90_1 < 0.05,
        "2D terrain recession timestep sensitivity exceeded 5%"
    );
    Ok(())
}

#[test]
fn composite_surface_drainage_with_soil_and_controls_restarts_bit_identically() -> TestResult {
    if !worker_binary().is_file() {
        return Ok(());
    }
    let make_batch = || -> Result<SurfaceDrainageBatch<Runtime>, Box<dyn std::error::Error>> {
        let (surface, center_cell, center_bed) = build_valley_surface(5, 2.0, 5.0e-4, 40.0)?;
        let drainage = open_drainage()?;
        let setup = valley_setup(center_cell, center_bed, 2.0);
        Ok(SurfaceDrainageBatch::new(surface, drainage, setup)?)
    };

    let mut reference = make_batch()?;
    for target in [15.0, 30.0] {
        while reference.time_s() < target - 1e-12 {
            let trial = reference.trial_to(target)?;
            trial.accept();
        }
    }

    let mut split = make_batch()?;
    while split.time_s() < 15.0 - 1e-12 {
        let trial = split.trial_to(15.0)?;
        trial.accept();
    }
    let snap = split.snapshot_data()?;

    let mut restarted = make_batch()?;
    restarted.import_snapshot(&snap)?;
    while restarted.time_s() < 30.0 - 1e-12 {
        let trial = restarted.trial_to(30.0)?;
        trial.accept();
    }

    assert_eq!(
        restarted.surface().state()?.cells(),
        reference.surface().state()?.cells()
    );
    assert_eq!(restarted.drainage().state(), reference.drainage().state());
    assert_eq!(
        restarted.cumulative_inlet_transfers_m3(),
        reference.cumulative_inlet_transfers_m3()
    );
    Ok(())
}
