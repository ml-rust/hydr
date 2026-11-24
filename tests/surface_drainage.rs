//! Integration tests for conservative 2D surface / pinned SWMM v5.2.4 drainage coupling.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "support/impervious.rs"]
mod impervious;

use hydr::coupling::{
    DrainageControlClaim, SURFACE_DRAINAGE_CONTRACT_VERSION, SURFACE_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceDrainageBatch, SurfaceDrainageConfig, SurfaceDrainageError, SurfaceDrainageInlet,
    SurfaceDrainageSetup,
};
use hydr::drainage::{
    SwmmControlAuthority, SwmmDrainage, SwmmDrainageDefinition, SwmmSurchargeMethod,
};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{BoundaryConditions, ForcingInterval, ForcingSchedule, Solver, SolverConfig};
use impervious::impervious_solver;
use runtime::Runtime;
use std::path::PathBuf;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const PROVENANCE: &str = "EPA SWMM v5.2.4 commit 7952ca837988b1c32f791812eccc9fd64547e093";
const SURCHARGE_EVIDENCE: &str = "Coupled surface-drainage verification under EXTRAN";

fn worker_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resource/reference-engines/swmm/build-drainage-worker/swmm-drainage-worker")
}

/// 3x3 closed basin (10 m x 10 m cells, area = 100 m² each) with uniform bed and depth.
fn make_surface(
    bed_m: f64,
    depth_m: f64,
    end_s: f64,
) -> Result<Solver<Runtime>, Box<dyn std::error::Error>> {
    let grid = CartesianGrid::new(3, 3, 10.0, 10.0)?;
    let n = grid.cell_count();
    let reference = SpatialReference::new(
        "synthetic projected metres".into(),
        "NAVD88".into(),
        [0.0, 0.0],
    )?;
    let domain = Domain::new(
        grid,
        vec![bed_m; n],
        vec![0.03; n],
        vec![true; n],
        reference,
    )?;
    let cells = vec![
        CellState {
            h: depth_m,
            qx: 0.0,
            qy: 0.0,
        };
        n
    ];
    let state = FlowState::new(&domain, cells)?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: end_s,
            potential_evaporation: vec![0.0; n],
            rain: vec![0.0; n],
            withdrawal: vec![0.0; n],
            boundaries: Vec::new(),
        }],
    )?;
    let solver = impervious_solver(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig {
            gravity: 9.81,
            max_step: 10.0,
            ..SolverConfig::default()
        },
        0.0,
    )?;
    Ok(solver)
}

fn surface_total_volume_m3(solver: &Solver<Runtime>) -> Result<f64, Box<dyn std::error::Error>> {
    let area = solver.domain().grid().cell_area();
    let vol: f64 = solver.state()?.cells().iter().map(|c| c.h * area).sum();
    Ok(vol)
}

/// SWMM drainage network with invert J1 at 10.0 ft (3.048 m) and native control rule on OR1.
fn capture_network_inp() -> String {
    r#"[TITLE]
Surface-Drainage Capture Network

[OPTIONS]
FLOW_UNITS           CFS
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
REPORT_START_DATE    01/01/2025
REPORT_START_TIME    00:00:00
END_DATE             01/01/2025
END_TIME             01:00:00
ROUTING_STEP         00:00:02
MINIMUM_STEP         0.5
VARIABLE_STEP        0.75
SURCHARGE_METHOD     EXTRAN

[JUNCTIONS]
J1  10.0  8.0  0.0  10.0  0.0
J2   8.0  8.0  0.0  10.0  0.0

[OUTFALLS]
OUT1  6.0  FREE  NO

[CONDUITS]
C1  J1  J2    200.0  0.013  0.0  0.0  0.0  0.0
C2  J2  OUT1  200.0  0.013  0.0  0.0  0.0  0.0

[ORIFICES]
OR1  J1  J2  SIDE  0.5  0.65  NO  0.05
OR2  J1  J2  SIDE  0.5  0.65  NO  0.05

[XSECTIONS]
C1   CIRCULAR  2.0  0  0  0  1
C2   CIRCULAR  2.0  0  0  0  1
OR1  CIRCULAR  1.0  0  0  0
OR2  CIRCULAR  1.0  0  0  0

[CONTROLS]
RULE R_OR1
IF NODE J1 DEPTH > 1.0
THEN ORIFICE OR1 SETTING = 1.0
ELSE ORIFICE OR1 SETTING = 0.5
"#
    .to_string()
}

/// Pressurized surcharge network using Preissmann `SLOT`: J1 invert 10.0 ft (3.048 m),
/// fullDepth 3.123 ft (~4.0 m rim), surchargeDepth 15.0 ft, with 40 cfs inflow so H_d > 4.0 m.
fn surcharge_network_inp() -> String {
    r#"[TITLE]
Surface-Drainage Pressurized Surcharge Network

[OPTIONS]
FLOW_UNITS           CFS
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
REPORT_START_DATE    01/01/2025
REPORT_START_TIME    00:00:00
END_DATE             01/01/2025
END_TIME             01:00:00
ROUTING_STEP         00:00:02
MINIMUM_STEP         0.5
VARIABLE_STEP        0.75
SURCHARGE_METHOD     SLOT

[JUNCTIONS]
J1  10.0  3.123  0.0  15.0  0.0
J2   9.5  3.123  0.0  15.0  0.0

[OUTFALLS]
OUT1  9.0  FREE  NO

[CONDUITS]
C1  J1  J2    300.0  0.013  0.0  0.0  0.0  0.0
C2  J2  OUT1  300.0  0.013  0.0  0.0  0.0  0.0

[XSECTIONS]
C1  CIRCULAR  1.5  0  0  0  1
C2  CIRCULAR  1.5  0  0  0  1

[INFLOWS]
J1  FLOW  ""  FLOW  1.0  1.0  40.0
"#
    .to_string()
}

fn open_drainage(
    inp: String,
    method: SwmmSurchargeMethod,
) -> Result<SwmmDrainage, Box<dyn std::error::Error>> {
    let bin = worker_binary();
    assert!(
        bin.is_file(),
        "pinned SWMM worker missing at {}",
        bin.display()
    );
    let def = SwmmDrainageDefinition::new(
        inp,
        "NAVD88",
        method,
        SwmmControlAuthority::NativeSwmm,
        bin,
        PROVENANCE,
        SURCHARGE_EVIDENCE,
    )?;
    Ok(SwmmDrainage::open(def)?)
}

fn capture_setup(
    rim_elevation_m: f64,
    datum: &str,
    claims: Vec<DrainageControlClaim>,
) -> SurfaceDrainageSetup {
    SurfaceDrainageSetup {
        vertical_datum: datum.to_string(),
        config: SurfaceDrainageConfig {
            max_step_s: 5.0,
            min_step_s: 0.1,
            max_retries: 8,
            max_iterations: 20,
            head_tolerance_m: 1e-4,
            volume_tolerance_m3: 1e-4,
            gravity: 9.81,
        },
        inlets: vec![SurfaceDrainageInlet {
            id: "IN_CENTER".to_string(),
            surface_cell: 4,
            drainage_node_id: "J1".to_string(),
            rim_elevation_m,
            weir_crest_length_m: 2.0,
            weir_cd: 1.7,
            orifice_area_m2: 0.35,
            orifice_cd: 0.62,
            allow_surcharge_to_surface: true,
        }],
        control_claims: claims,
    }
}

#[test]
fn surface_drainage_admission_rejects_datum_rim_and_authority_conflicts() -> TestResult {
    // 1. Vertical datum mismatch
    let bad_datum = SurfaceDrainageBatch::new(
        make_surface(5.0, 0.5, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        capture_setup(5.0, "WGS84", Vec::new()),
    );
    assert!(matches!(bad_datum, Err(SurfaceDrainageError::Input(_))));

    // 2. Rim below SWMM node invert (invert = 3.048 m)
    let low_rim_invert = SurfaceDrainageBatch::new(
        make_surface(2.0, 0.5, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        capture_setup(2.5, "NAVD88", Vec::new()),
    );
    assert!(matches!(
        low_rim_invert,
        Err(SurfaceDrainageError::Input(_))
    ));

    // 3. Rim below surface cell bed (bed = 5.0 m)
    let low_rim_bed = SurfaceDrainageBatch::new(
        make_surface(5.0, 0.5, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        capture_setup(4.5, "NAVD88", Vec::new()),
    );
    assert!(matches!(low_rim_bed, Err(SurfaceDrainageError::Input(_))));

    // 4. External authority claim conflicting with native SWMM [CONTROLS] on OR1
    let conflict = SurfaceDrainageBatch::new(
        make_surface(5.0, 0.5, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        capture_setup(
            5.0,
            "NAVD88",
            vec![DrainageControlClaim {
                link_id: "OR1".to_string(),
                authority: SwmmControlAuthority::HydrActuatorImposedPosition,
            }],
        ),
    );
    assert!(matches!(
        conflict,
        Err(SurfaceDrainageError::Input(ref msg)) if msg.contains("duplicate-parameter-authority")
    ));

    Ok(())
}

#[test]
fn surface_drainage_capture_surcharge_rollback_and_restart_conserve_mass() -> TestResult {
    // Part A: Surface -> Drainage capture, exact surface mass balance, rollback, and restart.
    let setup = capture_setup(5.0, "NAVD88", Vec::new());

    // Reference uninterrupted run to t = 20 s.
    let mut reference = SurfaceDrainageBatch::new(
        make_surface(5.0, 0.6, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        setup.clone(),
    )?;
    assert!(
        reference
            .authority()
            .native_controlled_links()
            .contains("OR1")
    );
    let initial_vol = surface_total_volume_m3(reference.surface())?;
    for target in [10.0, 20.0] {
        while reference.time_s() < target - 1e-12 {
            let trial = reference.trial_to(target)?;
            trial.accept();
        }
    }
    let ref_surface_cells = reference.surface().state()?.cells().to_vec();
    let ref_drainage_state = reference.drainage().state().clone();
    let ref_transfer = reference.cumulative_inlet_transfers_m3()[0];
    let final_vol = surface_total_volume_m3(reference.surface())?;
    assert!(
        ref_transfer > 1.0,
        "expected significant surface->drainage capture, got {ref_transfer} m³"
    );
    assert!(
        ((initial_vol - final_vol) - ref_transfer).abs() < 1e-9,
        "surface mass balance mismatch: dV={} m³, transfer={ref_transfer} m³",
        initial_vol - final_vol
    );
    assert!(
        ref_drainage_state.continuity.external_inflow_m3 > 0.5 * ref_transfer,
        "SWMM must receive coupled inflow volume, got {} m³",
        ref_drainage_state.continuity.external_inflow_m3
    );

    // Perturbed run: advance to t = 10 s, drop a trial without accepting, then continue to 20 s.
    let mut perturbed = SurfaceDrainageBatch::new(
        make_surface(5.0, 0.6, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        setup.clone(),
    )?;
    while perturbed.time_s() < 10.0 - 1e-12 {
        let trial = perturbed.trial_to(10.0)?;
        trial.accept();
    }
    let mid_snap = perturbed.snapshot_data()?;
    assert_eq!(mid_snap.schema, SURFACE_DRAINAGE_SNAPSHOT_SCHEMA);
    assert_eq!(mid_snap.contract, SURFACE_DRAINAGE_CONTRACT_VERSION);
    let mid_steps = perturbed.accepted_steps();
    let mid_time = perturbed.time_s();

    {
        let dropped_trial = perturbed.trial_to(15.0)?;
        assert!(dropped_trial.diagnostics().end_s > mid_time);
        // Dropped without calling accept().
    }
    assert_eq!(perturbed.accepted_steps(), mid_steps);
    assert_eq!(perturbed.time_s().to_bits(), mid_time.to_bits());

    while perturbed.time_s() < 20.0 - 1e-12 {
        let trial = perturbed.trial_to(20.0)?;
        trial.accept();
    }
    assert_eq!(
        perturbed.surface().state()?.cells(),
        ref_surface_cells.as_slice(),
        "dropped trial rollback must preserve surface state bit-for-bit"
    );
    assert_eq!(
        perturbed.drainage().state(),
        &ref_drainage_state,
        "dropped trial rollback must preserve SWMM drainage state bit-for-bit"
    );

    // Cold restart into fresh Solver + fresh SWMM worker from the t = 10 s snapshot.
    let mut restarted = SurfaceDrainageBatch::new(
        make_surface(5.0, 0.6, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        setup,
    )?;
    restarted.import_snapshot(&mid_snap)?;
    while restarted.time_s() < 20.0 - 1e-12 {
        let trial = restarted.trial_to(20.0)?;
        trial.accept();
    }
    assert_eq!(
        restarted.surface().state()?.cells(),
        ref_surface_cells.as_slice(),
        "cold restart must match uninterrupted surface state bit-for-bit"
    );
    assert_eq!(
        restarted.drainage().state(),
        &ref_drainage_state,
        "cold restart must match uninterrupted SWMM drainage state bit-for-bit"
    );

    // Part B: Pressurized sewer -> dry surface surcharge (negative signed transfer).
    let mut surcharge_setup = capture_setup(4.0, "NAVD88", Vec::new());
    surcharge_setup.config.min_step_s = 0.01;
    surcharge_setup.config.max_retries = 12;
    surcharge_setup.inlets[0].orifice_area_m2 = 0.12;
    let mut surcharged = SurfaceDrainageBatch::new(
        make_surface(4.0, 0.0, 120.0)?,
        open_drainage(surcharge_network_inp(), SwmmSurchargeMethod::Slot)?,
        surcharge_setup,
    )?;
    while surcharged.time_s() < 60.0 - 1e-12 {
        let trial = surcharged.trial_to(60.0)?;
        trial.accept();
    }
    let surcharge_transfer = surcharged.cumulative_inlet_transfers_m3()[0];
    let surcharged_surface_vol = surface_total_volume_m3(surcharged.surface())?;
    assert!(
        surcharge_transfer < -0.5,
        "expected pressurized sewer->surface surcharge (< -0.5 m³), got {surcharge_transfer} m³"
    );
    assert!(
        (surcharged_surface_vol - (-surcharge_transfer)).abs() < 1e-9,
        "surcharged surface volume ({surcharged_surface_vol} m³) must equal -transfer ({} m³)",
        -surcharge_transfer
    );

    // Part C: Live hydr-actuated control on OR2 (HydrActuatorImposedPosition) + native rejection on OR1.
    let hydr_ctrl_setup = capture_setup(
        5.0,
        "NAVD88",
        vec![DrainageControlClaim {
            link_id: "OR2".to_string(),
            authority: SwmmControlAuthority::HydrActuatorImposedPosition,
        }],
    );
    let mut controlled = SurfaceDrainageBatch::new(
        make_surface(5.0, 0.6, 60.0)?,
        open_drainage(capture_network_inp(), SwmmSurchargeMethod::Extran)?,
        hydr_ctrl_setup,
    )?;
    // Commanding OR1 (owned by native SWMM [CONTROLS]) must reject.
    assert!(matches!(
        controlled.trial_to_with_commands(5.0, &[("OR1".to_string(), 0.2)]),
        Err(SurfaceDrainageError::Input(ref msg)) if msg.contains("duplicate-parameter-authority")
    ));
    // Commanding OR2 (owned by HydrActuatorImposedPosition) imposes setting = 0.35.
    let trial = controlled.trial_to_with_commands(5.0, &[("OR2".to_string(), 0.35)])?;
    trial.accept();
    let or2_idx = controlled
        .drainage()
        .link_index("OR2")
        .ok_or("missing OR2")?;
    let or2_setting = controlled.drainage().state().links[or2_idx].setting;
    assert!(
        (or2_setting - 0.35).abs() < 1e-12,
        "imposed hydr command on OR2 must set SWMM Link.setting = 0.35, got {or2_setting}"
    );

    Ok(())
}
