//! Integration tests for `hydr::drainage` (`hydr-drainage`) over the pinned EPA SWMM v5.2.4 worker.

use hydr::drainage::{
    DrainageError, SWMM_DRAINAGE_CONTRACT_VERSION, SWMM_DRAINAGE_SNAPSHOT_SCHEMA,
    SwmmControlAuthority, SwmmDrainage, SwmmDrainageDefinition, SwmmSurchargeMethod,
};
use std::path::PathBuf;

const PROVENANCE: &str = "EPA SWMM v5.2.4 commit 7952ca837988b1c32f791812eccc9fd64547e093";
const SURCHARGE_EVIDENCE: &str = "Dynamic-wave test network verified under EXTRAN and SLOT";

fn worker_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resource/reference-engines/swmm/build-drainage-worker/swmm-drainage-worker")
}

/// Dynamic-wave network with storage, pump, orifice with travel time, and native controls
/// (`TIMEOPEN` + depth threshold) that exercise SWMM hidden state across rollback/restart.
fn controlled_network_inp(surcharge: &str) -> String {
    format!(
        r#"[TITLE]
Hydr SWMM Drainage Verification Network

[OPTIONS]
FLOW_UNITS           CFS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
REPORT_START_DATE    01/01/2025
REPORT_START_TIME    00:00:00
END_DATE             01/01/2025
END_TIME             02:00:00
ROUTING_STEP         00:00:05
MINIMUM_STEP         0.5
VARIABLE_STEP        0.75
INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
FORCE_MAIN_EQUATION  H-W
SURCHARGE_METHOD     {surcharge}
MIN_SURFAREA         12.566
MAX_TRIALS           8
HEAD_TOLERANCE       0.005

[JUNCTIONS]
J1  10.0  8.0  0.0  20.0  0.0
J2   8.0  8.0  0.0  20.0  0.0
J3   5.0  8.0  0.0  20.0  0.0

[OUTFALLS]
OUT1  2.0  FREE  NO
OUT2  2.0  FREE  NO

[STORAGE]
SU1  9.0  10.0  0.5  FUNCTIONAL  1000  0  0  0

[CONDUITS]
C1   J1   SU1  200.0  0.013  0.0  0.0  0.0  0.0
C2   J2   J3   250.0  0.013  0.0  0.0  0.0  0.0
C3   J3   OUT1 200.0  0.013  0.0  0.0  0.0  0.0

[ORIFICES]
OR1  SU1  J2   SIDE  0.5  0.65  NO  0.05

[PUMPS]
P1   SU1  OUT2  PC1  ON  0.0  0.0

[XSECTIONS]
C1   CIRCULAR  2.0  0  0  0  1
C2   CIRCULAR  2.0  0  0  0  1
C3   CIRCULAR  2.0  0  0  0  1
OR1  CIRCULAR  1.5  0  0  0

[CURVES]
PC1  PUMP2  0.0  0.0
PC1         1.0  4.0
PC1         3.0  10.0
PC1         8.0  14.0

[INFLOWS]
J1   FLOW  ""  FLOW  1.0  1.0  8.0

[CONTROLS]
RULE R_ORIFICE_OPEN
IF NODE SU1 DEPTH > 1.5
THEN ORIFICE OR1 SETTING = 1.0
ELSE ORIFICE OR1 SETTING = 0.25

RULE R_PUMP_DELAY
IF NODE SU1 DEPTH > 2.0
AND LINK OR1 TIMEOPEN >= 0.05
THEN PUMP P1 STATUS = ON
ELSE PUMP P1 STATUS = OFF
"#
    )
}

/// Pressurized surcharge network (same geometry as `swmm_drainage_verify.py`).
fn surcharge_network_inp(surcharge: &str) -> String {
    format!(
        r#"[TITLE]
Surcharge Test ({surcharge})

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
SURCHARGE_METHOD     {surcharge}

[JUNCTIONS]
J1  10.0  10.0  0.0  0.0  0.0
J2   9.0  10.0  0.0  0.0  0.0

[OUTFALLS]
OUT1  8.0  FREE  NO

[CONDUITS]
C1  J1  J2    300.0  0.013  0.0  0.0  0.0  0.0
C2  J2  OUT1  300.0  0.013  0.0  0.0  0.0  0.0

[XSECTIONS]
C1  CIRCULAR  2.0  0  0  0  1
C2  CIRCULAR  2.0  0  0  0  1

[INFLOWS]
J1  FLOW  ""  FLOW  1.0  1.0  40.0
"#
    )
}

fn make_def(inp: String, method: SwmmSurchargeMethod) -> SwmmDrainageDefinition {
    match SwmmDrainageDefinition::new(
        inp,
        "NAVD88",
        method,
        SwmmControlAuthority::NativeSwmm,
        worker_binary(),
        PROVENANCE,
        SURCHARGE_EVIDENCE,
    ) {
        Ok(def) => def,
        Err(err) => unreachable!("valid test definition must construct: {err}"),
    }
}

#[test]
fn swmm_drainage_admission_rejects_invalid_definitions_and_missing_worker() {
    let inp = controlled_network_inp("EXTRAN");

    let empty_datum = SwmmDrainageDefinition::new(
        inp.clone(),
        "   ",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::NativeSwmm,
        worker_binary(),
        PROVENANCE,
        SURCHARGE_EVIDENCE,
    );
    assert!(matches!(empty_datum, Err(DrainageError::Input(_))));

    let hybrid_def = SwmmDrainageDefinition::new(
        inp.clone(),
        "NAVD88",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::HydrCommandsNativeActuation,
        worker_binary(),
        PROVENANCE,
        SURCHARGE_EVIDENCE,
    );
    assert!(hybrid_def.is_ok());

    let missing_bin_def = SwmmDrainageDefinition::new(
        inp,
        "NAVD88",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::NativeSwmm,
        PathBuf::from("/nonexistent/path/to/swmm-drainage-worker"),
        PROVENANCE,
        SURCHARGE_EVIDENCE,
    );
    assert!(missing_bin_def.is_ok());
    if let Ok(def) = missing_bin_def {
        let opened = SwmmDrainage::open(def);
        assert!(matches!(opened, Err(DrainageError::BackendUnavailable(_))));
    }
}

#[test]
fn swmm_drainage_rejects_surcharge_method_mismatch_and_tampered_snapshots()
-> Result<(), Box<dyn std::error::Error>> {
    let bin = worker_binary();
    if !bin.is_file() {
        eprintln!(
            "skipping live SWMM drainage test: worker missing at {}",
            bin.display()
        );
        return Ok(());
    }

    // .inp declares SLOT, definition declares Extran -> must reject.
    let mismatch_def = make_def(controlled_network_inp("SLOT"), SwmmSurchargeMethod::Extran);
    assert!(matches!(
        SwmmDrainage::open(mismatch_def),
        Err(DrainageError::Input(_))
    ));

    let def = make_def(
        controlled_network_inp("EXTRAN"),
        SwmmSurchargeMethod::Extran,
    );
    let mut engine = SwmmDrainage::open(def)?;
    let valid = engine.snapshot_data();
    assert_eq!(valid.schema, SWMM_DRAINAGE_SNAPSHOT_SCHEMA);
    assert_eq!(valid.contract, SWMM_DRAINAGE_CONTRACT_VERSION);

    let mut bad_schema = valid.clone();
    bad_schema.schema = "forged-schema".to_string();
    assert!(matches!(
        engine.import_snapshot(&bad_schema),
        Err(DrainageError::Input(_))
    ));

    let mut bad_contract = valid.clone();
    bad_contract.contract = "forged-contract".to_string();
    assert!(matches!(
        engine.import_snapshot(&bad_contract),
        Err(DrainageError::Input(_))
    ));

    let mut bad_inp = valid.clone();
    bad_inp.inp_sha256 = "0".repeat(64);
    assert!(matches!(
        engine.import_snapshot(&bad_inp),
        Err(DrainageError::Input(_))
    ));

    let mut bad_datum = valid.clone();
    bad_datum.vertical_datum = "WGS84".to_string();
    assert!(matches!(
        engine.import_snapshot(&bad_datum),
        Err(DrainageError::Input(_))
    ));

    let mut bad_digest = valid.clone();
    bad_digest.state_digest_sha256 = "f".repeat(64);
    assert!(matches!(
        engine.import_snapshot(&bad_digest),
        Err(DrainageError::Input(_))
    ));

    Ok(())
}

#[test]
fn swmm_drainage_rollback_and_cold_restart_are_bit_identical()
-> Result<(), Box<dyn std::error::Error>> {
    let bin = worker_binary();
    if !bin.is_file() {
        eprintln!(
            "skipping live SWMM drainage test: worker missing at {}",
            bin.display()
        );
        return Ok(());
    }

    let def = make_def(
        controlled_network_inp("EXTRAN"),
        SwmmSurchargeMethod::Extran,
    );

    // 1. Uninterrupted reference run: 6 steps of 60 s, then 6 steps of 60 s (total 720 s).
    let mut reference = SwmmDrainage::open(def.clone())?;
    for step in 1..=12 {
        let trial = reference.trial_to(step as f64 * 60.0, &[])?;
        trial.accept();
    }
    let ref_state = reference.state().clone();

    // 2. Perturbed run: advance 6 steps (360 s), run a large perturbing trial with extra inflow
    // and drop it without accepting, then continue 6 steps to 720 s.
    let mut perturbed = SwmmDrainage::open(def.clone())?;
    for step in 1..=6 {
        let trial = perturbed.trial_to(step as f64 * 60.0, &[])?;
        trial.accept();
    }
    let mid_snapshot = perturbed.snapshot_data();
    assert_eq!(perturbed.accepted_steps(), 6);
    assert!((perturbed.time_s() - 360.0).abs() <= 1e-9);

    let j1_idx = perturbed.node_index("J1").ok_or("missing node J1")?;
    {
        // Perturbing trial that changes depths, controls, and storage, then drops (rollback).
        let bad_trial = perturbed.trial_to(600.0, &[(j1_idx, 2.5)])?;
        assert!(bad_trial.end_s() > 360.0);
        // Dropped without calling accept().
    }
    assert_eq!(perturbed.accepted_steps(), 6);
    assert!((perturbed.time_s() - 360.0).abs() <= 1e-9);

    for step in 7..=12 {
        let trial = perturbed.trial_to(step as f64 * 60.0, &[])?;
        trial.accept();
    }
    assert_eq!(
        perturbed.state(),
        &ref_state,
        "dropped trial rollback must match uninterrupted reference bit-for-bit"
    );

    // 3. Cold-process restart from the 360 s snapshot in a brand-new worker process.
    let mut restarted = SwmmDrainage::resume(def, &mid_snapshot)?;
    assert_eq!(restarted.accepted_steps(), 6);
    for step in 7..=12 {
        let trial = restarted.trial_to(step as f64 * 60.0, &[])?;
        trial.accept();
    }
    assert_eq!(
        restarted.state(),
        &ref_state,
        "cold-process restart must match uninterrupted reference bit-for-bit"
    );

    Ok(())
}

#[test]
fn swmm_drainage_extran_and_slot_pressurized_surcharge_and_si_inflow_continuity()
-> Result<(), Box<dyn std::error::Error>> {
    let bin = worker_binary();
    if !bin.is_file() {
        eprintln!(
            "skipping live SWMM drainage test: worker missing at {}",
            bin.display()
        );
        return Ok(());
    }

    // Pressurized surcharge under EXTRAN vs SLOT (conduit crown = 2.0 ft = 0.6096 m).
    let crown_m = 2.0 * hydr::drainage::FT_TO_M;
    for (inp_kw, method) in [
        ("EXTRAN", SwmmSurchargeMethod::Extran),
        ("SLOT", SwmmSurchargeMethod::Slot),
    ] {
        let def = make_def(surcharge_network_inp(inp_kw), method);
        let mut engine = SwmmDrainage::open(def)?;
        assert_eq!(engine.summary().surcharge_method, method);
        let mut max_j1_depth_m = 0.0_f64;
        for step in 1..=20 {
            let trial = engine.trial_to(step as f64 * 30.0, &[])?;
            let state = trial.accept();
            let j1 = &state.nodes[0];
            max_j1_depth_m = max_j1_depth_m.max(j1.depth_m);
        }
        assert!(
            max_j1_depth_m > crown_m,
            "{inp_kw} must surcharge above crown ({crown_m:.4} m), got {max_j1_depth_m:.4} m"
        );
        assert!(
            engine.state().continuity.continuity_error_pct.abs() < 5.0,
            "{inp_kw} continuity error too large: {}%",
            engine.state().continuity.continuity_error_pct
        );
    }

    // Coupled SI inflow injection: inject 0.25 m³/s for 600 s (= 150.0 m³), then 0.0 m³/s
    // for 300 s, and verify external_inflow_m3 reflects the SI conversion accurately.
    let zero_inflow_inp = controlled_network_inp("EXTRAN").replace(
        "J1   FLOW  \"\"  FLOW  1.0  1.0  8.0",
        "J1   FLOW  \"\"  FLOW  1.0  1.0  0.0",
    );
    let def = make_def(zero_inflow_inp, SwmmSurchargeMethod::Extran);
    let mut engine = SwmmDrainage::open(def)?;
    let j1_idx = engine.node_index("J1").ok_or("missing node J1")?;
    let rate_m3_per_s = 0.25;
    for step in 1..=20 {
        let trial = engine.trial_to(step as f64 * 30.0, &[(j1_idx, rate_m3_per_s)])?;
        trial.accept();
    }
    let after_pulse_inflow_m3 = engine.state().continuity.external_inflow_m3;
    let expected_m3 = rate_m3_per_s * 600.0;
    assert!(
        (after_pulse_inflow_m3 - expected_m3).abs() / expected_m3 < 0.01,
        "SI injected volume mismatch: got {after_pulse_inflow_m3:.4} m³, expected {expected_m3:.4} m³"
    );

    // Passing empty inflow slice must clear previous apiExtInflow to 0.0.
    // SWMM integrates nodal lateral inflow trapezoidally, so the first substep after
    // returning to 0.0 completes the trailing half-trapezoid and brings total volume to
    // exact `rate * 600 s`; all subsequent zero-inflow steps must add 0.0 m³.
    let step21 = engine.trial_to(21.0 * 30.0, &[])?;
    step21.accept();
    let after_trailing_trapezoid_m3 = engine.state().continuity.external_inflow_m3;
    let max_trapezoid_half_step_m3 = 0.5 * rate_m3_per_s * 5.0;
    assert!(
        (after_trailing_trapezoid_m3 - expected_m3).abs() <= max_trapezoid_half_step_m3 + 1e-9,
        "total trapezoidal SI volume ({after_trailing_trapezoid_m3:.6} m³) outside half-step bound around {expected_m3:.6} m³"
    );

    for step in 22..=30 {
        let trial = engine.trial_to(step as f64 * 30.0, &[])?;
        trial.accept();
    }
    let after_zero_inflow_m3 = engine.state().continuity.external_inflow_m3;
    assert!(
        (after_zero_inflow_m3 - after_trailing_trapezoid_m3).abs() < 1e-12,
        "omitting a node from node_inflows_m3_per_s must add zero inflow after trailing substep (delta = {})",
        after_zero_inflow_m3 - after_trailing_trapezoid_m3
    );

    Ok(())
}
