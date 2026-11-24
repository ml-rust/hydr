//! Pinned EPA SWMM v5.2.4 drainage capability declaration checked against admission.

use hydr::capability::{ComponentId, UnsupportedReason};
use hydr::drainage::{
    DrainageError, SwmmControlAuthority, SwmmDrainage, SwmmDrainageDefinition, SwmmSurchargeMethod,
    swmm_drainage_capabilities,
};
use std::path::PathBuf;

fn worker_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resource/reference-engines/swmm/build-drainage-worker/swmm-drainage-worker")
}

fn minimal_inp(surcharge: &str) -> String {
    format!(
        r#"[OPTIONS]
FLOW_UNITS           CFS
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
REPORT_START_DATE    01/01/2025
REPORT_START_TIME    00:00:00
END_DATE             01/01/2025
END_TIME             00:10:00
ROUTING_STEP         00:00:05
SURCHARGE_METHOD     {surcharge}

[JUNCTIONS]
J1  10.0  8.0  0.0  0.0  0.0

[OUTFALLS]
OUT1  8.0  FREE  NO

[CONDUITS]
C1  J1  OUT1  100.0  0.013  0.0  0.0  0.0  0.0

[XSECTIONS]
C1  CIRCULAR  2.0  0  0  0  1
"#
    )
}

#[test]
fn swmm_drainage_declaration_matches_admission() -> Result<(), Box<dyn std::error::Error>> {
    let caps = swmm_drainage_capabilities();
    assert_eq!(caps.component, ComponentId::SwmmDrainage);

    let rejected_ids: Vec<&str> = caps
        .unsupported
        .iter()
        .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission)
        .map(|u| u.id)
        .collect();
    assert_eq!(
        rejected_ids,
        [
            "surcharge-method-mismatch",
            "missing-worker-binary",
            "tampered-snapshot",
        ]
    );

    // 1. missing-worker-binary -> BackendUnavailable
    let missing_def = SwmmDrainageDefinition::new(
        minimal_inp("EXTRAN"),
        "NAVD88",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::NativeSwmm,
        PathBuf::from("/nonexistent/swmm-drainage-worker"),
        "EPA SWMM v5.2.4",
        "Capability admission test",
    )?;
    assert!(matches!(
        SwmmDrainage::open(missing_def),
        Err(DrainageError::BackendUnavailable(_))
    ));

    // 2. surcharge-method-mismatch -> Input
    let bin = worker_binary();
    assert!(
        bin.is_file(),
        "pinned SWMM worker missing at {}",
        bin.display()
    );
    let mismatch_def = SwmmDrainageDefinition::new(
        minimal_inp("SLOT"),
        "NAVD88",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::NativeSwmm,
        bin.clone(),
        "EPA SWMM v5.2.4",
        "Capability admission test",
    )?;
    assert!(matches!(
        SwmmDrainage::open(mismatch_def),
        Err(DrainageError::Input(_))
    ));

    // 3. tampered-snapshot -> Input
    let valid_def = SwmmDrainageDefinition::new(
        minimal_inp("EXTRAN"),
        "NAVD88",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::NativeSwmm,
        bin,
        "EPA SWMM v5.2.4",
        "Capability admission test",
    )?;
    let mut engine = SwmmDrainage::open(valid_def)?;
    let mut forged = engine.snapshot_data();
    forged.state_digest_sha256 = "0".repeat(64);
    assert!(matches!(
        engine.import_snapshot(&forged),
        Err(DrainageError::Input(_))
    ));

    Ok(())
}
