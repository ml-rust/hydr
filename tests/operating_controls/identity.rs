//! Operating control package schema and contract identity.

use super::fixture::*;
use hydr::structures::*;

fn destination() -> Result<OperatingControlBatch<crate::runtime::Runtime>> {
    batch("positive_delays_exact_expiry", 1)
}

fn package() -> Result<OperatingControlSnapshotData> {
    Ok(accept(destination()?, 1.0, 2.0)?.snapshot_data()?)
}

fn rejects(changed: &OperatingControlSnapshotData, needles: &[&str]) -> Result {
    let mut target = destination()?;
    let before = target.snapshot_data()?;
    let error = rejected(target.restore_data(changed))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition),
        "{error}"
    );
    let message = error.to_string();
    for needle in needles {
        assert!(message.contains(needle), "missing {needle}: {message}");
    }
    assert_eq!(target.snapshot_data()?, before);
    Ok(())
}

#[test]
fn operating_control_package_carries_identity() -> Result {
    assert_eq!(
        OPERATING_CONTROL_SNAPSHOT_SCHEMA,
        "hydr-operating-control-checkpoint-2"
    );
    assert_eq!(
        OPERATING_CONTROL_CONTRACT_VERSION,
        "hydr-operating-control-numr-cpu-1"
    );
    let data = package()?;
    assert_eq!(data.schema, OPERATING_CONTROL_SNAPSHOT_SCHEMA);
    assert_eq!(data.contract, OPERATING_CONTROL_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn operating_control_import_rejects_foreign_schema() -> Result {
    for found in [
        "hydr-operating-control-checkpoint-0",
        "hydr-operating-control-checkpoint-1",
        "",
    ] {
        let mut changed = package()?;
        changed.schema = found.into();
        rejects(
            &changed,
            &[
                &format!("found \"{found}\""),
                OPERATING_CONTROL_SNAPSHOT_SCHEMA,
                "operating control checkpoint schema",
                "capture a new package",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn operating_control_import_rejects_foreign_contract() -> Result {
    let mut changed = package()?;
    changed.contract = "hydr-operating-control-numr-gpu-1".into();
    rejects(
        &changed,
        &[
            "found \"hydr-operating-control-numr-gpu-1\"",
            OPERATING_CONTROL_CONTRACT_VERSION,
            "operating control checkpoint contract",
            "capture a new package",
        ],
    )
}
