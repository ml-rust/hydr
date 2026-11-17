//! Controlled gate package schema and contract identity, including nested children.

use super::fixture::*;
use hydr::structures::*;

fn package() -> Result<ControlledGateSnapshotData> {
    Ok(accept(Config::switching().build()?, 1.0, Some(2.0))?.snapshot_data()?)
}

fn rejects(changed: &ControlledGateSnapshotData, needles: &[&str]) -> Result {
    let mut target = Config::switching().build()?;
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
fn controlled_gate_package_carries_identity() -> Result {
    assert_eq!(
        CONTROLLED_GATE_SNAPSHOT_SCHEMA,
        "hydr-controlled-gate-checkpoint-1"
    );
    assert_eq!(
        CONTROLLED_GATE_CONTRACT_VERSION,
        "hydr-controlled-gate-numr-cpu-1"
    );
    let data = package()?;
    assert_eq!(data.schema, CONTROLLED_GATE_SNAPSHOT_SCHEMA);
    assert_eq!(data.contract, CONTROLLED_GATE_CONTRACT_VERSION);
    assert_eq!(data.controls.schema, OPERATING_CONTROL_SNAPSHOT_SCHEMA);
    assert_eq!(data.actuators.schema, GATE_ACTUATOR_SNAPSHOT_SCHEMA);
    Ok(())
}

#[test]
fn controlled_gate_import_rejects_foreign_schema() -> Result {
    for found in ["hydr-controlled-gate-checkpoint-0", ""] {
        let mut changed = package()?;
        changed.schema = found.into();
        rejects(
            &changed,
            &[
                &format!("found \"{found}\""),
                CONTROLLED_GATE_SNAPSHOT_SCHEMA,
                "controlled gate checkpoint schema",
                "capture a new package",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn controlled_gate_import_rejects_foreign_contract() -> Result {
    let mut changed = package()?;
    changed.contract = "hydr-controlled-gate-numr-gpu-1".into();
    rejects(
        &changed,
        &[
            "found \"hydr-controlled-gate-numr-gpu-1\"",
            CONTROLLED_GATE_CONTRACT_VERSION,
            "controlled gate checkpoint contract",
            "capture a new package",
        ],
    )
}

#[test]
fn controlled_gate_import_rejects_nested_actuator_schema() -> Result {
    let mut changed = package()?;
    changed.actuators.schema = "hydr-gate-actuator-checkpoint-0".into();
    rejects(
        &changed,
        &[
            "found \"hydr-gate-actuator-checkpoint-0\"",
            GATE_ACTUATOR_SNAPSHOT_SCHEMA,
            "gate actuator checkpoint schema",
            "capture a new package",
        ],
    )
}
