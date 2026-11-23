//! Gate actuator package schema and contract identity.

use super::fixture::*;
use hydr::structures::*;

fn destination() -> Result<GateActuatorBatch<crate::runtime::Runtime>> {
    build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)
}

fn package() -> Result<GateActuatorSnapshotData> {
    Ok(walk(destination()?, 4.0, &[])?.snapshot_data()?)
}

fn rejects(changed: &GateActuatorSnapshotData, needles: &[&str]) -> Result {
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
fn gate_actuator_package_carries_identity() -> Result {
    assert_eq!(
        GATE_ACTUATOR_SNAPSHOT_SCHEMA,
        "hydr-gate-actuator-checkpoint-2"
    );
    assert_eq!(
        GATE_ACTUATOR_CONTRACT_VERSION,
        "hydr-gate-actuator-numr-cpu-1"
    );
    let data = package()?;
    assert_eq!(data.schema, GATE_ACTUATOR_SNAPSHOT_SCHEMA);
    assert_eq!(data.contract, GATE_ACTUATOR_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn gate_actuator_import_rejects_foreign_schema() -> Result {
    for found in [
        "hydr-gate-actuator-checkpoint-0",
        "hydr-gate-actuator-checkpoint-1",
        "",
    ] {
        let mut changed = package()?;
        changed.schema = found.into();
        rejects(
            &changed,
            &[
                &format!("found \"{found}\""),
                GATE_ACTUATOR_SNAPSHOT_SCHEMA,
                "gate actuator checkpoint schema",
                "capture a new package",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn gate_actuator_import_rejects_foreign_contract() -> Result {
    let mut changed = package()?;
    changed.contract = "hydr-gate-actuator-numr-gpu-1".into();
    rejects(
        &changed,
        &[
            "found \"hydr-gate-actuator-numr-gpu-1\"",
            GATE_ACTUATOR_CONTRACT_VERSION,
            "gate actuator checkpoint contract",
            "capture a new package",
        ],
    )
}
