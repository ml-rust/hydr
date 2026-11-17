//! Controlled pump package schema and contract identity, including the nested controls.

use super::fixture::*;
use hydr::structures::*;

fn context() -> Result<StructureBatch<crate::runtime::Runtime>> {
    Ok(StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?)
}

fn package() -> Result<ControlledPumpSnapshotData> {
    let batch = Config::switching().build(&context()?, &["pump"])?;
    Ok(accept(batch, 1.0, Some(2.0))?.snapshot_data()?)
}

fn rejects(changed: &ControlledPumpSnapshotData, needles: &[&str]) -> Result {
    let mut target = Config::switching().build(&context()?, &["pump"])?;
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
fn controlled_pump_package_carries_identity() -> Result {
    assert_eq!(
        CONTROLLED_PUMP_SNAPSHOT_SCHEMA,
        "hydr-controlled-pump-checkpoint-1"
    );
    assert_eq!(
        CONTROLLED_PUMP_CONTRACT_VERSION,
        "hydr-controlled-pump-numr-cpu-1"
    );
    let data = package()?;
    assert_eq!(data.schema, CONTROLLED_PUMP_SNAPSHOT_SCHEMA);
    assert_eq!(data.contract, CONTROLLED_PUMP_CONTRACT_VERSION);
    assert_eq!(data.controls.schema, OPERATING_CONTROL_SNAPSHOT_SCHEMA);
    Ok(())
}

#[test]
fn controlled_pump_import_rejects_foreign_schema() -> Result {
    for found in ["hydr-controlled-pump-checkpoint-0", ""] {
        let mut changed = package()?;
        changed.schema = found.into();
        rejects(
            &changed,
            &[
                &format!("found \"{found}\""),
                CONTROLLED_PUMP_SNAPSHOT_SCHEMA,
                "controlled pump checkpoint schema",
                "capture a new package",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn controlled_pump_import_rejects_foreign_contract() -> Result {
    let mut changed = package()?;
    changed.contract = "hydr-controlled-pump-numr-gpu-1".into();
    rejects(
        &changed,
        &[
            "found \"hydr-controlled-pump-numr-gpu-1\"",
            CONTROLLED_PUMP_CONTRACT_VERSION,
            "controlled pump checkpoint contract",
            "capture a new package",
        ],
    )
}

#[test]
fn controlled_pump_import_rejects_nested_control_contract() -> Result {
    let mut changed = package()?;
    changed.controls.contract = "hydr-operating-control-numr-gpu-1".into();
    rejects(
        &changed,
        &[
            "found \"hydr-operating-control-numr-gpu-1\"",
            OPERATING_CONTROL_CONTRACT_VERSION,
            "operating control checkpoint contract",
        ],
    )
}
