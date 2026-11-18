//! Channel package schema and contract identity.

use super::{TestResult, advance, reach};
use hydr::channel::{CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA, ChannelError};
use hydr::storage::LEVEL_POOL_SNAPSHOT_SCHEMA;

#[test]
fn channel_package_carries_identity() -> TestResult {
    assert_eq!(CHANNEL_SNAPSHOT_SCHEMA, "hydr-channel-checkpoint-2");
    assert_eq!(CHANNEL_CONTRACT_VERSION, "hydr-channel-1d-numr-cpu-2");
    let package = advance(reach()?, 2)?.snapshot_data()?;
    assert_eq!(package.schema, CHANNEL_SNAPSHOT_SCHEMA);
    assert_eq!(package.contract, CHANNEL_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn channel_import_rejects_foreign_schema() -> TestResult {
    let origin = reach()?;
    let package = advance(origin.clone(), 2)?.snapshot_data()?;
    let before = origin.snapshot_data()?;
    // Schema `1` packages predate the FluxPort endpoint and must be captured again.
    for found in ["hydr-channel-checkpoint-1", "hydr-channel-checkpoint-0", ""] {
        let mut changed = package.clone();
        changed.schema = found.into();
        let Err(error) = origin.import_snapshot(&changed) else {
            return Err("foreign schema was accepted".into());
        };
        assert!(
            matches!(error, ChannelError::InvalidInput { .. }),
            "unexpected kind: {error}"
        );
        let message = error.to_string();
        assert!(
            message.contains(&format!("found \"{found}\"")),
            "missing found value: {message}"
        );
        assert!(
            message.contains(CHANNEL_SNAPSHOT_SCHEMA),
            "missing expected schema: {message}"
        );
        assert!(
            message.contains("capture a new package"),
            "missing action: {message}"
        );
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn channel_import_rejects_foreign_contract() -> TestResult {
    let origin = reach()?;
    let package = advance(origin.clone(), 2)?.snapshot_data()?;
    let before = origin.snapshot_data()?;
    // Contract `1` predates the FluxPort endpoint.
    for found in ["hydr-channel-1d-numr-gpu-1", "hydr-channel-1d-numr-cpu-1"] {
        let mut changed = package.clone();
        changed.contract = found.into();
        let Err(error) = origin.import_snapshot(&changed) else {
            return Err(format!("foreign contract {found} was accepted").into());
        };
        assert!(
            matches!(error, ChannelError::InvalidInput { .. }),
            "unexpected kind: {error}"
        );
        let message = error.to_string();
        assert!(message.contains("contract"), "missing contract: {message}");
        assert!(message.contains(found), "missing found value: {message}");
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn channel_import_rejects_level_pool_schema() -> TestResult {
    let origin = reach()?;
    let mut changed = advance(origin.clone(), 2)?.snapshot_data()?;
    changed.schema = LEVEL_POOL_SNAPSHOT_SCHEMA.into();
    let before = origin.snapshot_data()?;
    let Err(error) = origin.import_snapshot(&changed) else {
        return Err("level-pool schema was accepted".into());
    };
    let message = error.to_string();
    assert!(
        message.contains(LEVEL_POOL_SNAPSHOT_SCHEMA),
        "missing found value: {message}"
    );
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}
