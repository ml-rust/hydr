//! Complete reservoir continuation and atomic owned-package admission.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "nonlinear_reservoir/fixture.rs"]
mod fixture;

use fixture::{Result, parameters, single};
use hydr::channel::CHANNEL_SNAPSHOT_SCHEMA;
use hydr::runoff::{
    RESERVOIR_CONTRACT_VERSION, RESERVOIR_SNAPSHOT_SCHEMA, ReservoirConfig, ReservoirError,
    ReservoirInput, ReservoirParameters, ReservoirSnapshotData,
};

fn identity_package() -> Result<ReservoirSnapshotData> {
    let batch = single(parameters()?, 0.25 * 0.3048, ReservoirConfig::default())?;
    Ok(batch
        .trial_to(
            1.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: 0.002,
            }],
        )?
        .into_next()
        .snapshot_data()?)
}

fn reject_identity(changed: &ReservoirSnapshotData) -> Result<String> {
    let mut destination = single(parameters()?, 0.25 * 0.3048, ReservoirConfig::default())?;
    let before = destination.snapshot_data()?;
    let Err(error) = destination.restore_data(changed) else {
        return Err("foreign identity was accepted".into());
    };
    assert!(
        matches!(error, ReservoirError::InvalidInput { .. }),
        "unexpected kind: {error}"
    );
    assert_eq!(destination.snapshot_data()?, before);
    Ok(error.to_string())
}

#[test]
fn reservoir_package_carries_identity() -> Result {
    assert_eq!(RESERVOIR_SNAPSHOT_SCHEMA, "hydr-reservoir-checkpoint-1");
    assert_eq!(RESERVOIR_CONTRACT_VERSION, "hydr-reservoir-numr-cpu-1");
    let package = identity_package()?;
    assert_eq!(package.schema, RESERVOIR_SNAPSHOT_SCHEMA);
    assert_eq!(package.contract, RESERVOIR_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn reservoir_import_rejects_foreign_schema() -> Result {
    let package = identity_package()?;
    for found in ["hydr-reservoir-checkpoint-0", ""] {
        let mut changed = package.clone();
        changed.schema = found.into();
        let message = reject_identity(&changed)?;
        assert!(
            message.contains(&format!("found \"{found}\"")),
            "missing found value: {message}"
        );
        assert!(
            message.contains(RESERVOIR_SNAPSHOT_SCHEMA),
            "missing expected schema: {message}"
        );
        assert!(
            message.contains("capture a new package"),
            "missing action: {message}"
        );
    }
    Ok(())
}

#[test]
fn reservoir_import_rejects_foreign_contract() -> Result {
    let mut changed = identity_package()?;
    changed.contract = "hydr-reservoir-numr-gpu-1".into();
    let message = reject_identity(&changed)?;
    assert!(message.contains("contract"), "missing contract: {message}");
    assert!(
        message.contains("hydr-reservoir-numr-gpu-1"),
        "missing found value: {message}"
    );
    assert!(
        message.contains(RESERVOIR_CONTRACT_VERSION),
        "missing expected contract: {message}"
    );
    Ok(())
}

#[test]
fn reservoir_import_rejects_channel_schema() -> Result {
    let mut changed = identity_package()?;
    changed.schema = CHANNEL_SNAPSHOT_SCHEMA.into();
    let message = reject_identity(&changed)?;
    assert!(
        message.contains(CHANNEL_SNAPSHOT_SCHEMA),
        "missing found value: {message}"
    );
    assert!(
        message.contains(RESERVOIR_SNAPSHOT_SCHEMA),
        "missing expected schema: {message}"
    );
    Ok(())
}

#[test]
fn trusted_and_owned_restart_reproduce_complete_accepted_continuation() -> Result {
    let p = parameters()?;
    let initial_depth = 0.25 * 0.3048;
    let initial = single(p, initial_depth, ReservoirConfig::default())?;
    let mut continued = initial
        .trial_to(
            1.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: 0.0,
            }],
        )?
        .into_next();
    let snapshot = continued.snapshot();
    let package = continued.snapshot_data()?;
    let mut trusted = single(p, initial_depth, ReservoirConfig::default())?;
    let mut owned = single(p, initial_depth, ReservoirConfig::default())?;
    trusted.restore(&snapshot)?;
    owned.restore_data(&package)?;
    assert_eq!(owned.snapshot_data()?, package);
    for (end, rate) in [(2.0, 0.3048 / 1024.0), (3.0, 0.0), (4.0, 0.002)] {
        let inputs = [ReservoirInput {
            incoming_rate_m_per_s: rate,
        }];
        continued = continued.trial_to(end, &inputs)?.into_next();
        trusted = trusted.trial_to(end, &inputs)?.into_next();
        owned = owned.trial_to(end, &inputs)?.into_next();
        assert_eq!(continued.snapshot_data()?, trusted.snapshot_data()?);
        assert_eq!(continued.snapshot_data()?, owned.snapshot_data()?);
    }
    assert_eq!(continued.accepted_steps(), 4);
    assert!(continued.ledger().incoming_volume_m3 > 0.0);
    assert!(continued.ledger().export_volume_m3 > 0.0);
    Ok(())
}

#[test]
fn initial_owned_package_preserves_negative_zero_depth_bits() -> Result {
    let p = ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.01, 0.03, 0.125)?;
    let original = single(p, -0.0, ReservoirConfig::default())?;
    let package = original.snapshot_data()?;
    assert_eq!(package.states[0].depth_m.to_bits(), (-0.0_f64).to_bits());
    let mut destination = single(p, 0.0, ReservoirConfig::default())?;
    destination.restore_data(&package)?;
    assert_eq!(
        destination.states()?[0].depth_m.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        destination.snapshot_data()?.states[0].depth_m.to_bits(),
        (-0.0_f64).to_bits()
    );
    Ok(())
}

#[test]
fn exact_definition_identity_includes_geometry_control_bits_and_element_ids() -> Result {
    let p = ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.01, 0.03, 0.125)?;
    let config = ReservoirConfig {
        root_relative_depth: 0.0,
        ..ReservoirConfig::default()
    };
    let original = single(p, 0.0, config.clone())?;
    let package = original.snapshot_data()?;
    let mut corrupt = Vec::new();
    let mut changed = package.clone();
    changed.elements[0].id = "different storage".into();
    corrupt.push(changed);
    let mut changed = package.clone();
    changed.elements[0].parameters =
        ReservoirParameters::from_si(1.0, 1.0, -0.0, 0.01, 0.03, 0.125)?;
    corrupt.push(changed);
    let mut changed = package.clone();
    changed.config.root_relative_depth = -0.0;
    corrupt.push(changed);
    let mut changed = package.clone();
    changed.config.root_iterations += 1;
    corrupt.push(changed);
    let mut destination = single(p, 0.0, config)?;
    let before = destination.snapshot_data()?;
    for (index, package) in corrupt.iter().enumerate() {
        assert!(
            destination.restore_data(package).is_err(),
            "definition corruption {index}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn inconsistent_state_ledger_and_timing_reject_before_mutation() -> Result {
    let p = parameters()?;
    let original = single(p, 0.25 * 0.3048, ReservoirConfig::default())?
        .trial_to(
            1.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: 0.001,
            }],
        )?
        .into_next();
    let package = original.snapshot_data()?;
    let changes: &[fn(&mut ReservoirSnapshotData)] = &[
        |data| data.time = f64::NAN,
        |data| data.time = -1.0,
        |data| data.time = 0.5,
        |data| data.states.clear(),
        |data| data.states[0].depth_m = -1.0,
        |data| data.states[0].depth_m = f64::INFINITY,
        |data| data.states[0].depth_m += 1.0,
        |data| data.ledger.initial_storage_m3 += 1.0,
        |data| data.ledger.storage_m3 += 1.0,
        |data| data.ledger.incoming_volume_m3 = -1.0,
        |data| data.ledger.incoming_volume_m3 = 0.0,
        |data| data.ledger.export_volume_m3 = -1.0,
        |data| data.ledger.export_volume_m3 = 0.0,
        |data| data.ledger.residual_m3 = 1.0,
        |data| data.accepted_steps = 0,
        |data| data.last_step = None,
    ];
    let mut destination = single(p, 0.0, ReservoirConfig::default())?;
    let before = destination.snapshot_data()?;
    for (index, modify) in changes.iter().enumerate() {
        let mut corrupted = package.clone();
        modify(&mut corrupted);
        assert!(
            destination.restore_data(&corrupted).is_err(),
            "package corruption {index}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn forged_root_source_and_branch_receipts_reject_atomically() -> Result {
    let p = parameters()?;
    let original = single(p, 0.25 * 0.3048, ReservoirConfig::default())?
        .trial_to(
            1.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: 0.0,
            }],
        )?
        .into_next();
    let package = original.snapshot_data()?;
    let mut destination = single(p, 0.0, ReservoirConfig::default())?;
    let before = destination.snapshot_data()?;
    for quantity in 0..26 {
        let mut corrupted = package.clone();
        let step = corrupted
            .last_step
            .as_mut()
            .ok_or("missing accepted step")?;
        let receipt = step
            .receipts
            .first_mut()
            .ok_or("missing reservoir receipt")?;
        match quantity {
            0 => step.start = -1.0,
            1 => step.end += 1.0,
            2 => step.dt *= 2.0,
            3 => step.root_iterations = usize::MAX,
            4 => step.storage_before_m3 += 1.0,
            5 => step.storage_after_m3 += 1.0,
            6 => step.incoming_volume_m3 = 1.0,
            7 => step.export_volume_m3 = 0.0,
            8 => step.constitutive_residual_m3 = 1.0,
            9 => step.water_residual_m3 = 1.0,
            10 => receipt.element = usize::MAX,
            11 => receipt.old_depth_m = -1.0,
            12 => receipt.incoming_rate_m_per_s = -1.0,
            13 => receipt.incoming_depth_m = 1.0,
            14 => receipt.available_depth_m += 1.0,
            15 => receipt.active_duration_s = 0.0,
            16 => receipt.root_lower_m = receipt.root_upper_m + 1.0,
            17 => receipt.root_upper_m = f64::NAN,
            18 => receipt.depth_m += 1.0,
            19 => receipt.export_depth_m = 0.0,
            20 => receipt.export_volume_m3 = 0.0,
            21 => receipt.endpoint_rate_m_per_s = None,
            22 => receipt.endpoint_discharge_m3_per_s = None,
            23 => receipt.law_depth_m = 0.0,
            24 => receipt.law_volume_m3 = 0.0,
            25 => step.receipts.clear(),
            _ => return Err("unknown receipt corruption".into()),
        }
        assert!(
            destination.restore_data(&corrupted).is_err(),
            "receipt corruption {quantity}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn instantaneous_receipts_cannot_forge_a_finite_endpoint_law() -> Result {
    let p = ReservoirParameters::from_native(1024.0, 1024.0, 32.0, 1.0 / 64.0, 0.0, 0.125)?;
    let original = single(p, 0.0, ReservoirConfig::default())?
        .trial_to(
            256.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: 0.3048 / 1024.0,
            }],
        )?
        .into_next();
    let mut package = original.snapshot_data()?;
    let receipt = &mut package
        .last_step
        .as_mut()
        .ok_or("missing accepted step")?
        .receipts[0];
    receipt.endpoint_rate_m_per_s = Some(0.0);
    receipt.endpoint_discharge_m3_per_s = Some(0.0);
    let mut destination = single(p, 0.0, ReservoirConfig::default())?;
    let before = destination.snapshot_data()?;
    assert!(destination.restore_data(&package).is_err());
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}
