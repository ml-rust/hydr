use super::fixture::{Result, constant, curve, input, rejection, single, unit};
use hydr::storage::{
    LEVEL_POOL_CONTRACT_VERSION, LEVEL_POOL_SNAPSHOT_SCHEMA, LevelPoolBatch, LevelPoolCurve,
    LevelPoolError, LevelPoolSnapshotData,
};

fn history() -> Result<LevelPoolBatch<crate::runtime::Runtime>> {
    let mut batch = single(constant()?, 1.0)?;
    for end in [1.0, 2.0] {
        batch = batch.trial_to(end, &[input(0.2, 0.1)])?.into_next();
    }
    Ok(batch)
}

#[test]
fn trusted_and_owned_restore_preserve_the_complete_accepted_partition() -> Result {
    let mut continued = history()?;
    let snapshot = continued.snapshot();
    let package = snapshot.data()?;
    assert_eq!(snapshot.time(), 2.0);
    let mut trusted = single(constant()?, 0.0)?;
    let mut owned = single(constant()?, 0.0)?;
    trusted.restore(&snapshot)?;
    owned.restore_data(&package)?;
    assert_eq!(owned.snapshot_data()?, package);
    for (end, exchange) in [(3.0, input(0.2, 0.1)), (4.0, input(0.1, 0.2))] {
        continued = continued.trial_to(end, &[exchange])?.into_next();
        trusted = trusted.trial_to(end, &[exchange])?.into_next();
        owned = owned.trial_to(end, &[exchange])?.into_next();
        assert_eq!(continued.snapshot_data()?, trusted.snapshot_data()?);
        assert_eq!(continued.snapshot_data()?, owned.snapshot_data()?);
        assert_eq!(continued.element_ledgers()?, owned.element_ledgers()?);
        assert_eq!(continued.totals()?, owned.totals()?);
    }
    Ok(())
}

#[test]
fn initial_signed_zero_and_volume_bits_survive_owned_restoration() -> Result {
    let original = single(constant()?, -0.0)?;
    let package = original.snapshot_data()?;
    let mut destination = single(constant()?, 1.0)?;
    destination.restore_data(&package)?;
    assert_eq!(
        destination.states()?[0].volume_m3.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        destination.element_ledgers()?[0]
            .initial_volume_m3
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(destination.outputs()?[0].stage_m, None);
    Ok(())
}

#[test]
fn immutable_curve_datum_provenance_and_support_identity_reject_changes() -> Result {
    let package = history()?.snapshot_data()?;
    let mut destination = single(constant()?, 0.0)?;
    let before = destination.snapshot_data()?;
    for mode in 0..7 {
        let mut changed = package.clone();
        let element = &mut changed.elements[0];
        match mode {
            0 => element.id = "different-pool".into(),
            1 => element.storage_support_identity = "different-storage-support".into(),
            2 => element.connected_pool_assertion = "different connected-pool evidence".into(),
            3 => {
                element.curve = LevelPoolCurve::new(
                    element.curve.knots().to_vec(),
                    "different datum".into(),
                    element.curve.provenance().into(),
                )?
            }
            4 => {
                element.curve = LevelPoolCurve::new(
                    element.curve.knots().to_vec(),
                    element.curve.vertical_reference().into(),
                    "different curve provenance".into(),
                )?
            }
            5 => {
                let mut knots = element.curve.knots().to_vec();
                knots[0].volume_m3 = -0.0;
                element.curve = LevelPoolCurve::new(
                    knots,
                    element.curve.vertical_reference().into(),
                    element.curve.provenance().into(),
                )?;
            }
            6 => changed.config.volume_absolute = changed.config.volume_absolute.next_up(),
            _ => return Err("unknown definition mutation".into()),
        }
        assert!(
            destination.restore_data(&changed).is_err(),
            "definition mutation={mode}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn incompatible_state_ledger_time_and_history_counts_reject_atomically() -> Result {
    let package = history()?.snapshot_data()?;
    let changes: &[fn(&mut LevelPoolSnapshotData)] = &[
        |data| data.time = f64::NAN,
        |data| data.time = -1.0,
        |data| data.time = 1.5,
        |data| data.states.clear(),
        |data| data.states[0].volume_m3 = -1.0,
        |data| data.states[0].volume_m3 = 5.0,
        |data| data.element_ledgers.clear(),
        |data| data.element_ledgers[0].element = usize::MAX,
        |data| data.element_ledgers[0].volume_m3 += 0.01,
        |data| data.element_ledgers[0].cumulative_incoming_volume_m3 = -1.0,
        |data| data.element_ledgers[0].cumulative_outgoing_volume_m3 = f64::INFINITY,
        |data| data.element_ledgers[0].incoming_roundoff_bound_m3 = -1.0,
        |data| data.element_ledgers[0].outgoing_roundoff_bound_m3 = f64::NAN,
        |data| data.element_ledgers[0].residual_m3 = 1.0,
        |data| data.accepted_steps = 0,
        |data| data.last_step = None,
    ];
    let mut destination = single(constant()?, 0.0)?;
    let before = destination.snapshot_data()?;
    for (index, change) in changes.iter().enumerate() {
        let mut changed = package.clone();
        change(&mut changed);
        assert!(
            destination.restore_data(&changed).is_err(),
            "package mutation={index}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn original_exchange_and_twosum_witnesses_are_recomputed_on_import() -> Result {
    let package = history()?.snapshot_data()?;
    let mut destination = single(constant()?, 0.0)?;
    let before = destination.snapshot_data()?;
    for mode in 0..22 {
        let mut changed = package.clone();
        let step = changed.last_step.as_mut().ok_or("missing accepted step")?;
        let receipt = step
            .receipts
            .first_mut()
            .ok_or("missing accepted receipt")?;
        match mode {
            0 => receipt.old_volume_m3 = -1.0,
            1 => receipt.incoming_volume_m3 += 0.125,
            2 => receipt.outgoing_volume_m3 = 2.0,
            3 => receipt.remaining_volume_m3 = receipt.remaining_volume_m3.next_up(),
            4 => receipt.outgoing_witness_m3 = receipt.outgoing_witness_m3.next_up(),
            5 => receipt.incoming_witness_m3 = receipt.incoming_witness_m3.next_up(),
            6 => receipt.water_residual_m3 += 1.0,
            7 => receipt.witness_consistency_m3 = 1.0,
            8 => receipt.witness_consistency_bound_m3 = -1.0,
            9 => receipt.stage_m = None,
            10 => receipt.stage_m = Some(-1.0),
            11 => receipt.reconstructed_volume_m3 += 1.0,
            12 => receipt.inverse_residual_m3 = 1.0,
            13 => receipt.cumulative_incoming_before_m3 += 0.1,
            14 => receipt.cumulative_incoming_after_m3 += 1.0,
            15 => receipt.cumulative_incoming_witness_m3 += 0.1,
            16 => receipt.incoming_bound_before_m3 = f64::NAN,
            17 => receipt.outgoing_bound_after_m3 = -1.0,
            18 => step.start = -1.0,
            19 => step.dt *= 2.0,
            20 => step.receipts.clear(),
            21 => receipt.element = usize::MAX,
            _ => return Err("unknown receipt mutation".into()),
        }
        assert!(
            destination.restore_data(&changed).is_err(),
            "receipt mutation={mode}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn zero_history_requires_initial_current_bits_and_no_cumulative_state() -> Result {
    let original = single(constant()?, 1.0)?;
    let package = original.snapshot_data()?;
    let mut changed = package.clone();
    let ledger = &mut changed.element_ledgers[0];
    ledger.initial_volume_m3 = 1.0_f64.next_down();
    ledger.residual_m3 = ledger.volume_m3 - ledger.initial_volume_m3;
    assert!(ledger.residual_m3.abs() < 1e-9);
    let mut destination = single(constant()?, 0.0)?;
    let before = destination.snapshot_data()?;
    assert!(destination.restore_data(&changed).is_err());
    assert_eq!(destination.snapshot_data()?, before);
    let mut changed = package;
    changed.accepted_steps = 1;
    assert!(destination.restore_data(&changed).is_err());
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

#[test]
fn imported_previous_storage_requires_a_resolvable_wet_stage() -> Result {
    let source = single(unit(2.0)?, 2.0)?;
    let source = source.trial_to(1.0, &[input(0.0, 1.5)])?.into_next();
    let source = source.trial_to(2.0, &[input(1.5, 0.0)])?.into_next();
    let mut package = source.snapshot_data()?;
    let target_curve = curve(&[(0.0, 1e16), (2.0, 1e16 + 2.0)])?;
    package.elements[0].curve = target_curve.clone();
    let receipt = package
        .last_step
        .as_mut()
        .ok_or("missing accepted step")?
        .receipts
        .first_mut()
        .ok_or("missing final receipt")?;
    assert_eq!(receipt.old_volume_m3, 0.5);
    assert_eq!(receipt.volume_m3, 2.0);
    assert_eq!(receipt.reconstructed_volume_m3, 2.0);
    assert_eq!(receipt.water_residual_m3, 0.0);
    receipt.stage_m = Some(1e16 + 2.0);
    let mut target = single(target_curve, 2.0)?;
    let before = target.snapshot_data()?;
    let error = rejection(target.restore_data(&package))?;
    assert_eq!(error.element(), Some(0));
    assert!(
        error
            .to_string()
            .contains("positive storage cannot resolve wet stage above its origin"),
        "unexpected rejection: {error}"
    );
    assert_eq!(target.snapshot_data()?, before);
    Ok(())
}

#[test]
fn level_pool_package_carries_identity() -> Result {
    assert_eq!(LEVEL_POOL_SNAPSHOT_SCHEMA, "hydr-level-pool-checkpoint-1");
    assert_eq!(LEVEL_POOL_CONTRACT_VERSION, "hydr-level-pool-numr-cpu-1");
    let package = history()?.snapshot_data()?;
    assert_eq!(package.schema, LEVEL_POOL_SNAPSHOT_SCHEMA);
    assert_eq!(package.contract, LEVEL_POOL_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn level_pool_import_rejects_foreign_schema() -> Result {
    let package = history()?.snapshot_data()?;
    let mut destination = single(constant()?, 0.0)?;
    let before = destination.snapshot_data()?;
    for found in ["hydr-level-pool-checkpoint-0", ""] {
        let mut changed = package.clone();
        changed.schema = found.into();
        let error = rejection(destination.restore_data(&changed))?;
        assert!(
            matches!(error, LevelPoolError::InvalidInput { .. }),
            "unexpected kind: {error}"
        );
        let message = error.to_string();
        assert!(
            message.contains(&format!("found \"{found}\"")),
            "missing found value: {message}"
        );
        assert!(
            message.contains(LEVEL_POOL_SNAPSHOT_SCHEMA),
            "missing expected schema: {message}"
        );
        assert!(
            message.contains("capture a new package"),
            "missing action: {message}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn level_pool_import_rejects_foreign_contract() -> Result {
    let mut changed = history()?.snapshot_data()?;
    changed.contract = "hydr-level-pool-numr-gpu-1".into();
    let mut destination = single(constant()?, 0.0)?;
    let before = destination.snapshot_data()?;
    let error = rejection(destination.restore_data(&changed))?;
    assert!(
        matches!(error, LevelPoolError::InvalidInput { .. }),
        "unexpected kind: {error}"
    );
    let message = error.to_string();
    assert!(message.contains("contract"), "missing contract: {message}");
    assert!(
        message.contains("hydr-level-pool-numr-gpu-1"),
        "missing found value: {message}"
    );
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

#[test]
fn level_pool_import_rejects_channel_schema() -> Result {
    let mut changed = history()?.snapshot_data()?;
    changed.schema = "hydr-channel-checkpoint-1".into();
    let mut destination = single(constant()?, 0.0)?;
    let before = destination.snapshot_data()?;
    let error = rejection(destination.restore_data(&changed))?;
    let message = error.to_string();
    assert!(
        message.contains("hydr-channel-checkpoint-1"),
        "missing found value: {message}"
    );
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}
