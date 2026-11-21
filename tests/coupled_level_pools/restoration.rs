use super::fixture::{Result, orifice_case};
use hydr::coupling::{
    COUPLED_LEVEL_POOL_CONTRACT_VERSION, COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA, CoupledLevelPoolBatch,
    CoupledLevelPoolError, CoupledLevelPoolSnapshotData,
};
use hydr::storage::LEVEL_POOL_SNAPSHOT_SCHEMA;

fn advanced() -> Result<CoupledLevelPoolBatch<crate::runtime::Runtime>> {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    Ok(CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?
    .trial_to(0.25)?
    .into_next())
}

/// Import `changed`, then check the input error names every `needle` and the destination is unchanged.
fn assert_rejected(changed: &CoupledLevelPoolSnapshotData, needles: &[&str]) -> Result {
    let mut destination = advanced()?;
    let before = destination.snapshot_data()?;
    let Err(error) = destination.restore_data(changed) else {
        return Err("foreign identity was accepted".into());
    };
    assert!(
        matches!(error, CoupledLevelPoolError::Input(_)),
        "unexpected kind: {error}"
    );
    let message = error.to_string();
    for needle in needles {
        assert!(message.contains(needle), "missing {needle:?}: {message}");
    }
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

#[test]
fn coupled_level_pool_package_carries_identity() -> Result {
    assert_eq!(
        COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA,
        "hydr-coupled-level-pool-checkpoint-2"
    );
    assert_eq!(
        COUPLED_LEVEL_POOL_CONTRACT_VERSION,
        "hydr-coupled-level-pool-numr-cpu-2"
    );
    let package = advanced()?.snapshot_data()?;
    assert_eq!(package.schema, COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA);
    assert_eq!(package.contract, COUPLED_LEVEL_POOL_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn coupled_level_pool_import_rejects_foreign_schema() -> Result {
    let package = advanced()?.snapshot_data()?;
    for found in [
        "hydr-coupled-level-pool-checkpoint-0",
        "hydr-coupled-level-pool-checkpoint-1",
        "",
    ] {
        let mut changed = package.clone();
        changed.schema = found.into();
        assert_rejected(
            &changed,
            &[
                "coupled level pool checkpoint schema",
                &format!("found \"{found}\""),
                COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA,
                "capture a new package",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn coupled_level_pool_import_rejects_foreign_contract() -> Result {
    let mut changed = advanced()?.snapshot_data()?;
    changed.contract = "hydr-coupled-level-pool-numr-gpu-1".into();
    assert_rejected(
        &changed,
        &[
            "coupled level pool checkpoint contract",
            "found \"hydr-coupled-level-pool-numr-gpu-1\"",
            COUPLED_LEVEL_POOL_CONTRACT_VERSION,
        ],
    )
}

#[test]
fn coupled_level_pool_import_rejects_nested_start_pools_schema() -> Result {
    let mut changed = advanced()?.snapshot_data()?;
    let start = changed
        .start
        .as_mut()
        .ok_or("advanced package lacks its start frame")?;
    start.pools.schema = "hydr-level-pool-checkpoint-0".into();
    assert_rejected(
        &changed,
        &[
            "coupled level pool checkpoint start.pools",
            "level pool checkpoint schema",
            "found \"hydr-level-pool-checkpoint-0\"",
            LEVEL_POOL_SNAPSHOT_SCHEMA,
        ],
    )
}

#[test]
fn trusted_and_owned_restart_reproduce_complete_continuation() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let initial_batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.clone(),
        initial.clone(),
    )?;
    let mut batch = initial_batch.trial_to(0.25)?.into_next();
    let trusted = batch.snapshot();
    let owned = batch.snapshot_data()?;
    assert_eq!(trusted.data()?, owned);
    let expected = batch.trial_to(0.5)?.into_next().snapshot_data()?;

    batch = batch.trial_to(0.5)?.into_next();
    batch.restore(&trusted)?;
    assert_eq!(batch.snapshot_data()?, owned);
    assert_eq!(batch.trial_to(0.5)?.into_next().snapshot_data()?, expected);

    let mut reconstructed = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    reconstructed.restore_data(&owned)?;
    assert_eq!(reconstructed.snapshot_data()?, owned);
    assert_eq!(
        reconstructed.trial_to(0.5)?.into_next().snapshot_data()?,
        expected
    );
    Ok(())
}

#[test]
fn forged_transfer_head_and_support_records_reject_atomically() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let mut batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?
    .trial_to(0.25)?
    .into_next();
    let original = batch.snapshot_data()?;
    for mode in 0..9 {
        let mut altered = original.clone();
        let last = altered
            .last_step
            .as_mut()
            .ok_or("Missing coupled receipt")?;
        match mode {
            0 => last.transfers[0].signed_volume_m3 += 0.25,
            1 => last.transfers[0].proposed_volume_m3 += 0.25,
            2 => last.transfers[0].donor_limit_factor = 0.5,
            3 => last.transfers[0].midpoint_stage_a_m = None,
            4 => last.iteration.end_stages_m[0] += 0.25,
            5 => last.iteration.pool_wet[0] = false,
            6 => last.iteration.signed_volumes_m3[0] += 0.25,
            7 => last.head_residual_m = f64::NAN,
            8 => last.iterations = altered.setup.config.iteration_limit,
            _ => return Err("Unknown checkpoint mutation".into()),
        }
        assert!(
            batch.restore_data(&altered).is_err(),
            "Accepted altered receipt mode={mode}"
        );
        assert_eq!(batch.snapshot_data()?, original);
    }
    Ok(())
}

#[test]
fn missing_or_altered_interval_start_frames_reject_atomically() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let mut batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?
    .trial_to(0.25)?
    .into_next();
    let original = batch.snapshot_data()?;
    for mode in 0..6 {
        let mut altered = original.clone();
        match mode {
            0 => altered.start = None,
            1 => altered.last_step = None,
            2 => {
                altered.start = None;
                altered.last_step = None;
            }
            3..=5 => {
                let start = altered
                    .start
                    .as_mut()
                    .ok_or("Missing retained interval start")?;
                match mode {
                    3 => start.pools.time = 0.125,
                    4 => start.pools.states[0].volume_m3 += 0.25,
                    5 => start.pools.accepted_steps += 1,
                    _ => return Err("Unknown retained start mutation".into()),
                }
            }
            _ => return Err("Unknown checkpoint frame mutation".into()),
        }
        assert!(
            batch.restore_data(&altered).is_err(),
            "Accepted altered start frame mode={mode}"
        );
        assert_eq!(batch.snapshot_data()?, original);
    }
    Ok(())
}

#[test]
fn different_connection_or_numerical_definition_rejects_trusted_restore() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let source = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.clone(),
        initial.clone(),
    )?
    .trial_to(0.25)?
    .into_next();
    let checkpoint = source.snapshot();
    for mode in 0..2 {
        let mut changed = setup.clone();
        if mode == 0 {
            changed.connections[0]
                .endpoint_a
                .physical_connection_assertion
                .push_str(" different");
        } else {
            changed.config.head_absolute_m = 2e-10;
        }
        let mut destination = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            changed,
            initial.clone(),
        )?;
        let before = destination.snapshot_data()?;
        assert!(destination.restore(&checkpoint).is_err());
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}
