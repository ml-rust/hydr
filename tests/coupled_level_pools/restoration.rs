use super::fixture::{Result, orifice_case};
use hydr::coupling::CoupledLevelPoolBatch;

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
            0 => altered.start_pools = None,
            1 => altered.last_step = None,
            2 => {
                altered.start_pools = None;
                altered.last_step = None;
            }
            3..=5 => {
                let start = altered
                    .start_pools
                    .as_mut()
                    .ok_or("Missing retained interval start")?;
                match mode {
                    3 => start.time = 0.125,
                    4 => start.states[0].volume_m3 += 0.25,
                    5 => start.accepted_steps += 1,
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
