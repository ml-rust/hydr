use super::fixture::Result;
use super::source_fixture::{enabled, interval, pump_case, rates};

#[test]
fn trusted_and_owned_restart_preserve_source_history_and_repeat_continuation() -> Result {
    let (setup, initial) = pump_case([4.0, 2.0])?;
    let forcing = interval(vec![rates(1.0, 0.25, 0.5), rates(0.0, 0.0, 0.0)]);
    let mut batch = enabled(setup.clone(), initial.clone())?
        .trial_with_sources_to(0.25, &forcing)?
        .into_next();
    let trusted = batch.snapshot();
    let owned = batch.snapshot_data()?;
    assert_eq!(trusted.data()?, owned);
    let expected = batch
        .trial_with_sources_to(0.5, &forcing)?
        .into_next()
        .snapshot_data()?;
    batch = batch.trial_with_sources_to(0.5, &forcing)?.into_next();
    batch.restore(&trusted)?;
    assert_eq!(batch.snapshot_data()?, owned);
    assert_eq!(
        batch
            .trial_with_sources_to(0.5, &forcing)?
            .into_next()
            .snapshot_data()?,
        expected
    );

    let mut cold = enabled(setup, initial)?;
    cold.restore_data(&owned)?;
    assert_eq!(cold.snapshot_data()?, owned);
    assert_eq!(
        cold.trial_with_sources_to(0.5, &forcing)?
            .into_next()
            .snapshot_data()?,
        expected
    );
    Ok(())
}

#[test]
fn altered_forcing_policies_and_source_accounting_reject_atomically() -> Result {
    let (setup, initial) = pump_case([4.0, 2.0])?;
    let forcing = interval(vec![rates(1.0, 0.25, 0.5), rates(0.0, 0.0, 0.0)]);
    let mut batch = enabled(setup, initial)?
        .trial_with_sources_to(0.25, &forcing)?
        .into_next();
    let original = batch.snapshot_data()?;
    for mode in 0..9 {
        let mut altered = original.clone();
        match mode {
            0 => {
                altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing coupled receipt")?
                    .source_interval = None
            }
            1..=4 => {
                let interval = altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing coupled receipt")?
                    .source_interval
                    .as_mut()
                    .ok_or("Missing retained source interval")?;
                match mode {
                    1 => interval.start_s = 0.125,
                    2 => interval.end_s = 0.125,
                    3 => interval.sources[0].rain_rate_m_per_s = 0.5,
                    4 => interval.sources.clear(),
                    _ => return Err("Unknown interval mutation".into()),
                }
            }
            5 => altered.pools.source_policies = None,
            6 => {
                altered
                    .start_pools
                    .as_mut()
                    .ok_or("Missing native start")?
                    .source_policies = None
            }
            7 => {
                altered.pools.element_ledgers[0]
                    .sources
                    .as_mut()
                    .ok_or("Missing source accounting")?
                    .cumulative_rain_volume_m3 += 0.25
            }
            8 => {
                altered
                    .pools
                    .last_step
                    .as_mut()
                    .ok_or("Missing native receipt")?
                    .receipts[0]
                    .source
                    .as_mut()
                    .ok_or("Missing source receipt")?
                    .rain_volume_m3 += 0.25
            }
            _ => return Err("Unknown checkpoint mutation".into()),
        }
        assert!(
            batch.restore_data(&altered).is_err(),
            "Accepted source checkpoint mutation={mode}"
        );
        assert_eq!(batch.snapshot_data()?, original);
    }
    Ok(())
}
