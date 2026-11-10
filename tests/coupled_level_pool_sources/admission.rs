use super::fixture::{Result, orifice_case};
use super::source_fixture::{Batch, enabled, interval, policies, rates};

#[test]
fn source_policies_require_complete_explicit_physical_evidence() -> Result {
    for mode in 0..4 {
        let (setup, initial) = orifice_case([4.0, 2.0])?;
        let mut policy = policies(2);
        match mode {
            0 => policy.clear(),
            1 => {
                policy.pop();
            }
            2 => policy[0].dry_basal_footprint_assertion.clear(),
            3 => policy[1].bottom_loss_provenance.clear(),
            _ => return Err("Unknown policy mutation".into()),
        }
        assert!(
            Batch::new_with_sources(crate::runtime::client(), setup, initial, policy).is_err(),
            "Accepted source policy mutation={mode}"
        );
    }
    Ok(())
}

#[test]
fn source_calls_require_enabled_policies_and_explicit_forcing_even_for_zero_rates() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let closed = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?;
    let original = closed.snapshot_data()?;
    assert!(closed.source_policies().is_none());
    assert!(
        closed
            .trial_with_sources_to(0.25, &interval(vec![rates(0.0, 0.0, 0.0); 2]))
            .is_err()
    );
    assert_eq!(closed.snapshot_data()?, original);

    let enabled = enabled(setup, initial)?;
    let original = enabled.snapshot_data()?;
    assert_eq!(enabled.source_policies(), Some(policies(2).as_slice()));
    assert!(enabled.trial_to(0.25).is_err());
    assert_eq!(enabled.snapshot_data()?, original);
    Ok(())
}

#[test]
fn invalid_interval_coverage_and_incomplete_rates_reject_atomically() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let batch = enabled(setup, initial)?;
    let before = batch.snapshot_data()?;
    for mode in 0..11 {
        let mut forcing = interval(vec![rates(0.0, 0.0, 0.0); 2]);
        match mode {
            0 => forcing.start_s = 0.125,
            1 => forcing.end_s = 0.125,
            2 => forcing.start_s = f64::NAN,
            3 => forcing.end_s = f64::INFINITY,
            4 => forcing.sources.clear(),
            5 => {
                forcing.sources.pop();
            }
            6 => forcing.sources.push(rates(0.0, 0.0, 0.0)),
            7 => forcing.sources[0].rain_rate_m_per_s = -1.0,
            8 => forcing.sources[1].bottom_loss_rate_m3_per_s = f64::NAN,
            9 => forcing.sources[0].potential_evaporation_rate_m_per_s = f64::INFINITY,
            10 => forcing.start_s = forcing.end_s,
            _ => return Err("Unknown forcing mutation".into()),
        }
        assert!(
            batch.trial_with_sources_to(0.25, &forcing).is_err(),
            "Accepted forcing mutation={mode}"
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn gaps_after_an_accepted_interval_do_not_advance_state() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let forcing = interval(vec![rates(0.25, 0.0, 0.0); 2]);
    let batch = enabled(setup, initial)?
        .trial_with_sources_to(0.25, &forcing)?
        .into_next();
    let before = batch.snapshot_data()?;
    let mut gap = forcing;
    gap.start_s = 0.5;
    assert!(batch.trial_with_sources_to(0.75, &gap).is_err());
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn unrepresentable_source_products_reject_without_partial_acceptance() -> Result {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.config.minimum_step_s = 2.0;
    setup.config.maximum_step_s = 2.0;
    setup.config.retry_limit = 0;
    let batch = enabled(setup, initial)?;
    let before = batch.snapshot_data()?;
    assert!(
        batch
            .trial_with_sources_to(2.0, &interval(vec![rates(f64::MAX, 0.0, 0.0); 2]))
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}
