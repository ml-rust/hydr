use super::fixture::{Batch, Result, exact_case, forcing, shortened_case};
use hydr::coupling::RunoffPoolError;
use hydr::runoff::Subarea;

#[test]
fn outlet_support_ownership_and_complete_pairing_are_required() -> Result {
    for mode in 0..8 {
        let (mut setup, initial) = exact_case()?;
        match mode {
            0 => setup.outlets.clear(),
            1 => setup.outlets.push(setup.outlets[0].clone()),
            2 => setup.outlets[0].subcatchment_id = "unknown".into(),
            3 => setup.outlets[0].subarea = Subarea::Pervious,
            4 => setup.outlets[0].boundary_id = "unknown".into(),
            5 => setup.outlets[0].free_outlet_assertion.clear(),
            6 => setup.outlets[0].nonoverlap_assertion.clear(),
            7 => setup.pools.external_inflows.clear(),
            _ => return Err("Unknown outlet mutation".into()),
        }
        assert!(
            Batch::new(crate::runtime::client(), setup, initial).is_err(),
            "Accepted outlet mutation={mode}"
        );
    }
    Ok(())
}

#[test]
fn invalid_forcing_and_targets_preserve_both_children_and_parent_receipt() -> Result {
    let (setup, initial) = exact_case()?;
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    for mode in 0..7 {
        let mut forcing = forcing(1, 0.25, 0.0);
        match mode {
            0 => forcing.start_s = 0.25,
            1 => forcing.end_s = 0.25,
            2 => forcing.subcatchments.clear(),
            3 => forcing.subcatchments[0].rain_rate_m_per_s = -1.0,
            4 => forcing.subcatchments[0].potential_evaporation_m_per_s = f64::NAN,
            5 => forcing.pool_sources = Some(vec![]),
            6 => forcing.end_s = f64::INFINITY,
            _ => return Err("Unknown meteorology mutation".into()),
        }
        assert!(
            batch.trial_with_forcing_to(0.5, &forcing).is_err(),
            "Accepted forcing mutation={mode}"
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            batch
                .trial_with_forcing_to(end, &forcing(1, 0.25, 0.0))
                .is_err()
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    let next = batch
        .trial_with_forcing_to(0.5, &forcing(1, 0.25, 0.0))?
        .into_next();
    let accepted = next.snapshot_data()?;
    let mut gap = forcing(1, 0.25, 0.0);
    gap.start_s = 0.75;
    assert!(next.trial_with_forcing_to(1.0, &gap).is_err());
    assert_eq!(next.snapshot_data()?, accepted);
    Ok(())
}

#[test]
fn exhausted_common_retries_discard_completed_runoff_and_pool_trials() -> Result {
    let (mut setup, initial, forcing) = shortened_case()?;
    setup.config.retry_limit = 0;
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    let error = batch
        .trial_with_forcing_to(1.0, &forcing)
        .err()
        .ok_or("Unconverged common interval was accepted")?;
    assert!(
        matches!(error, RunoffPoolError::Unresolved { retries: 0, .. }),
        "{error}"
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}
