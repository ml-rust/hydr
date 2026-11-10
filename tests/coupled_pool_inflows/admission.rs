use super::fixture::{Result, near};
use super::inflow_fixture::{binding, forcing, rate};
use super::source_fixture::{Batch, inactive_case, pump_case};

#[test]
fn external_boundaries_require_complete_unique_identity_and_ownership_evidence() -> Result {
    for mode in 0..9 {
        let (mut setup, initial) = inactive_case([1.0, 1.0])?;
        setup.external_inflows = vec![binding("left", "a"), binding("right", "b")];
        match mode {
            0 => setup.external_inflows[1].boundary_id = "left".into(),
            1 => setup.external_inflows[1].input_id = setup.external_inflows[0].input_id.clone(),
            2 => setup.external_inflows[0].pool_id = "missing".into(),
            3 => setup.external_inflows[0].boundary_assertion.clear(),
            4 => setup.external_inflows[0].ownership_assertion.clear(),
            5 => setup.external_inflows[0].physical_support_id.clear(),
            6 => setup.external_inflows[0].boundary_id.clear(),
            7 => setup.external_inflows[0].input_id.clear(),
            8 => setup.external_inflows[1].ownership_assertion = "   ".into(),
            _ => return Err("Unknown boundary mutation".into()),
        }
        assert!(
            Batch::new(crate::runtime::client(), setup, initial).is_err(),
            "Accepted boundary mutation={mode}"
        );
    }
    Ok(())
}

#[test]
fn distinct_owned_inputs_can_share_a_declared_physical_support() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    let mut left = binding("left", "a");
    let mut right = binding("right", "a");
    left.physical_support_id = "shared-support".into();
    right.physical_support_id = "shared-support".into();
    left.ownership_assertion =
        "First independent selected input excludes the other input, represented storage, and local rainfall".into();
    right.ownership_assertion =
        "Second independent selected input excludes the other input, represented storage, and local rainfall".into();
    setup.external_inflows = vec![left, right];
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let next = batch
        .trial_with_forcing_to(1.0, &forcing(vec![rate("left", 0.25), rate("right", 0.5)]))?
        .into_next();
    near(next.pools().states()?[0].volume_m3, 1.75);
    let ledgers = next.external_inflows()?;
    assert_eq!(ledgers[0].boundary_id, "left");
    assert_eq!(ledgers[1].boundary_id, "right");
    near(ledgers[0].cumulative_volume_m3, 0.25);
    near(ledgers[1].cumulative_volume_m3, 0.5);
    Ok(())
}

#[test]
fn complete_forcing_including_zero_rates_is_required_before_any_trial() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.external_inflows = vec![binding("left", "a"), binding("right", "b")];
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    assert!(batch.trial_to(0.5).is_err());
    for mode in 0..10 {
        let mut interval = forcing(vec![rate("left", 0.0), rate("right", 0.0)]);
        match mode {
            0 => interval.inflows.clear(),
            1 => {
                interval.inflows.pop();
            }
            2 => interval.inflows[1] = rate("left", 0.0),
            3 => interval.inflows[1] = rate("unknown", 0.0),
            4 => interval.inflows[0].input_id = "different-input".into(),
            5 => interval.inflows[0].rate_m3_per_s = -1.0,
            6 => interval.inflows[0].rate_m3_per_s = f64::INFINITY,
            7 => interval.inflows[0].rate_m3_per_s = f64::NAN,
            8 => interval.start_s = 0.25,
            9 => interval.end_s = 0.25,
            _ => return Err("Unknown external forcing mutation".into()),
        }
        assert!(
            batch.trial_with_forcing_to(0.5, &interval).is_err(),
            "Accepted external forcing mutation={mode}"
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    let next = batch
        .trial_with_forcing_to(0.5, &forcing(vec![rate("right", 0.0), rate("left", 0.0)]))?
        .into_next();
    for ledger in next.external_inflows()? {
        assert_eq!(ledger.cumulative_volume_m3.to_bits(), 0.0_f64.to_bits());
        assert_eq!(ledger.roundoff_bound_m3.to_bits(), 0.0_f64.to_bits());
    }
    let accepted = next.snapshot_data()?;
    let mut gap = forcing(vec![rate("left", 0.0), rate("right", 0.0)]);
    gap.start_s = 0.75;
    assert!(next.trial_with_forcing_to(1.0, &gap).is_err());
    assert_eq!(next.snapshot_data()?, accepted);
    Ok(())
}

#[test]
fn unrepresentable_products_and_absorbed_external_assembly_reject_atomically() -> Result {
    for mode in 0..4 {
        let (mut setup, initial) = if mode == 3 {
            pump_case([4.0, 1.0])?
        } else {
            inactive_case([1.0, 1.0])?
        };
        let (dt, records) = match mode {
            0 => (2.0, vec![rate("external", f64::MAX)]),
            1 => (0.5, vec![rate("external", f64::from_bits(1))]),
            2 => (
                1.0,
                vec![rate("external", 1.0), rate("small", f64::EPSILON / 4.0)],
            ),
            3 => (1.0, vec![rate("external", f64::EPSILON / 4.0)]),
            _ => return Err("Unknown inflow arithmetic fixture".into()),
        };
        setup.config.minimum_step_s = dt;
        setup.config.maximum_step_s = dt;
        setup.config.retry_limit = 0;
        let pool = if mode == 3 { "b" } else { "a" };
        setup.external_inflows = vec![binding("external", pool)];
        if mode == 2 {
            setup.external_inflows.push(binding("small", pool));
        }
        let batch = Batch::new(crate::runtime::client(), setup, initial)?;
        let before = batch.snapshot_data()?;
        assert!(
            batch.trial_with_forcing_to(dt, &forcing(records)).is_err(),
            "Accepted unrepresentable external volume mode={mode}"
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}
