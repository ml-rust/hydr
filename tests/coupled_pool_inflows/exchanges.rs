use super::fixture::{Result, assemble, near, orifice_case};
use super::inflow_fixture::{binding, forcing, rate};
use super::laws::{pump, running};
use super::source_fixture::{Batch, enabled, inactive_case, interval, policies, pump_case, rates};

#[test]
fn two_prescribed_inflows_add_their_original_interval_volumes() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.config.maximum_step_s = 2.0;
    setup.external_inflows = vec![binding("left", "a"), binding("right", "b")];
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    let next = batch
        .trial_with_forcing_to(2.0, &forcing(vec![rate("left", 0.25), rate("right", 0.75)]))?
        .into_next();
    assert_eq!(batch.snapshot_data()?, before);
    assert_eq!(next.time(), 2.0);
    near(next.pools().states()?[0].volume_m3, 1.5);
    near(next.pools().states()?[1].volume_m3, 2.5);
    let ledgers = next.external_inflows()?;
    near(ledgers[0].cumulative_volume_m3, 0.5);
    near(ledgers[1].cumulative_volume_m3, 1.5);
    near(
        ledgers
            .iter()
            .map(|ledger| ledger.cumulative_volume_m3)
            .sum(),
        2.0,
    );
    assert!(
        next.last_step()?
            .ok_or("Missing external interval")?
            .mass_residual_m3
            .abs()
            <= 1e-9 + 4e-12
    );
    Ok(())
}

#[test]
fn multiple_boundary_permutations_preserve_named_inputs_and_pool_storage() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.config.maximum_step_s = 2.0;
    setup.external_inflows = vec![binding("small", "a"), binding("large", "a")];
    let mut reversed = setup.clone();
    reversed.external_inflows.reverse();
    let first = Batch::new(crate::runtime::client(), setup, initial.clone())?
        .trial_with_forcing_to(2.0, &forcing(vec![rate("large", 0.5), rate("small", 0.25)]))?
        .into_next();
    let second = Batch::new(crate::runtime::client(), reversed, initial)?
        .trial_with_forcing_to(2.0, &forcing(vec![rate("small", 0.25), rate("large", 0.5)]))?
        .into_next();
    assert_eq!(
        first.pools().snapshot_data()?,
        second.pools().snapshot_data()?
    );
    near(first.pools().states()?[0].volume_m3, 2.5);
    let other = second.external_inflows()?;
    let ledgers = first.external_inflows()?;
    near(
        ledgers
            .iter()
            .map(|ledger| ledger.cumulative_volume_m3)
            .sum(),
        1.5,
    );
    for a in ledgers {
        let b = other
            .iter()
            .find(|b| b.boundary_id == a.boundary_id)
            .ok_or("Missing permuted boundary ledger")?;
        assert_eq!(&a, b);
    }
    Ok(())
}

#[test]
fn external_water_cannot_finance_same_interval_structure_withdrawals() -> Result {
    let (mut setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[1.0, 1.0],
        vec![pump("pump", "a", "b", &[(-16.0, 2.0), (16.0, 2.0)])?],
        vec![running("pump")],
    )?;
    setup.external_inflows = vec![binding("donor", "a")];
    let next = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(1.0, &forcing(vec![rate("donor", 1.0)]))?
        .into_next();
    let step = next.last_step()?.ok_or("Missing donor-limited interval")?;
    near(step.transfers[0].proposed_volume_m3, 2.0);
    assert!(step.transfers[0].signed_volume_m3 <= 1.0);
    near(step.transfers[0].signed_volume_m3, 1.0);
    near(next.pools().states()?[0].volume_m3, 1.0);
    near(next.pools().states()?[1].volume_m3, 2.0);
    Ok(())
}

#[test]
fn prescribed_inflows_follow_local_sources_and_structure_allocation() -> Result {
    let (mut setup, initial) = pump_case([4.0, 2.0])?;
    setup.external_inflows = vec![binding("left", "a"), binding("right", "b")];
    let mut forcing = forcing(vec![rate("left", 0.5), rate("right", 0.25)]);
    forcing.sources = Some(vec![rates(1.0, 0.25, 0.5), rates(0.0, 0.0, 0.0)]);
    let next = enabled(setup, initial)?
        .trial_with_forcing_to(1.0, &forcing)?
        .into_next();
    near(next.pools().states()?[0].volume_m3, 3.75);
    near(next.pools().states()?[1].volume_m3, 3.25);
    near(
        next.external_inflows()?
            .iter()
            .map(|ledger| ledger.cumulative_volume_m3)
            .sum(),
        0.75,
    );
    near(
        next.last_step()?
            .ok_or("Missing combined forcing interval")?
            .transfers[0]
            .signed_volume_m3,
        1.0,
    );
    Ok(())
}

#[test]
fn bottom_loss_and_evaporation_cannot_consume_later_external_water() -> Result {
    let (mut setup, initial) = inactive_case([0.0, 1.0])?;
    setup.external_inflows = vec![binding("dry", "a")];
    let mut forcing = forcing(vec![rate("dry", 1.0)]);
    forcing.sources = Some(vec![rates(0.0, 1.0, 1.0), rates(0.0, 0.0, 0.0)]);
    let next = enabled(setup, initial)?
        .trial_with_forcing_to(1.0, &forcing)?
        .into_next();
    near(next.pools().states()?[0].volume_m3, 1.0);
    let native = next
        .pools()
        .last_step()?
        .ok_or("Missing native loss accounting")?;
    assert_eq!(native.bottom_loss_volume_m3, 0.0);
    assert_eq!(native.evaporation_volume_m3, 0.0);
    Ok(())
}

#[test]
fn shortening_recalculates_external_products_at_the_actual_duration() -> Result {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.config.iteration_limit = 8;
    setup.external_inflows = vec![binding("left", "a"), binding("right", "b")];
    let next = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(1.0, &forcing(vec![rate("left", 0.25), rate("right", 0.25)]))?
        .into_next();
    let dt = next.time();
    assert!(dt > 0.0 && dt < 1.0);
    let transfer = dt * 0.5 * ((dt * dt + 8.0).sqrt() - dt);
    near(
        next.last_step()?
            .ok_or("Missing shortened external interval")?
            .transfers[0]
            .signed_volume_m3,
        transfer,
    );
    near(
        next.pools().states()?[0].volume_m3,
        4.0 + 0.25 * dt - transfer,
    );
    near(
        next.pools().states()?[1].volume_m3,
        2.0 + 0.25 * dt + transfer,
    );
    for receipt in next
        .last_step()?
        .ok_or("Missing external receipts")?
        .external_inflows
    {
        assert_eq!(receipt.dt_s.to_bits(), dt.to_bits());
        near(receipt.volume_m3, 0.25 * dt);
    }
    Ok(())
}

#[test]
fn explicit_zero_boundaries_preserve_closed_and_source_only_hydraulic_bits() -> Result {
    for with_sources in [false, true] {
        let (mut setup, initial) = orifice_case([4.0, 2.0])?;
        if with_sources {
            setup.source_policies = Some(policies(2));
        }
        let baseline = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?;
        setup.external_inflows = vec![binding("zero", "a")];
        let extended = Batch::new(crate::runtime::client(), setup, initial)?;
        let mut zero = forcing(vec![rate("zero", 0.0)]);
        let baseline = if with_sources {
            zero.sources = Some(vec![rates(0.25, 0.0, 0.0); 2]);
            baseline
                .trial_with_forcing_to(0.25, &interval(vec![rates(0.25, 0.0, 0.0); 2]))?
                .into_next()
        } else {
            baseline.trial_to(0.25)?.into_next()
        };
        let extended = extended.trial_with_forcing_to(0.25, &zero)?.into_next();
        assert_eq!(
            baseline.pools().snapshot_data()?,
            extended.pools().snapshot_data()?
        );
        let a = baseline.last_step()?.ok_or("Missing baseline interval")?;
        let b = extended
            .last_step()?
            .ok_or("Missing zero-boundary interval")?;
        assert_eq!(a.transfers, b.transfers);
        assert_eq!(a.iteration, b.iteration);
        assert_eq!(a.mass_residual_m3.to_bits(), b.mass_residual_m3.to_bits());
    }
    Ok(())
}
