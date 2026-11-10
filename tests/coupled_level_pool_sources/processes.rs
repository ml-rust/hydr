use super::fixture::{DATUM, Result, assemble, near, orifice_case};
use super::laws::{pump, running};
use super::source_fixture::{enabled, inactive_case, interval, pump_case, rates};
use hydr::coupling::CoupledLevelPoolBatch;
use hydr::storage::{LevelPoolCurve, LevelPoolKnot};

#[test]
fn rain_bottom_loss_and_evaporation_precede_shared_pump_withdrawal() -> Result {
    let (setup, initial) = pump_case([4.0, 2.0])?;
    let batch = enabled(setup, initial)?;
    let original = batch.snapshot_data()?;
    let forcing = interval(vec![rates(1.0, 0.25, 0.5), rates(0.0, 0.0, 0.0)]);
    let next = batch.trial_with_forcing_to(1.0, &forcing)?.into_next();
    assert_eq!(batch.snapshot_data()?, original);
    assert_eq!(next.time(), 1.0);
    let volumes = next.pools().states()?;
    near(volumes[0].volume_m3, 3.25);
    near(volumes[1].volume_m3, 3.0);
    let step = next.last_step()?.ok_or("Missing source-coupled interval")?;
    assert_eq!(step.forcing_interval, Some(forcing));
    near(step.transfers[0].signed_volume_m3, 1.0);
    assert!(step.mass_residual_m3.abs() <= 1e-9 + 1e-12 * 6.25);
    let native = next
        .pools()
        .last_step()?
        .ok_or("Missing native source receipt")?;
    near(native.rain_volume_m3, 1.0);
    near(native.bottom_loss_volume_m3, 0.25);
    near(native.evaporation_volume_m3, 0.5);
    assert_eq!(next.pools().accepted_steps(), 1);
    Ok(())
}

#[test]
fn interval_rain_can_supply_a_structure_withdrawal() -> Result {
    let (setup, initial) = pump_case([0.25, 2.0])?;
    let batch = enabled(setup, initial)?;
    let next = batch
        .trial_with_forcing_to(
            1.0,
            &interval(vec![rates(1.0, 0.0, 0.0), rates(0.0, 0.0, 0.0)]),
        )?
        .into_next();
    near(next.pools().states()?[0].volume_m3, 0.25);
    near(
        next.last_step()?
            .ok_or("Missing rain-funded transfer")?
            .transfers[0]
            .signed_volume_m3,
        1.0,
    );
    Ok(())
}

#[test]
fn capped_losses_remove_middle_pool_availability_before_incoming_credit() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0), ("c", 0.0)],
        &[4.0, 0.25, 1.0],
        vec![
            pump("ab", "a", "b", &[(-16.0, 1.0), (16.0, 1.0)])?,
            pump("bc", "b", "c", &[(-16.0, 1.0), (16.0, 1.0)])?,
        ],
        vec![running("ab"), running("bc")],
    )?;
    let batch = enabled(setup, initial)?;
    let next = batch
        .trial_with_forcing_to(
            1.0,
            &interval(vec![
                rates(0.0, 0.0, 0.0),
                rates(0.0, 1.0, 1.0),
                rates(0.0, 0.0, 0.0),
            ]),
        )?
        .into_next();
    for (actual, expected) in next.pools().states()?.iter().zip([3.0, 1.0, 1.0]) {
        near(actual.volume_m3, expected);
    }
    let step = next.last_step()?.ok_or("Missing capped source interval")?;
    near(step.transfers[0].signed_volume_m3, 1.0);
    assert_eq!(step.transfers[1].signed_volume_m3, 0.0);
    near(step.transfers[1].proposed_volume_m3, 1.0);
    assert_eq!(step.transfers[1].donor_limit_factor, 0.0);
    let native = next
        .pools()
        .last_step()?
        .ok_or("Missing capped native receipt")?;
    near(native.bottom_loss_volume_m3, 0.25);
    assert_eq!(native.evaporation_volume_m3, 0.0);
    Ok(())
}

#[test]
fn rain_area_uses_the_accepted_right_segment_before_crossing_knots() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.pools[0].curve = LevelPoolCurve::new(
        [(0.0, 0.0), (1.0, 1.0), (3.0, 2.0), (7.0, 3.0)]
            .into_iter()
            .map(|(volume_m3, stage_m)| LevelPoolKnot { volume_m3, stage_m })
            .collect(),
        DATUM.into(),
        "Synthetic changing-area storage with explicit right-segment knot policy".into(),
    )?;
    let batch = enabled(setup, initial)?;
    let next = batch
        .trial_with_forcing_to(
            1.0,
            &interval(vec![rates(1.5, 0.0, 0.0), rates(0.0, 0.0, 0.0)]),
        )?
        .into_next();
    let output = &next.pools().outputs()?[0];
    near(output.volume_m3, 4.0);
    near(output.stage_m.ok_or("Rain-filled pool has no stage")?, 2.25);
    let native = next
        .pools()
        .last_step()?
        .ok_or("Missing variable-area receipt")?;
    let source = native.receipts[0]
        .source
        .as_ref()
        .ok_or("Missing frozen-area evidence")?;
    near(source.frozen_area_m2, 2.0);
    near(source.rain_volume_m3, 3.0);
    Ok(())
}

#[test]
fn dry_basal_footprint_accepts_rain_without_inventing_an_initial_stage() -> Result {
    let (setup, initial) = inactive_case([0.0, 1.0])?;
    let batch = enabled(setup, initial)?;
    assert_eq!(batch.pools().outputs()?[0].stage_m, None);
    let next = batch
        .trial_with_forcing_to(
            1.0,
            &interval(vec![rates(0.5, 0.0, 0.0), rates(0.0, 0.0, 0.0)]),
        )?
        .into_next();
    near(next.pools().states()?[0].volume_m3, 0.5);
    near(
        next.pools().outputs()?[0]
            .stage_m
            .ok_or("Rain remains unsupported")?,
        0.5,
    );
    Ok(())
}

#[test]
fn retries_recalculate_rain_and_transfer_from_the_original_accepted_state() -> Result {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.config.iteration_limit = 8;
    let batch = enabled(setup, initial)?;
    let original = batch.snapshot_data()?;
    let forcing = interval(vec![rates(0.25, 0.0, 0.0); 2]);
    let next = batch.trial_with_forcing_to(1.0, &forcing)?.into_next();
    assert_eq!(batch.snapshot_data()?, original);
    let dt = next.time();
    assert!(dt > 0.0 && dt < 1.0);
    let expected = dt * 0.5 * ((dt * dt + 8.0).sqrt() - dt);
    let step = next
        .last_step()?
        .ok_or("Missing shortened source interval")?;
    assert!(step.retries > 0);
    near(step.transfers[0].signed_volume_m3, expected);
    near(
        next.pools().states()?[0].volume_m3,
        4.0 + 0.25 * dt - expected,
    );
    near(
        next.pools().states()?[1].volume_m3,
        2.0 + 0.25 * dt + expected,
    );
    let native = next
        .pools()
        .last_step()?
        .ok_or("Missing retried native interval")?;
    near(native.rain_volume_m3, 0.5 * dt);
    assert_eq!(next.pools().accepted_steps(), 1);
    assert_eq!(step.forcing_interval, Some(forcing));
    Ok(())
}

#[test]
fn explicit_zero_forcing_retains_closed_hydraulic_bits_and_zero_counters() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let closed = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.clone(),
        initial.clone(),
    )?;
    let enabled = enabled(setup, initial)?;
    let closed = closed.trial_to(0.25)?.into_next();
    let enabled = enabled
        .trial_with_forcing_to(0.25, &interval(vec![rates(0.0, 0.0, 0.0); 2]))?
        .into_next();
    for (a, b) in closed
        .pools()
        .states()?
        .iter()
        .zip(enabled.pools().states()?)
    {
        assert_eq!(a.volume_m3.to_bits(), b.volume_m3.to_bits());
    }
    let mut closed_step = closed.last_step()?.ok_or("Missing closed receipt")?;
    let mut enabled_step = enabled
        .last_step()?
        .ok_or("Missing explicit-zero receipt")?;
    assert!(closed_step.forcing_interval.take().is_none());
    assert!(enabled_step.forcing_interval.take().is_some());
    assert_eq!(closed_step, enabled_step);
    for ledger in enabled.pools().element_ledgers()? {
        let source = ledger.sources.ok_or("Missing enabled source counters")?;
        assert_eq!(
            source.cumulative_rain_volume_m3.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            source.cumulative_bottom_loss_volume_m3.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            source.cumulative_evaporation_volume_m3.to_bits(),
            0.0_f64.to_bits()
        );
    }
    Ok(())
}
