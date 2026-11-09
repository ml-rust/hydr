use super::fixture::{Result, element, input, rejection, single, tolerance, unit};
use hydr::storage::{LevelPoolBatch, LevelPoolConfig, LevelPoolState};

#[test]
fn lifetime_throughput_can_exceed_each_elements_storage_capacity() -> Result {
    let mut batch = single(unit(1.0)?, 1.0)?;
    for end in [1.0, 2.0, 3.0] {
        batch = batch.trial_to(end, &[input(1.0, 1.0)])?.into_next();
        let ledger = &batch.element_ledgers()?[0];
        assert_eq!(ledger.volume_m3, 1.0);
        assert_eq!(ledger.cumulative_incoming_volume_m3, end);
        assert_eq!(ledger.cumulative_outgoing_volume_m3, end);
        assert!(ledger.residual_m3.abs() <= tolerance(end));
    }
    assert!(
        batch.element_ledgers()?[0].cumulative_incoming_volume_m3
            > batch.elements()[0].curve.maximum_volume_m3()
    );
    Ok(())
}

#[test]
fn unbounded_theoretical_ceiling_preserves_valid_finite_throughput() -> Result {
    let mut batch = single(unit(f64::MAX)?, 1.0)?;
    for end in [1.0, 2.0] {
        batch = batch.trial_to(end, &[input(1.0, 1.0)])?.into_next();
    }
    assert_eq!(batch.states()?[0].volume_m3, 1.0);
    assert_eq!(
        batch.element_ledgers()?[0].cumulative_incoming_volume_m3,
        2.0
    );
    let package = batch.snapshot_data()?;
    let mut restored = single(unit(f64::MAX)?, 0.0)?;
    restored.restore_data(&package)?;
    assert_eq!(restored.snapshot_data()?, package);
    Ok(())
}

#[test]
fn actual_cumulative_overflow_rejects_while_the_last_finite_state_remains_accepted() -> Result {
    let batch = single(unit(f64::MAX)?, 1e308)?;
    let batch = batch.trial_to(1.0, &[input(1e308, 1e308)])?.into_next();
    let before = batch.snapshot_data()?;
    assert!(batch.trial_to(2.0, &[input(1e308, 1e308)]).is_err());
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn opposite_element_drift_rejects_even_when_aggregate_water_residual_is_zero() -> Result {
    let elements = vec![element("left", unit(4.0)?), element("right", unit(4.0)?)];
    let states = vec![LevelPoolState { volume_m3: 1.0 }; 2];
    let mut source = LevelPoolBatch::new(
        elements.clone(),
        states.clone(),
        0.0,
        LevelPoolConfig::default(),
    )?;
    for end in [1.0, 2.0] {
        source = source.trial_to(end, &[input(0.0, 0.0); 2])?.into_next();
    }
    let mut package = source.snapshot_data()?;
    package.element_ledgers[0].initial_volume_m3 = 0.75;
    package.element_ledgers[1].initial_volume_m3 = 1.25;
    package.element_ledgers[0].residual_m3 = 0.25;
    package.element_ledgers[1].residual_m3 = -0.25;
    assert_eq!(
        package
            .element_ledgers
            .iter()
            .map(|ledger| ledger.residual_m3)
            .sum::<f64>(),
        0.0
    );
    let mut destination = LevelPoolBatch::new(elements, states, 0.0, LevelPoolConfig::default())?;
    let before = destination.snapshot_data()?;
    let error = rejection(destination.restore_data(&package))?;
    assert_eq!(error.element(), Some(0));
    assert!(
        error
            .to_string()
            .contains("per-element cumulative physical water residual")
    );
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

#[test]
fn capacity_and_step_count_bound_huge_cancelling_counters_before_relative_tolerance() -> Result {
    let mut source = single(unit(1.0)?, 0.7)?;
    for end in [1.0, 2.0] {
        source = source.trial_to(end, &[input(0.0, 0.0)])?.into_next();
    }
    let mut package = source.snapshot_data()?;
    let ledger = &mut package.element_ledgers[0];
    ledger.initial_volume_m3 = 0.5;
    ledger.cumulative_incoming_volume_m3 = 1e100;
    ledger.cumulative_outgoing_volume_m3 = 1e100;
    ledger.residual_m3 = ((ledger.volume_m3 - ledger.initial_volume_m3)
        - ledger.cumulative_incoming_volume_m3)
        + ledger.cumulative_outgoing_volume_m3;
    assert_eq!(ledger.residual_m3, 0.0);
    let receipt = package
        .last_step
        .as_mut()
        .ok_or("missing accepted step")?
        .receipts
        .first_mut()
        .ok_or("missing accepted receipt")?;
    receipt.cumulative_incoming_before_m3 = 1e100;
    receipt.cumulative_incoming_after_m3 = 1e100;
    receipt.cumulative_outgoing_before_m3 = 1e100;
    receipt.cumulative_outgoing_after_m3 = 1e100;
    let mut destination = single(unit(1.0)?, 0.7)?;
    let before = destination.snapshot_data()?;
    let error = rejection(destination.restore_data(&package))?;
    assert_eq!(error.element(), Some(0));
    assert!(error.to_string().contains("capacity-and-step ceiling"));
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

fn scaled_bound_integer(bound: f64) -> Result<u128> {
    let scaled = bound * 2.0_f64.powi(120);
    if !scaled.is_finite() || scaled < 0.0 || scaled.fract() != 0.0 || scaled >= 2.0_f64.powi(128) {
        return Err(format!(
            "bound={bound:.17e}: exact scaled reference requires a nonnegative u128 integer"
        )
        .into());
    }
    Ok(scaled as u128)
}

#[test]
fn aggregate_addition_error_bounds_cover_the_exact_sum_of_element_bounds() -> Result {
    for count in [1, 31, 32, 33, 64, 65] {
        for amount in [0.0, 0.1] {
            let curve = unit(4.0)?;
            let elements = (0..count)
                .map(|index| element(&format!("pool-{index}"), curve.clone()))
                .collect();
            let states = vec![LevelPoolState { volume_m3: 1.0 }; count];
            let mut batch = LevelPoolBatch::new(elements, states, 0.0, LevelPoolConfig::default())?;
            for end in [1.0, 2.0, 3.0] {
                batch = batch
                    .trial_to(end, &vec![input(amount, amount); count])?
                    .into_next();
            }
            let ledgers = batch.element_ledgers()?;
            let totals = batch.totals()?;
            for incoming in [true, false] {
                let mut exact = 0_u128;
                for ledger in &ledgers {
                    let bound = if incoming {
                        ledger.incoming_roundoff_bound_m3
                    } else {
                        ledger.outgoing_roundoff_bound_m3
                    };
                    exact = exact
                        .checked_add(scaled_bound_integer(bound)?)
                        .ok_or("exact element-bound sum exceeds u128")?;
                }
                let reported = if incoming {
                    totals.incoming_roundoff_bound_m3
                } else {
                    totals.outgoing_roundoff_bound_m3
                };
                assert!(
                    scaled_bound_integer(reported)? >= exact,
                    "count={count}, amount={amount}, incoming={incoming}"
                );
                if amount == 0.0 {
                    assert_eq!(reported.to_bits(), 0.0_f64.to_bits());
                } else {
                    assert!(exact > 0);
                }
                if count == 1 {
                    let element_bound = if incoming {
                        ledgers[0].incoming_roundoff_bound_m3
                    } else {
                        ledgers[0].outgoing_roundoff_bound_m3
                    };
                    assert_eq!(reported.to_bits(), element_bound.to_bits());
                }
            }
        }
    }
    Ok(())
}
