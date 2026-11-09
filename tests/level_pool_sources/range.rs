use super::fixture::{Result, changing, curve, enabled, exchange, rejection, sources};

#[test]
fn only_used_rate_area_duration_products_limit_source_admission() -> Result {
    let batch = enabled(curve(&[(0.0, 0.0), (1e300, 1.0)])?, 0.0)?;
    let rate = 1e-200_f64;
    let duration = 1e-200;
    assert_eq!(rate * duration, 0.0);
    let trial = batch.trial_with_sources_to(
        duration,
        &[exchange(0.0, 0.0)],
        &[sources(rate, rate, 0.0)],
    )?;
    let step = trial.diagnostics()?;
    assert!(step.rain_volume_m3 > 0.0);
    assert_eq!(
        step.rain_volume_m3.to_bits(),
        step.evaporation_volume_m3.to_bits()
    );
    assert_eq!(trial.outputs()?[0].volume_m3, 0.0);
    assert_eq!(trial.outputs()?[0].stage_m, None);
    Ok(())
}

#[test]
fn positive_used_products_reject_underflow_and_overflow_even_on_dry_storage() -> Result {
    for (pool_curve, end, rates) in [
        (
            curve(&[(0.0, 0.0), (1e-300, 1.0)])?,
            1.0,
            sources(1e-300, 0.0, 0.0),
        ),
        (
            curve(&[(0.0, 0.0), (1e-300, 1.0)])?,
            1.0,
            sources(0.0, 1e-300, 0.0),
        ),
        (changing()?, 0.5, sources(0.0, 0.0, f64::from_bits(1))),
        (changing()?, 1.0, sources(f64::MAX, 0.0, 0.0)),
        (changing()?, 1.0, sources(0.0, f64::MAX, 0.0)),
        (changing()?, 2.0, sources(0.0, 0.0, f64::MAX)),
    ] {
        let batch = enabled(pool_curve, 0.0)?;
        let before = batch.snapshot_data()?;
        let error = rejection(batch.trial_with_sources_to(end, &[exchange(0.0, 0.0)], &[rates]))?;
        assert_eq!(error.element(), Some(0));
        assert!(error.to_string().contains("unrepresentable"));
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn actual_source_operations_cannot_absorb_positive_water_changes() -> Result {
    for (maximum, initial, rates) in [
        (2.0_f64.powi(54), 2.0_f64.powi(53), sources(1.0, 0.0, 0.0)),
        (2.0_f64.powi(55), 2.0_f64.powi(54), sources(0.0, 1.0, 0.0)),
        (2.0_f64.powi(55), 2.0_f64.powi(54), sources(0.0, 0.0, 1.0)),
    ] {
        let batch = enabled(curve(&[(0.0, 0.0), (maximum, maximum)])?, initial)?;
        let before = batch.snapshot_data()?;
        assert!(
            batch
                .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[rates])
                .is_err()
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn potential_and_unmet_demand_never_enlarge_physical_tolerance_scales() -> Result {
    let batch = enabled(changing()?, 5.0)?;
    let trial =
        batch.trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[sources(0.0, 0.0, 1000.0)])?;
    let step = trial.diagnostics()?;
    assert_eq!(step.bottom_loss_demand_m3, 1000.0);
    assert_eq!(step.bottom_loss_volume_m3, 5.0);
    assert_eq!(step.unmet_bottom_loss_m3, 995.0);
    assert_eq!(step.physical_scale_m3, 5.0);
    let next = trial.into_next();
    let ledger = next.element_ledgers()?;
    let source = ledger[0]
        .sources
        .as_ref()
        .ok_or("missing source counters")?;
    assert_eq!(source.cumulative_bottom_loss_demand_m3, 1000.0);
    assert_eq!(source.cumulative_bottom_loss_volume_m3, 5.0);
    let dry = enabled(changing()?, 0.0)?;
    let step = dry
        .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[sources(0.0, 1e300, 0.0)])?
        .diagnostics()?;
    assert!(step.potential_evaporation_volume_m3 > 1e300);
    assert_eq!(step.evaporation_volume_m3, 0.0);
    assert_eq!(step.physical_scale_m3, 0.0);
    Ok(())
}

#[test]
fn source_shapes_and_nonnegative_finite_rates_remain_explicit() -> Result {
    let batch = enabled(changing()?, 0.0)?;
    let before = batch.snapshot_data()?;
    for values in [
        vec![],
        vec![sources(-1.0, 0.0, 0.0)],
        vec![sources(0.0, f64::NAN, 0.0)],
        vec![sources(0.0, 0.0, f64::INFINITY)],
    ] {
        assert!(
            batch
                .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &values)
                .is_err()
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}
