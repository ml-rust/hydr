use super::fixture::{
    Result, changing, curve, element, enabled, exchange, policy, rejection, sources, tolerance,
};
use hydr::storage::{LevelPoolBatch, LevelPoolConfig, LevelPoolState};

#[test]
fn dry_rain_bottom_priority_and_combined_sources_keep_original_process_volumes() -> Result {
    for (initial, rates, rain, bottom, evaporation, final_volume) in [
        (0.0, sources(0.25, 0.0, 0.0), 5.0, 0.0, 0.0, 5.0),
        (5.0, sources(0.0, 0.5, 4.0), 0.0, 5.0, 0.0, 0.0),
        (0.0, sources(0.25, 0.1, 1.0), 5.0, 2.0, 2.0, 1.0),
    ] {
        let batch = enabled(changing()?, initial)?;
        let trial = batch.trial_with_sources_to(2.0, &[exchange(0.0, 0.0)], &[rates])?;
        let step = trial.diagnostics()?;
        assert_eq!(step.rain_volume_m3, rain);
        assert_eq!(step.bottom_loss_volume_m3, bottom);
        assert_eq!(step.evaporation_volume_m3, evaporation);
        assert_eq!(trial.outputs()?[0].volume_m3, final_volume);
        assert!(step.water_residual_m3.abs() <= tolerance(step.physical_scale_m3));
    }
    let batch = enabled(curve(&[(0.0, 100.0), (20.0, 102.0)])?, 2.0)?;
    let trial =
        batch.trial_with_sources_to(2.0, &[exchange(0.0, 0.0)], &[sources(0.2, 0.1, 0.5)])?;
    let step = trial.diagnostics()?;
    assert_eq!(
        (
            step.rain_volume_m3,
            step.bottom_loss_volume_m3,
            step.evaporation_volume_m3,
            trial.outputs()?[0].volume_m3
        ),
        (4.0, 1.0, 2.0, 3.0)
    );
    Ok(())
}

#[test]
fn right_sided_knots_use_volume_support_and_the_maximum_uses_the_last_segment() -> Result {
    for (volume, segment, area) in [
        (0.0, 0, 10.0),
        (10.0_f64.next_down(), 0, 10.0),
        (10.0, 1, 20.0),
        (10.0_f64.next_up(), 1, 20.0),
        (30.0, 1, 20.0),
    ] {
        let batch = enabled(changing()?, volume)?;
        let trial =
            batch.trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[sources(0.01, 0.01, 0.0)])?;
        let step = trial.diagnostics()?;
        let source = step.receipts[0]
            .source
            .as_ref()
            .ok_or("missing source-area receipt")?;
        assert_eq!(source.segment, segment);
        assert_eq!(source.frozen_area_m2, area);
    }
    Ok(())
}

#[test]
fn both_loss_caps_reconcile_original_potentials_and_strict_availability_reasons() -> Result {
    for evaporation in [true, false] {
        let batch = enabled(curve(&[(0.0, 0.0), (1.0, 1.0)])?, 0.1)?;
        let rates = if evaporation {
            sources(0.0, 1.0, 0.0)
        } else {
            sources(0.0, 0.0, 1.0)
        };
        let step = batch
            .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[rates])?
            .diagnostics()?;
        let source = step.receipts[0]
            .source
            .as_ref()
            .ok_or("missing cap receipt")?;
        let cap = if evaporation {
            &source.evaporation_cap
        } else {
            &source.bottom_loss_cap
        };
        assert_eq!(cap.potential_volume_m3, 1.0);
        assert_eq!(cap.actual_volume_m3.to_bits(), 0.1_f64.to_bits());
        assert_eq!(cap.unmet_volume_m3.to_bits(), 0.9_f64.to_bits());
        assert_eq!(
            cap.unmet_witness_m3.to_bits(),
            (-2.0_f64.powi(-55)).to_bits()
        );
        assert!(cap.availability_limited);
        assert!(cap.reconciliation_consistency_m3.abs() <= cap.reconciliation_bound_m3);
    }
    let batch = enabled(changing()?, 5.0)?;
    let step = batch
        .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[sources(0.0, 0.0, 5.0)])?
        .diagnostics()?;
    let source = step.receipts[0]
        .source
        .as_ref()
        .ok_or("missing equal-demand source")?;
    assert!(!source.bottom_loss_cap.availability_limited);
    assert!(!source.evaporation_cap.availability_limited);
    Ok(())
}

#[test]
fn known_rain_supplies_outgoing_after_sources_but_external_incoming_does_not() -> Result {
    let dry = enabled(changing()?, 0.0)?;
    let trial =
        dry.trial_with_sources_to(2.0, &[exchange(0.0, 5.0)], &[sources(0.25, 0.0, 0.0)])?;
    assert_eq!(trial.outputs()?[0].volume_m3, 0.0);
    assert_eq!(trial.diagnostics()?.outgoing_volume_m3, 5.0);
    for (initial, exchange, rates, end) in [
        (0.0, exchange(10.0, 2.0), sources(0.1, 0.0, 0.0), 1.0),
        (5.0, exchange(2.0, 1.0), sources(0.0, 0.0, 4.0), 2.0),
    ] {
        let batch = enabled(changing()?, initial)?;
        let before = batch.snapshot_data()?;
        assert!(
            batch
                .trial_with_sources_to(end, &[exchange], &[rates])
                .is_err()
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn balanced_capacity_uses_algebraic_intermediates_but_actual_processes_obey_resolution() -> Result {
    let batch = enabled(changing()?, 30.0)?;
    let trial =
        batch.trial_with_sources_to(2.0, &[exchange(0.0, 0.0)], &[sources(0.25, 0.25, 0.0)])?;
    let step = trial.diagnostics()?;
    let source = step.receipts[0]
        .source
        .as_ref()
        .ok_or("missing balanced source")?;
    assert_eq!(source.after_rain_volume_m3, 40.0);
    assert_eq!(source.post_source_volume_m3, 30.0);
    assert_eq!(
        (step.rain_volume_m3, step.evaporation_volume_m3),
        (10.0, 10.0)
    );
    assert_eq!(trial.outputs()?[0].volume_m3, 30.0);
    for rates in [
        sources(1.0, 1.0, 0.0),
        sources(0.5, 0.0, 20.0),
        sources(0.5, 1.0, 0.0),
    ] {
        let before = batch.snapshot_data()?;
        let error = rejection(batch.trial_with_sources_to(2.0, &[exchange(0.0, 0.0)], &[rates]))?;
        assert_eq!(error.element(), Some(0));
        assert!(error.to_string().contains("source-step resolution"));
        assert_eq!(batch.snapshot_data()?, before);
    }
    let batch = enabled(changing()?, 20.0)?;
    let before = batch.snapshot_data()?;
    assert!(
        batch
            .trial_with_sources_to(2.0, &[exchange(0.0, 20.0)], &[sources(0.5, 0.0, 0.0)])
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn source_enabling_is_explicit_and_zero_sources_preserve_the_complete_legacy_path() -> Result {
    let pool_curve = changing()?;
    let plain = LevelPoolBatch::new(
        vec![element("pool", pool_curve.clone())],
        vec![LevelPoolState { volume_m3: 5.0 }],
        0.0,
        LevelPoolConfig::default(),
    )?;
    let enabled = enabled(pool_curve, 5.0)?;
    let input = [exchange(0.2, 0.1)];
    let plain_trial = plain.trial_to(1.0, &input)?;
    let legacy = enabled.trial_to(1.0, &input)?;
    let zero = enabled.trial_with_sources_to(1.0, &input, &[sources(0.0, 0.0, 0.0)])?;
    assert_eq!(plain_trial.outputs()?, zero.outputs()?);
    assert_eq!(plain_trial.diagnostics()?, zero.diagnostics()?);
    assert_eq!(legacy.diagnostics()?, zero.diagnostics()?);
    assert_eq!(
        legacy.into_next().snapshot_data()?,
        zero.into_next().snapshot_data()?
    );
    let before = plain.snapshot_data()?;
    assert!(
        plain
            .trial_with_sources_to(1.0, &input, &[sources(0.0, 0.0, 0.0)])
            .is_err()
    );
    assert_eq!(plain.snapshot_data()?, before);
    let mut invalid = policy();
    invalid.dry_basal_footprint_assertion.clear();
    assert!(
        LevelPoolBatch::new_with_sources(
            vec![element("pool", changing()?)],
            vec![LevelPoolState { volume_m3: 0.0 }],
            0.0,
            LevelPoolConfig::default(),
            vec![invalid]
        )
        .is_err()
    );
    Ok(())
}
