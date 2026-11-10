use super::fixture::{Result, changing, element, enabled, exchange, policy, rejection, sources};
use hydr::storage::{LevelPoolBatch, LevelPoolConfig, LevelPoolElementLedger, LevelPoolState};

fn history() -> Result<LevelPoolBatch<crate::runtime::Runtime>> {
    let source = enabled(changing()?, 5.0)?;
    let source = source
        .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[sources(0.25, 0.1, 0.5)])?
        .into_next();
    Ok(source.trial_to(2.0, &[exchange(0.2, 0.1)])?.into_next())
}

fn legacy_history() -> Result<LevelPoolBatch<crate::runtime::Runtime>> {
    let mut batch = enabled(changing()?, 5.0)?;
    for end in [1.0, 2.0] {
        batch = batch.trial_to(end, &[exchange(0.0, 0.0)])?.into_next();
    }
    Ok(batch)
}

fn cumulative_residual(ledger: &LevelPoolElementLedger) -> Result<f64> {
    let source = ledger.sources.as_ref().ok_or("missing source ledger")?;
    let exchange_residual = ((ledger.volume_m3 - ledger.initial_volume_m3)
        - ledger.cumulative_incoming_volume_m3)
        + ledger.cumulative_outgoing_volume_m3;
    Ok(((exchange_residual - source.cumulative_rain_volume_m3)
        + source.cumulative_bottom_loss_volume_m3)
        + source.cumulative_evaporation_volume_m3)
}

#[test]
fn source_history_and_final_legacy_steps_restore_complete_counters_and_policies() -> Result {
    let mut continued = history()?;
    let snapshot = continued.snapshot();
    let package = snapshot.data()?;
    assert!(
        package
            .last_step
            .as_ref()
            .ok_or("missing accepted step")?
            .receipts[0]
            .source
            .is_none()
    );
    let process = package.element_ledgers[0]
        .sources
        .as_ref()
        .ok_or("missing persisted source ledger")?;
    assert_eq!(process.cumulative_rain_volume_m3, 2.5);
    let mut trusted = enabled(changing()?, 0.0)?;
    let mut owned = enabled(changing()?, 0.0)?;
    trusted.restore(&snapshot)?;
    owned.restore_data(&package)?;
    assert_eq!(owned.snapshot_data()?, package);
    for end in [3.0, 4.0] {
        let rates = [sources(0.1, 0.05, 0.25)];
        let input = [exchange(0.1, 0.2)];
        continued = continued
            .trial_with_sources_to(end, &input, &rates)?
            .into_next();
        trusted = trusted
            .trial_with_sources_to(end, &input, &rates)?
            .into_next();
        owned = owned
            .trial_with_sources_to(end, &input, &rates)?
            .into_next();
        assert_eq!(continued.snapshot_data()?, trusted.snapshot_data()?);
        assert_eq!(continued.snapshot_data()?, owned.snapshot_data()?);
    }
    Ok(())
}

#[test]
fn source_policy_and_counter_presence_are_immutable_on_owned_import() -> Result {
    let package = history()?.snapshot_data()?;
    let mut target = enabled(changing()?, 0.0)?;
    let before = target.snapshot_data()?;
    for mode in 0..5 {
        let mut changed = package.clone();
        match mode {
            0 => changed.source_policies = None,
            1 => changed
                .source_policies
                .as_mut()
                .ok_or("missing source policy")?[0]
                .dry_basal_footprint_assertion
                .push(' '),
            2 => {
                changed
                    .source_policies
                    .as_mut()
                    .ok_or("missing source policy")?[0]
                    .bottom_loss_provenance = "different potential-demand model".into()
            }
            3 => changed.element_ledgers[0].sources = None,
            4 => changed
                .source_policies
                .as_mut()
                .ok_or("missing source policy")?
                .clear(),
            _ => return Err("unknown policy mutation".into()),
        }
        assert!(
            target.restore_data(&changed).is_err(),
            "policy mutation={mode}"
        );
        assert_eq!(target.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn source_products_caps_witnesses_and_cumulative_continuity_are_recomputed() -> Result {
    let source = enabled(changing()?, 5.0)?;
    let package = source
        .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[sources(0.0, 0.5, 4.0)])?
        .into_next()
        .snapshot_data()?;
    let mut target = enabled(changing()?, 0.0)?;
    let before = target.snapshot_data()?;
    for mode in 0..27 {
        let mut changed = package.clone();
        let receipt = changed
            .last_step
            .as_mut()
            .ok_or("missing accepted step")?
            .receipts
            .first_mut()
            .ok_or("missing accepted receipt")?;
        let source = receipt.source.as_mut().ok_or("missing source receipt")?;
        match mode {
            0 => source.segment = 1,
            1 => source.frozen_area_m2 = 20.0,
            2 => source.rain_rate_m_per_s = 0.1,
            3 => source.rain_rate_volume_m3_per_s = 1.0,
            4 => source.rain_volume_m3 = 1.0,
            5 => source.potential_evaporation_rate_volume_m3_per_s = 1.0,
            6 => source.potential_evaporation_volume_m3 = 1.0,
            7 => source.after_rain_volume_m3 += 1.0,
            8 => source.after_bottom_loss_volume_m3 += 1.0,
            9 => source.post_source_volume_m3 += 1.0,
            10 => source.bottom_loss_cap.actual_volume_m3 += 1.0,
            11 => source.bottom_loss_cap.unmet_volume_m3 += 1.0,
            12 => source.bottom_loss_cap.availability_limited = true,
            13 => source.evaporation_cap.availability_limited = false,
            14 => source.evaporation_cap.unmet_volume_m3 += 1.0,
            15 => source.evaporation_cap.unmet_witness_m3 = f64::from_bits(1),
            16 => source.evaporation_cap.reconciliation_residual_m3 = 1.0,
            17 => source.evaporation_cap.reconciliation_bound_m3 = -1.0,
            18 => source.rain_witness_m3 = f64::from_bits(1),
            19 => source.bottom_loss_witness_m3 = f64::from_bits(1),
            20 => source.evaporation_witness_m3 = f64::from_bits(1),
            21 => source.post_source_stage_m = Some(101.0),
            22 => source.post_source_inverse_residual_m3 = 1.0,
            23 => source.counters.bottom_loss.before_m3 = 1.0,
            24 => source.counters.evaporation.after_m3 += 1.0,
            25 => source.counters.bottom_loss_demand.witness_m3 = 1.0,
            26 => source.counters.bottom_loss.bound_after_m3 = -1.0,
            _ => return Err("unknown source receipt mutation".into()),
        }
        assert!(
            target.restore_data(&changed).is_err(),
            "source mutation={mode}"
        );
        assert_eq!(target.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn actual_bottom_counter_cannot_exceed_potential_demand_after_a_legacy_step() -> Result {
    let source = enabled(changing()?, 5.0)?;
    let source = source
        .trial_with_sources_to(1.0, &[exchange(0.0, 0.0)], &[sources(0.0, 0.0, 1.0)])?
        .into_next();
    let source = source.trial_to(2.0, &[exchange(0.0, 0.0)])?.into_next();
    let mut package = source.snapshot_data()?;
    assert!(
        package
            .last_step
            .as_ref()
            .ok_or("missing final step")?
            .receipts[0]
            .source
            .is_none()
    );
    package.element_ledgers[0]
        .sources
        .as_mut()
        .ok_or("missing source ledger")?
        .cumulative_bottom_loss_demand_m3 = 0.5;
    let mut target = enabled(changing()?, 5.0)?;
    let before = target.snapshot_data()?;
    let error = rejection(target.restore_data(&package))?;
    assert!(
        error
            .to_string()
            .contains("cumulative actual bottom loss exceeds cumulative supplied potential demand"),
        "unexpected error: {error}"
    );
    assert_eq!(target.snapshot_data()?, before);
    Ok(())
}

#[test]
fn source_counter_capacity_ceilings_precede_relative_physical_tolerance() -> Result {
    let package = legacy_history()?.snapshot_data()?;
    let mut target = enabled(changing()?, 5.0)?;
    let before = target.snapshot_data()?;
    for process in 0..3 {
        let mut changed = package.clone();
        let ledger = &mut changed.element_ledgers[0];
        let sources = ledger.sources.as_mut().ok_or("missing source counters")?;
        match process {
            0 => sources.cumulative_rain_volume_m3 = 1e100,
            1 => sources.cumulative_evaporation_volume_m3 = 1e100,
            2 => {
                sources.cumulative_bottom_loss_volume_m3 = 1e100;
                sources.cumulative_bottom_loss_demand_m3 = 1e100;
            }
            _ => return Err("unknown source counter".into()),
        }
        ledger.residual_m3 = cumulative_residual(ledger)?;
        let error = rejection(target.restore_data(&changed))?;
        assert!(
            error.to_string().contains("capacity-and-step ceiling"),
            "unexpected rejection: {error}"
        );
        assert_eq!(target.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn source_cumulative_opposite_drift_rejects_before_aggregate_cancellation() -> Result {
    let elements = vec![
        element("rain", changing()?),
        element("evaporation", changing()?),
    ];
    let states = vec![LevelPoolState { volume_m3: 5.0 }; 2];
    let mut source = LevelPoolBatch::<crate::runtime::Runtime>::new_with_sources(
        crate::runtime::client(),
        elements.clone(),
        states.clone(),
        0.0,
        LevelPoolConfig::default(),
        vec![policy(); 2],
    )?;
    for end in [1.0, 2.0] {
        source = source.trial_to(end, &[exchange(0.0, 0.0); 2])?.into_next();
    }
    let mut changed = source.snapshot_data()?;
    changed.element_ledgers[0]
        .sources
        .as_mut()
        .ok_or("missing rain ledger")?
        .cumulative_rain_volume_m3 = 0.25;
    changed.element_ledgers[1]
        .sources
        .as_mut()
        .ok_or("missing evaporation ledger")?
        .cumulative_evaporation_volume_m3 = 0.25;
    for ledger in &mut changed.element_ledgers {
        ledger.residual_m3 = cumulative_residual(ledger)?;
    }
    assert_eq!(
        changed
            .element_ledgers
            .iter()
            .map(|ledger| ledger.residual_m3)
            .sum::<f64>(),
        0.0
    );
    let mut target = LevelPoolBatch::<crate::runtime::Runtime>::new_with_sources(
        crate::runtime::client(),
        elements,
        states,
        0.0,
        LevelPoolConfig::default(),
        vec![policy(); 2],
    )?;
    let before = target.snapshot_data()?;
    let error = rejection(target.restore_data(&changed))?;
    assert!(
        error
            .to_string()
            .contains("per-element cumulative physical water residual"),
        "unexpected rejection: {error}"
    );
    assert_eq!(target.snapshot_data()?, before);
    Ok(())
}
