use super::fixture::{Result, assemble, near, orifice_case};
use super::laws::{pump, pump_with_suction, running};
use hydr::coupling::{CoupledLevelPoolBatch, CoupledLevelPoolError};

#[test]
fn connections_require_matching_identities_and_vertical_references() -> Result {
    for mode in 0..5 {
        let (mut setup, initial) = orifice_case([4.0, 2.0])?;
        match mode {
            0 => setup.connections.clear(),
            1 => setup.connections.push(setup.connections[0].clone()),
            2 => setup.connections[0].structure_id = "missing".into(),
            3 => setup.connections[0].endpoint_b.pool_id = "a".into(),
            4 => setup.structures[0].datum = "Different vertical reference".into(),
            _ => return Err("Unknown connection fixture".into()),
        }
        assert!(
            CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                setup,
                initial,
            )
            .is_err(),
            "Accepted incompatible connection mode={mode}"
        );
    }
    Ok(())
}

#[test]
fn dry_limits_and_attachment_evidence_are_explicit() -> Result {
    for mode in 0..5 {
        let (mut setup, initial) = orifice_case([4.0, 2.0])?;
        match mode {
            0 => setup.pool_support[0].zero_volume_stage_m = -1.0,
            1 => setup.pool_support[0].zero_volume_limit_assertion.clear(),
            2 => setup.connections[0].endpoint_a.bed_elevation_m = -1.0,
            3 => setup.connections[0].endpoint_a.bed_elevation_m = f64::NAN,
            4 => setup.connections[0]
                .endpoint_a
                .physical_connection_assertion
                .clear(),
            _ => return Err("Unknown support fixture".into()),
        }
        assert!(
            CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                setup,
                initial,
            )
            .is_err(),
            "Accepted unsupported physical declaration mode={mode}"
        );
    }
    Ok(())
}

#[test]
fn accuracy_budgets_and_work_limits_reject_invalid_values() -> Result {
    for mode in 0..7 {
        let (mut setup, initial) = orifice_case([4.0, 2.0])?;
        match mode {
            0 => setup.config.head_absolute_m = 0.0,
            1 => setup.config.transfer_absolute_m3 = f64::NAN,
            2 => setup.config.mass_absolute_m3 = -1.0,
            3 => setup.config.mass_relative = f64::INFINITY,
            4 => setup.config.minimum_step_s = 0.0,
            5 => setup.config.maximum_step_s = 0.5 * setup.config.minimum_step_s,
            6 => setup.config.iteration_limit = 0,
            _ => return Err("Unknown accuracy fixture".into()),
        }
        assert!(
            CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                setup,
                initial,
            )
            .is_err(),
            "Accepted invalid numerical control mode={mode}"
        );
    }
    Ok(())
}

#[test]
fn invalid_targets_preserve_the_complete_accepted_state() -> Result {
    let (setup, initial) = orifice_case([4.0, 2.0])?;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let before = batch.snapshot_data()?;
    for end in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert!(batch.trial_to(end).is_err(), "Accepted target={end}");
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn unresolved_small_edge_aggregation_rejects_without_losing_water() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0), ("c", 0.0)],
        &[4.0, 1.0, 1e-20],
        vec![
            pump("large", "a", "b", &[(-8.0, 1.0), (8.0, 1.0)])?,
            pump(
                "small",
                "a",
                "c",
                &[(-8.0, f64::EPSILON / 4.0), (8.0, f64::EPSILON / 4.0)],
            )?,
        ],
        vec![running("large"), running("small")],
    )?;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let before = batch.snapshot_data()?;
    let error = batch
        .trial_to(1.0)
        .err()
        .ok_or("Unresolved positive edge disappeared")?;
    assert!(
        matches!(
            error,
            CoupledLevelPoolError::Numerical(_) | CoupledLevelPoolError::Unresolved { .. }
        ),
        "{error}"
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn iteration_exhaustion_preserves_state_and_reports_bounded_retries() -> Result {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.config.iteration_limit = 1;
    setup.config.retry_limit = 0;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let before = batch.snapshot_data()?;
    let error = batch
        .trial_to(1.0)
        .err()
        .ok_or("Unconverged interval was accepted")?;
    match error {
        CoupledLevelPoolError::Unresolved { retries, dt_s, .. } => {
            assert_eq!(retries, 0);
            assert_eq!(dt_s, 1.0);
        }
        other => return Err(format!("Unexpected bounded-work error: {other}").into()),
    }
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn maximum_step_returns_an_explicitly_incomplete_requested_horizon() -> Result {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.config.maximum_step_s = 0.125;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let next = batch.trial_to(1.0)?.into_next();
    assert_eq!(next.time(), 0.125);
    let step = next.last_step()?.ok_or("Missing limited interval")?;
    assert_eq!(step.requested_end_s, 1.0);
    assert_eq!(step.end_s, 0.125);
    Ok(())
}

#[test]
fn suction_cutoff_shortening_recalculates_the_nonlinear_transfer() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[4.0, 2.0],
        vec![pump_with_suction(
            "pump",
            "a",
            "b",
            &[(-4.0, 7.0), (3.0, 0.0)],
            3.2,
        )?],
        vec![running("pump")],
    )?;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let before = batch.snapshot_data()?;
    let next = batch.trial_to(1.0)?.into_next();
    assert_eq!(batch.snapshot_data()?, before);
    assert_eq!(next.time(), 0.125);
    let step = next.last_step()?.ok_or("Missing shortened transfer")?;
    assert_eq!(step.retries, 3);
    assert_eq!(step.requested_end_s, 1.0);
    near(step.transfers[0].signed_volume_m3, 5.0 / 9.0);
    near(next.pools().states()?[0].volume_m3, 4.0 - 5.0 / 9.0);
    Ok(())
}

#[test]
fn unresolved_suction_cutoff_stops_at_the_minimum_interval() -> Result {
    let (mut setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[4.0, 2.0],
        vec![pump_with_suction(
            "pump",
            "a",
            "b",
            &[(-4.0, 7.0), (3.0, 0.0)],
            3.2,
        )?],
        vec![running("pump")],
    )?;
    setup.config.minimum_step_s = 0.25;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let before = batch.snapshot_data()?;
    let error = batch
        .trial_to(1.0)
        .err()
        .ok_or("Unresolved suction cutoff was accepted")?;
    match error {
        CoupledLevelPoolError::Unresolved { dt_s, retries, .. } => {
            assert_eq!(dt_s, 0.25);
            assert_eq!(retries, 2);
        }
        other => return Err(format!("Unexpected cutoff error: {other}").into()),
    }
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}
