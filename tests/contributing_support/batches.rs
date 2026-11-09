use super::fixture::{Result, batch, definition, measured, ordered, primary};
use hydr::runoff::{
    SubcatchmentBatch, SubcatchmentConfig, SubcatchmentState, SupportError, SupportErrorKind,
};

#[test]
fn missing_extra_or_foreign_numerical_batches_reject_without_advancement() -> Result {
    let fixture = ordered()?;
    let plan = fixture.plan()?;
    let packet = plan.admit_interval(fixture.interval.clone())?;
    assert!(plan.trial_subcatchments(&packet, None).is_err());
    for mode in 0..5 {
        let mut definitions = plan.active_definitions().to_vec();
        match mode {
            0 => definitions.reverse(),
            1 => definitions[0].id = "foreign-id".into(),
            2 => definitions[0] = definition("model-b", 1.01)?,
            3 => {}
            4 => {
                let mut geometry = *definitions[0].parameters.geometry();
                geometry.width_m = -0.0;
                definitions[0].parameters = hydr::runoff::SubcatchmentParameters::new(
                    geometry,
                    None,
                    hydr::runoff::SubcatchmentRouting::Direct,
                )?;
            }
            _ => return Err("unknown batch fixture".into()),
        }
        let states = definitions
            .iter()
            .map(|definition| SubcatchmentState::initialized(&definition.parameters))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let accepted = SubcatchmentBatch::new(
            definitions,
            states,
            if mode == 3 { 1.0 } else { 0.0 },
            SubcatchmentConfig::default(),
        )?;
        let before = accepted.snapshot_data()?;
        assert!(plan.trial_subcatchments(&packet, Some(&accepted)).is_err());
        assert_eq!(accepted.snapshot_data()?, before);
    }
    let inactive_fixture = measured()?;
    let inactive_plan = inactive_fixture.plan()?;
    let inactive_packet = inactive_plan.admit_interval(inactive_fixture.interval)?;
    let accepted = batch(&plan, 0.0)?;
    let before = accepted.snapshot_data()?;
    assert!(
        inactive_plan
            .trial_subcatchments(&inactive_packet, Some(&accepted))
            .is_err()
    );
    assert_eq!(accepted.snapshot_data()?, before);
    let foreign = fixture.plan()?.admit_interval(fixture.interval)?;
    let error = match plan.trial_subcatchments(&foreign, Some(&accepted)) {
        Ok(_) => return Err("foreign packet requires rejection".into()),
        Err(error) => error,
    };
    assert_eq!(error.kind(), Some(SupportErrorKind::Identity));
    assert_eq!(accepted.snapshot_data()?, before);
    Ok(())
}

#[test]
fn repeated_trials_are_immutable_and_accepted_time_prevents_packet_reuse() -> Result {
    let fixture = primary()?;
    let plan = fixture.plan()?;
    let packet = plan.admit_interval(fixture.interval)?;
    let accepted = batch(&plan, 0.0)?;
    let before = accepted.snapshot_data()?;
    let first = plan
        .trial_subcatchments(&packet, Some(&accepted))?
        .ok_or("missing first trial")?;
    let repeat = plan
        .trial_subcatchments(&packet, Some(&accepted))?
        .ok_or("missing repeated trial")?;
    assert_eq!(first.outputs()?, repeat.outputs()?);
    assert_eq!(first.diagnostics()?, repeat.diagnostics()?);
    assert_eq!(accepted.snapshot_data()?, before);
    let next = first.into_next();
    let before = next.snapshot_data()?;
    let error = match plan.trial_subcatchments(&packet, Some(&next)) {
        Ok(_) => return Err("reused interval requires rejection".into()),
        Err(error) => error,
    };
    assert_eq!(error.kind(), Some(SupportErrorKind::Coverage));
    assert_eq!(next.snapshot_data()?, before);
    Ok(())
}

#[test]
fn numerical_config_remains_batch_owned_and_original_trial_errors_remain_typed() -> Result {
    let fixture = primary()?;
    let plan = fixture.plan()?;
    let definitions = plan.active_definitions().to_vec();
    let states = definitions
        .iter()
        .map(|definition| SubcatchmentState::initialized(&definition.parameters))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut config = SubcatchmentConfig::default();
    config.reservoir.volume_absolute = 1e-8;
    let accepted = SubcatchmentBatch::new(definitions, states, 0.0, config)?;
    let packet = plan.admit_interval(fixture.interval.clone())?;
    plan.trial_subcatchments(&packet, Some(&accepted))?
        .ok_or("missing config-independent trial")?;
    let before = accepted.snapshot_data()?;
    let mut input = fixture.interval;
    input.end = 2.0;
    input.surface.end = 2.0;
    input.subcatchments[0].end = 2.0;
    input.subcatchments[0].input.rain_rate_m_per_s = f64::MAX;
    let packet = plan.admit_interval(input)?;
    assert!(matches!(
        plan.trial_subcatchments(&packet, Some(&accepted)),
        Err(SupportError::Subcatchment(_))
    ));
    assert_eq!(accepted.snapshot_data()?, before);
    Ok(())
}
