use super::fixture::*;
use hydr::structures::*;

#[test]
fn binding_requires_original_model_calendars_and_initial_target_agreement() -> Result {
    let config = Config::switching();
    let context = config.context()?;
    for variant in 0..5 {
        let mut changed = config.clone();
        match variant {
            0 => changed.start = -0.0,
            1 => changed.end = 5.0,
            2 => changed.times[1] = 2.5,
            3 => changed.target = 0.25,
            4 => changed.times.clear(),
            _ => return Err("Unknown binding variant".into()),
        }
        let error = rejected(ControlledGateBatch::<crate::runtime::Runtime>::new(
            config.controls(&context)?,
            changed.actuators(&context)?,
        ))?;
        assert_eq!(
            error.input_kind(),
            Some(StructureInputErrorKind::Definition),
            "{error}"
        );
    }
    let mut elements = context.elements().to_vec();
    elements[0].endpoint_a.push_str(" different");
    let foreign =
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), elements)?;
    let error = rejected(ControlledGateBatch::<crate::runtime::Runtime>::new(
        config.controls(&context)?,
        config.actuators(&foreign)?,
    ))?;
    assert_eq!(error.input_kind(), Some(StructureInputErrorKind::Identity));
    Ok(())
}

#[test]
fn binding_consumes_only_fresh_complete_gate_pairs() -> Result {
    let config = Config::switching();
    let context = config.context()?;
    let controls = config.controls(&context)?;
    let actors = config.actuators(&context)?;
    let advanced_control = controls.trial_to(0.5, &[])?.into_next();
    let advanced_actor = actors.trial_to(0.5, &[])?.into_next();
    for (control, actor) in [
        (advanced_control.clone(), actors.clone()),
        (controls.clone(), advanced_actor.clone()),
        (advanced_control, advanced_actor),
    ] {
        assert_eq!(
            rejected(ControlledGateBatch::<crate::runtime::Runtime>::new(
                control, actor
            ))?
            .input_kind(),
            Some(StructureInputErrorKind::Definition)
        );
    }
    let mut wide = config.clone();
    wide.width = 2;
    let context = wide.context()?;
    assert_eq!(
        rejected(ControlledGateBatch::<crate::runtime::Runtime>::new(
            wide.controls(&context)?,
            config.actuators(&context)?,
        ))?
        .input_kind(),
        Some(StructureInputErrorKind::Identity)
    );
    Ok(())
}

#[test]
fn pump_controllers_cannot_supply_gate_actuator_targets() -> Result {
    let config = Config::switching();
    let pump = StructureElement {
        id: "pump".into(),
        endpoint_a: "pool".into(),
        endpoint_b: "river".into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::AtoBOnly,
        law: StructureLaw::Pump(PumpParameters::new(
            PumpCurve::new(
                vec![
                    PumpCurveKnot {
                        head_m: 0.0,
                        discharge_m3_per_s: 1.0,
                    },
                    PumpCurveKnot {
                        head_m: 1.0,
                        discharge_m3_per_s: 0.0,
                    },
                ],
                "Analytical command admission curve".into(),
            )?,
            -1.0,
            PumpSpeedPolicy::ReferenceOnly,
            PumpApplicability {
                endpoint_stage_head_assertion: "Quiescent endpoints".into(),
                connection_losses_and_velocity_head_account: "Declared curve losses".into(),
                stopped_isolation_assertion: "Explicit stopped isolation".into(),
                operating_envelope: "Analytical command admission".into(),
            },
        )?),
    };
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![gate(0)?, pump],
    )?;
    let original = config.controls(&context)?;
    let mut definitions = original.definitions().to_vec();
    definitions[0].structure_id = "pump".into();
    definitions[0].low_command = ControlCommand::PumpStoppedIsolated;
    definitions[0].high_command = ControlCommand::PumpRunning { speed_ratio: 1.0 };
    let controls = context.with_operating_controls(
        definitions,
        original.initial_states().to_vec(),
        original.schedule().clone(),
        original.timing().clone(),
    )?;
    assert_eq!(
        rejected(ControlledGateBatch::<crate::runtime::Runtime>::new(
            controls,
            config.actuators(&context)?
        ))?
        .input_kind(),
        Some(StructureInputErrorKind::Identity)
    );
    Ok(())
}

#[test]
fn finite_empty_calendars_retain_constant_command_support() -> Result {
    let mut config = Config::switching();
    config.times.clear();
    config.opening = 0.5;
    config.rates = (0.0, 0.0);
    let batch = config.build()?;
    let event = batch.next_event()?.ok_or("Missing terminal event")?;
    assert_eq!(event.time_s, 4.0);
    assert!(event.terminal);
    assert!(event.decision_index.is_none());
    let batch = accept(batch, 4.0, None)?;
    position(&batch, 0.5)?;
    assert!(batch.next_event()?.is_none());
    assert_eq!(batch.controls().states()?[0].transition_count, 0);
    assert_eq!(batch.actuators().states()?[0].target_change_count, 0);
    assert_eq!(
        batch.actuators().states()?[0].phase,
        GateActuatorPhase::Stalled
    );
    assert_eq!(
        rejected(batch.trial_to(5.0, &[]))?.input_kind(),
        Some(StructureInputErrorKind::UnsupportedRegime)
    );
    Ok(())
}

#[test]
fn rejected_samples_and_event_crossing_preserve_both_children() -> Result {
    let batch = Config::switching().build()?;
    let before = batch.snapshot_data()?;
    for (end, packets, kind) in [
        (
            0.5,
            samples(1, 0.5, 2.0),
            StructureInputErrorKind::OperatingPoint,
        ),
        (1.0, Vec::new(), StructureInputErrorKind::OperatingPoint),
        (
            1.5,
            samples(1, 1.5, 2.0),
            StructureInputErrorKind::UnsupportedRegime,
        ),
        (
            1.0,
            samples(1, 0.5, 2.0),
            StructureInputErrorKind::UnsupportedRegime,
        ),
    ] {
        assert_eq!(
            rejected(batch.trial_to(end, &packets))?.input_kind(),
            Some(kind)
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}
