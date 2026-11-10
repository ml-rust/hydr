use super::fixture::*;
use hydr::structures::*;

#[test]
fn response_definitions_require_complete_original_pump_order_and_explicit_assumptions() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("a")?, pump("b")?],
    )?;
    let config = Config::switching();
    let controls = config.controls(&context, &["a", "b"])?;
    for definitions in [
        Vec::new(),
        responses(&["a"]),
        responses(&["b", "a"]),
        responses(&["a", "a"]),
        responses(&["a", "foreign"]),
    ] {
        kind(
            ControlledPumpBatch::<crate::runtime::Runtime>::new(controls.clone(), definitions),
            StructureInputErrorKind::Identity,
        )?;
    }
    for field in 0..2 {
        let mut definitions = responses(&["a", "b"]);
        if field == 0 {
            definitions[0].response_provenance.clear();
        } else {
            definitions[0].ideal_response_applicability.clear();
        }
        kind(
            ControlledPumpBatch::<crate::runtime::Runtime>::new(controls.clone(), definitions),
            StructureInputErrorKind::Definition,
        )?;
    }
    Ok(())
}

#[test]
fn gate_target_controls_cannot_acquire_instantaneous_pump_authority() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![gate("gate")?],
    )?;
    let mut config = Config::switching();
    config.commands = [
        ControlCommand::GateTarget { opening_m: 0.0 },
        ControlCommand::GateTarget { opening_m: 1.0 },
    ];
    kind(
        ControlledPumpBatch::<crate::runtime::Runtime>::new(
            config.controls(&context, &["gate"])?,
            responses(&["gate"]),
        ),
        StructureInputErrorKind::Definition,
    )
}

#[test]
fn advanced_controller_adoption_preserves_existing_time_memory_and_interval_count() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let config = Config::switching();
    let controls = config.controls(&context, &["pump"])?;
    let controls = controls
        .trial_to(1.0, &samples(&["pump"], 1.0, 2.0))?
        .into_next();
    let before = controls.snapshot_data()?;
    let adopted =
        ControlledPumpBatch::<crate::runtime::Runtime>::new(controls, responses(&["pump"]))?;
    assert_eq!(adopted.time().to_bits(), 1.0_f64.to_bits());
    assert_eq!(adopted.accepted_steps(), 1);
    assert_eq!(adopted.controls().snapshot_data()?, before);
    assert_eq!(
        adopted.operations()?[0].operating_point,
        StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
    );
    assert_eq!(
        adopted
            .next_event()
            .ok_or("Missing decision")?
            .decision_index,
        Some(1)
    );
    Ok(())
}

#[test]
fn advanced_adoption_keeps_pending_qualification_at_its_original_anchor() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let mut config = Config::switching();
    config.delays = (2.0, 0.0);
    let controls = config.controls(&context, &["pump"])?;
    let controls = controls
        .trial_to(1.0, &samples(&["pump"], 1.0, 2.0))?
        .into_next();
    let controls = controls
        .trial_to(2.0, &samples(&["pump"], 2.0, 2.0))?
        .into_next();
    let batch =
        ControlledPumpBatch::<crate::runtime::Runtime>::new(controls, responses(&["pump"]))?;
    assert_eq!(
        batch.controls().states()?[0]
            .pending
            .ok_or("Missing adopted pending timer")?
            .since_s
            .to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        batch.operations()?[0].operating_point,
        StructureOperatingPoint::PumpStoppedIsolated
    );
    let batch = accept(batch, 3.0, Some(2.0))?;
    assert_eq!(
        batch.operations()?[0].operating_point,
        StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
    );
    assert_eq!(batch.accepted_steps(), 3);
    assert_eq!(batch.controls().states()?[0].transition_count, 1);
    Ok(())
}

#[test]
fn rejected_and_discarded_response_trials_preserve_complete_control_memory() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let batch = Config::switching().build(&context, &["pump"])?;
    let before = batch.snapshot_data()?;
    drop(batch.trial_to(1.0, &samples(&["pump"], 1.0, 2.0))?);
    assert_eq!(batch.snapshot_data()?, before);
    for (end, observations, expected) in [
        (
            0.5,
            samples(&["pump"], 0.5, 2.0),
            StructureInputErrorKind::OperatingPoint,
        ),
        (1.0, Vec::new(), StructureInputErrorKind::OperatingPoint),
        (
            1.5,
            samples(&["pump"], 1.5, 2.0),
            StructureInputErrorKind::UnsupportedRegime,
        ),
    ] {
        kind(batch.trial_to(end, &observations), expected)?;
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}
