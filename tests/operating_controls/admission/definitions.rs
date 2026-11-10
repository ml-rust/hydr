use super::{fixture, reference_case, rejects};
use hydr::structures::*;

#[test]
fn controlled_subsets_preserve_actual_order_and_reject_conflicting_bindings() -> fixture::Result {
    let case = fixture::case("inclusive_thresholds_and_band_hold")?;
    let elements = (0..5)
        .map(|lane| reference_case::pump(&reference_case::device(lane)))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let structures = StructureBatch::new(elements)?;
    let definitions = vec![
        reference_case::definition(case, 0, &reference_case::device(1)),
        reference_case::definition(case, 1, &reference_case::device(3)),
    ];
    let initial = vec![
        reference_case::initial(case, 0),
        reference_case::initial(case, 1),
    ];
    let bind = |definitions, initial| {
        structures.with_operating_controls(
            definitions,
            initial,
            reference_case::schedule(case),
            reference_case::timing(case),
        )
    };
    let batch = bind(definitions.clone(), initial.clone())?;
    assert_eq!(batch.elements().len(), 5);
    assert_eq!(batch.definitions(), definitions);
    assert_eq!(
        batch
            .commands()?
            .iter()
            .map(|row| row.element)
            .collect::<Vec<_>>(),
        [1, 3]
    );

    let mut repeated_controller = definitions.clone();
    repeated_controller[1].controller_id = repeated_controller[0].controller_id.clone();
    let mut repeated_initial = initial.clone();
    repeated_initial[1].controller_id = repeated_initial[0].controller_id.clone();
    rejects(
        bind(repeated_controller, repeated_initial),
        StructureInputErrorKind::Identity,
        Some(3),
    )?;

    let mut repeated_device = definitions.clone();
    repeated_device[1].structure_id = repeated_device[0].structure_id.clone();
    rejects(
        bind(repeated_device, initial.clone()),
        StructureInputErrorKind::Identity,
        Some(1),
    )?;

    let mut reordered = definitions.clone();
    reordered.reverse();
    let mut reordered_initial = initial.clone();
    reordered_initial.reverse();
    rejects(
        bind(reordered, reordered_initial),
        StructureInputErrorKind::Identity,
        Some(1),
    )?;

    let mut unknown = definitions.clone();
    unknown[0].structure_id = "absent-device".into();
    rejects(
        bind(unknown, initial.clone()),
        StructureInputErrorKind::Identity,
        None,
    )?;

    let mut wrong_initial = initial.clone();
    wrong_initial[0].controller_id = "another-controller".into();
    rejects(
        bind(definitions.clone(), wrong_initial),
        StructureInputErrorKind::Identity,
        Some(1),
    )?;
    rejects(
        bind(definitions, Vec::new()),
        StructureInputErrorKind::Identity,
        None,
    )?;
    rejects(
        bind(Vec::new(), Vec::new()),
        StructureInputErrorKind::Definition,
        None,
    )
}

#[test]
fn thresholds_delays_and_timing_require_explicit_finite_domains() -> fixture::Result {
    let case = fixture::case("inclusive_thresholds_and_band_hold")?;
    let structures = StructureBatch::new(vec![reference_case::pump(&reference_case::device(0))?])?;
    let changes: [fn(&mut OperatingControlDefinition); 7] = [
        |definition| definition.lower_stage_m = definition.upper_stage_m,
        |definition| definition.lower_stage_m = 3.0,
        |definition| definition.upper_stage_m = f64::NAN,
        |definition| definition.lower_stage_m = f64::NEG_INFINITY,
        |definition| definition.high_qualification_delay_s = -f64::from_bits(1),
        |definition| definition.low_qualification_delay_s = f64::INFINITY,
        |definition| definition.high_qualification_delay_s = f64::NAN,
    ];
    for change in changes {
        let mut definition = reference_case::definition(case, 0, &reference_case::device(0));
        change(&mut definition);
        rejects(
            structures.with_operating_controls(
                vec![definition],
                vec![reference_case::initial(case, 0)],
                reference_case::schedule(case),
                reference_case::timing(case),
            ),
            StructureInputErrorKind::Definition,
            Some(0),
        )?;
    }
    let changes: [fn(&mut SampledControlTiming); 5] = [
        |timing| timing.maximum_decision_gap_s = 0.0,
        |timing| timing.maximum_decision_gap_s = f64::INFINITY,
        |timing| timing.maximum_timer_lateness_s = -f64::from_bits(1),
        |timing| timing.maximum_timer_lateness_s = f64::NAN,
        |timing| timing.sampled_applicability.clear(),
    ];
    for change in changes {
        let mut timing = reference_case::timing(case);
        change(&mut timing);
        rejects(
            structures.with_operating_controls(
                vec![reference_case::definition(
                    case,
                    0,
                    &reference_case::device(0),
                )],
                vec![reference_case::initial(case, 0)],
                reference_case::schedule(case),
                timing,
            ),
            StructureInputErrorKind::Definition,
            None,
        )?;
    }
    Ok(())
}

#[test]
fn commands_respect_actual_pump_policy_gate_range_and_passive_laws() -> fixture::Result {
    let case = fixture::case("inclusive_thresholds_and_band_hold")?;
    let pump = StructureBatch::new(vec![reference_case::pump(&reference_case::device(0))?])?;
    for command in [
        ControlCommand::GateTarget { opening_m: 0.25 },
        ControlCommand::PumpRunning { speed_ratio: 0.5 },
        ControlCommand::PumpRunning { speed_ratio: 0.0 },
        ControlCommand::PumpRunning {
            speed_ratio: f64::NAN,
        },
    ] {
        let mut definition = reference_case::definition(case, 0, &reference_case::device(0));
        definition.high_command = command;
        rejects(
            pump.with_operating_controls(
                vec![definition],
                vec![reference_case::initial(case, 0)],
                reference_case::schedule(case),
                reference_case::timing(case),
            ),
            StructureInputErrorKind::Definition,
            Some(0),
        )?;
    }

    let gate = StructureBatch::new(vec![reference_case::gate(&reference_case::device(0))?])?;
    let mut definition = reference_case::definition(case, 0, &reference_case::device(0));
    definition.low_command = ControlCommand::GateTarget { opening_m: 0.0 };
    definition.high_command = ControlCommand::GateTarget { opening_m: 1.0 };
    gate.with_operating_controls(
        vec![definition.clone()],
        vec![reference_case::initial(case, 0)],
        reference_case::schedule(case),
        reference_case::timing(case),
    )?;
    for command in [
        ControlCommand::PumpStoppedIsolated,
        ControlCommand::GateTarget {
            opening_m: -f64::from_bits(1),
        },
        ControlCommand::GateTarget {
            opening_m: 1.0 + f64::EPSILON,
        },
        ControlCommand::GateTarget {
            opening_m: f64::INFINITY,
        },
    ] {
        let mut changed = definition.clone();
        changed.high_command = command;
        rejects(
            gate.with_operating_controls(
                vec![changed],
                vec![reference_case::initial(case, 0)],
                reference_case::schedule(case),
                reference_case::timing(case),
            ),
            StructureInputErrorKind::Definition,
            Some(0),
        )?;
    }

    let mut passive = reference_case::pump(&reference_case::device(0))?;
    passive.law = StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
        1.0,
        0.0,
        1.0,
        0.6,
        9.81,
        OrificeApplicability {
            coefficient_provenance: "Synthetic command admission".into(),
            fully_submerged_applicability: "Quiescent submerged reference opening".into(),
            symmetric_reversal: true,
        },
    )?);
    let passive = StructureBatch::new(vec![passive])?;
    rejects(
        passive.with_operating_controls(
            vec![reference_case::definition(
                case,
                0,
                &reference_case::device(0),
            )],
            vec![reference_case::initial(case, 0)],
            reference_case::schedule(case),
            reference_case::timing(case),
        ),
        StructureInputErrorKind::Definition,
        Some(0),
    )
}

#[test]
fn schedules_check_original_order_support_and_terminal_cadence() -> fixture::Result {
    let case = fixture::case("inclusive_thresholds_and_band_hold")?;
    let structures = StructureBatch::new(vec![reference_case::pump(&reference_case::device(0))?])?;
    let changes: [fn(&mut ControlDecisionSchedule); 7] = [
        |schedule| schedule.initial_time_s = -1.0,
        |schedule| schedule.end_s = schedule.initial_time_s,
        |schedule| schedule.end_s = f64::INFINITY,
        |schedule| schedule.decision_times_s = vec![1.0, 1.0],
        |schedule| schedule.decision_times_s = vec![0.0],
        |schedule| schedule.decision_times_s = vec![f64::NAN],
        |schedule| schedule.decision_times_s = vec![6.0],
    ];
    for change in changes {
        let mut schedule = reference_case::schedule(case);
        change(&mut schedule);
        rejects(
            structures.with_operating_controls(
                vec![reference_case::definition(
                    case,
                    0,
                    &reference_case::device(0),
                )],
                vec![reference_case::initial(case, 0)],
                schedule,
                reference_case::timing(case),
            ),
            StructureInputErrorKind::Definition,
            None,
        )?;
    }
    for schedule in [
        ControlDecisionSchedule {
            initial_time_s: 0.0,
            decision_times_s: vec![2.0],
            end_s: 2.0,
        },
        ControlDecisionSchedule {
            initial_time_s: 0.0,
            decision_times_s: vec![1.0],
            end_s: 3.0,
        },
    ] {
        rejects(
            structures.with_operating_controls(
                vec![reference_case::definition(
                    case,
                    0,
                    &reference_case::device(0),
                )],
                vec![reference_case::initial(case, 0)],
                schedule,
                reference_case::timing(case),
            ),
            StructureInputErrorKind::Definition,
            None,
        )?;
    }
    let mut definition = reference_case::definition(case, 0, &reference_case::device(0));
    definition.signal.datum = "Another datum".into();
    rejects(
        structures.with_operating_controls(
            vec![definition],
            vec![reference_case::initial(case, 0)],
            reference_case::schedule(case),
            reference_case::timing(case),
        ),
        StructureInputErrorKind::Datum,
        Some(0),
    )
}
