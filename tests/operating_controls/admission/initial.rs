use super::{assertions::SampleChange, fixture, reference_case, rejects};
use hydr::structures::*;

fn pending() -> fixture::Result<ControlInitialState> {
    let case = fixture::case("positive_delays_exact_expiry")?;
    let mut state = reference_case::initial(case, 0);
    state.pending = Some(ControlPending {
        target: ControlLatch::High,
        since_s: -1.0,
    });
    state.antecedent_sample = Some(fixture::observe(0.0, 2.0, 0));
    Ok(state)
}

#[test]
fn initial_pending_memory_requires_opposing_qualifying_unexpired_antecedents() -> fixture::Result {
    let case = fixture::case("positive_delays_exact_expiry")?;
    let structures = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![reference_case::pump(&reference_case::device(0))?],
    )?;
    let bind = |state| {
        structures.with_operating_controls(
            vec![reference_case::definition(
                case,
                0,
                &reference_case::device(0),
            )],
            vec![state],
            reference_case::schedule(case),
            reference_case::timing(case),
        )
    };
    let valid = bind(pending()?)?;
    assert_eq!(
        valid
            .states()?
            .first()
            .ok_or("Missing initial controller")?
            .pending,
        pending()?.pending
    );
    let changes: [fn(&mut ControlInitialState); 8] = [
        |state| {
            state.pending = Some(ControlPending {
                target: ControlLatch::Low,
                since_s: -1.0,
            })
        },
        |state| {
            state.pending = Some(ControlPending {
                target: ControlLatch::High,
                since_s: 1.0,
            })
        },
        |state| {
            state.pending = Some(ControlPending {
                target: ControlLatch::High,
                since_s: f64::NAN,
            })
        },
        |state| {
            state.pending = Some(ControlPending {
                target: ControlLatch::High,
                since_s: -2.0,
            })
        },
        |state| state.antecedent_sample = None,
        |state| state.antecedent_sample = Some(fixture::observe(0.0, 1.5, 0)),
        |state| state.pending = None,
        |state| state.last_transition_at_s = Some(-1.0),
    ];
    for change in changes {
        let mut state = pending()?;
        change(&mut state);
        rejects(bind(state), StructureInputErrorKind::Definition, Some(0))?;
    }
    let mut state = pending()?;
    state.last_transition_at_s = Some(-2.0);
    bind(state)?;

    let mut definition = reference_case::definition(case, 0, &reference_case::device(0));
    definition.high_qualification_delay_s = 0.0;
    rejects(
        structures.with_operating_controls(
            vec![definition],
            vec![pending()?],
            reference_case::schedule(case),
            reference_case::timing(case),
        ),
        StructureInputErrorKind::Definition,
        Some(0),
    )?;
    Ok(())
}

#[test]
fn antecedent_packets_preserve_specific_causality_and_identity_categories() -> fixture::Result {
    let case = fixture::case("positive_delays_exact_expiry")?;
    let structures = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![reference_case::pump(&reference_case::device(0))?],
    )?;
    let changes: [SampleChange; 5] = [
        (
            |sample| sample.controller_id = "other-controller".into(),
            StructureInputErrorKind::Identity,
        ),
        (
            |sample| sample.datum = "other-datum".into(),
            StructureInputErrorKind::Datum,
        ),
        (
            |sample| sample.available_at_s = 0.5,
            StructureInputErrorKind::UnsupportedRegime,
        ),
        (
            |sample| sample.available_at_s = f64::NAN,
            StructureInputErrorKind::NonfiniteInput,
        ),
        (
            |sample| sample.stage_m = f64::INFINITY,
            StructureInputErrorKind::NonfiniteInput,
        ),
    ];
    for (change, kind) in changes {
        let mut state = pending()?;
        let sample = state
            .antecedent_sample
            .as_mut()
            .ok_or("Missing antecedent fixture")?;
        change(sample);
        rejects(
            structures.with_operating_controls(
                vec![reference_case::definition(
                    case,
                    0,
                    &reference_case::device(0),
                )],
                vec![state],
                reference_case::schedule(case),
                reference_case::timing(case),
            ),
            kind,
            Some(0),
        )?;
    }
    let mut state = pending()?;
    state.antecedent_sample = Some(fixture::observe(0.5, 2.0, 0));
    rejects(
        structures.with_operating_controls(
            vec![reference_case::definition(
                case,
                0,
                &reference_case::device(0),
            )],
            vec![state],
            reference_case::schedule(case),
            reference_case::timing(case),
        ),
        StructureInputErrorKind::Definition,
        Some(0),
    )?;
    Ok(())
}

#[test]
fn transition_chronology_and_known_initial_observation_spacing_are_checked() -> fixture::Result {
    let case = fixture::case("positive_delays_exact_expiry")?;
    let structures = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![reference_case::pump(&reference_case::device(0))?],
    )?;
    for time in [1.0, f64::INFINITY] {
        let mut state = reference_case::initial(case, 0);
        state.last_transition_at_s = Some(time);
        rejects(
            structures.with_operating_controls(
                vec![reference_case::definition(
                    case,
                    0,
                    &reference_case::device(0),
                )],
                vec![state],
                reference_case::schedule(case),
                reference_case::timing(case),
            ),
            StructureInputErrorKind::Definition,
            Some(0),
        )?;
    }
    let mut state = reference_case::initial(case, 0);
    state.antecedent_sample = Some(fixture::observe(0.0, 1.5, 0));
    state.last_transition_at_s = Some(0.5);
    let schedule = ControlDecisionSchedule {
        initial_time_s: 1.0,
        decision_times_s: vec![2.0],
        end_s: 2.0,
    };
    let mut timing = reference_case::timing(case);
    timing.maximum_decision_gap_s = 2.0;
    rejects(
        structures.with_operating_controls(
            vec![reference_case::definition(
                case,
                0,
                &reference_case::device(0),
            )],
            vec![state],
            schedule,
            timing,
        ),
        StructureInputErrorKind::Definition,
        Some(0),
    )?;

    let mut state = pending()?;
    state.antecedent_sample = Some(fixture::observe(-1.0, 2.0, 0));
    rejects(
        structures.with_operating_controls(
            vec![reference_case::definition(
                case,
                0,
                &reference_case::device(0),
            )],
            vec![state.clone()],
            reference_case::schedule(case),
            reference_case::timing(case),
        ),
        StructureInputErrorKind::Definition,
        Some(0),
    )?;
    let mut timing = reference_case::timing(case);
    timing.maximum_decision_gap_s = 2.0;
    let admitted = structures.with_operating_controls(
        vec![reference_case::definition(
            case,
            0,
            &reference_case::device(0),
        )],
        vec![state],
        reference_case::schedule(case),
        timing,
    )?;
    let next = admitted
        .trial_to(1.0, &[fixture::observe(1.0, 2.0, 0)])?
        .into_next();
    assert_eq!(
        next.states()?
            .first()
            .ok_or("Missing admitted antecedent controller")?
            .latch,
        ControlLatch::High
    );
    Ok(())
}
