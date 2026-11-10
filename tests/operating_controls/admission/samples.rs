use super::{assertions::SampleChange, fixture, reference_case, rejects};
use hydr::structures::*;

#[test]
fn decision_samples_keep_identity_datum_finiteness_and_causal_time() -> fixture::Result {
    let batch = fixture::batch("inclusive_thresholds_and_band_hold", 2)?;
    let before = batch.snapshot_data()?;
    let changes: [SampleChange; 11] = [
        (
            |sample| sample.controller_id = "other-controller".into(),
            StructureInputErrorKind::Identity,
        ),
        (
            |sample| sample.signal_id = "other-signal".into(),
            StructureInputErrorKind::Identity,
        ),
        (
            |sample| sample.support_id = "other-support".into(),
            StructureInputErrorKind::Identity,
        ),
        (
            |sample| sample.datum = "other-datum".into(),
            StructureInputErrorKind::Datum,
        ),
        (
            |sample| sample.stage_m = f64::NAN,
            StructureInputErrorKind::NonfiniteInput,
        ),
        (
            |sample| sample.stage_m = f64::INFINITY,
            StructureInputErrorKind::NonfiniteInput,
        ),
        (
            |sample| sample.observed_at_s = f64::NAN,
            StructureInputErrorKind::NonfiniteInput,
        ),
        (
            |sample| sample.available_at_s = f64::INFINITY,
            StructureInputErrorKind::NonfiniteInput,
        ),
        (
            |sample| {
                sample.observed_at_s = 0.5;
                sample.available_at_s = 0.5;
            },
            StructureInputErrorKind::UnsupportedRegime,
        ),
        (
            |sample| {
                sample.observed_at_s = 1.5;
                sample.available_at_s = 1.5;
            },
            StructureInputErrorKind::UnsupportedRegime,
        ),
        (
            |sample| sample.available_at_s = 2.0,
            StructureInputErrorKind::UnsupportedRegime,
        ),
    ];
    for (change, kind) in changes {
        let mut samples = vec![fixture::observe(1.0, 1.5, 0), fixture::observe(1.0, 1.5, 1)];
        change(&mut samples[1]);
        rejects(batch.trial_to(1.0, &samples), kind, Some(1))?;
        assert_eq!(batch.snapshot_data()?, before);
    }
    let mut samples = vec![fixture::observe(1.0, 1.5, 0), fixture::observe(1.0, 1.5, 1)];
    samples.swap(0, 1);
    rejects(
        batch.trial_to(1.0, &samples),
        StructureInputErrorKind::Identity,
        Some(0),
    )?;
    rejects(
        batch.trial_to(1.0, &samples[..1]),
        StructureInputErrorKind::OperatingPoint,
        None,
    )?;
    rejects(
        batch.trial_to(1.0, &[]),
        StructureInputErrorKind::OperatingPoint,
        None,
    )?;

    let mut samples = vec![fixture::observe(1.0, 1.5, 0), fixture::observe(1.0, 1.5, 1)];
    for sample in &mut samples {
        sample.available_at_s = -2.0;
    }
    let next = batch.trial_to(1.0, &samples)?.into_next();
    for state in next.states()? {
        assert_eq!(
            state
                .last_sample
                .ok_or("Missing accepted causal sample")?
                .available_at_s
                .to_bits(),
            (-2.0_f64).to_bits()
        );
    }
    Ok(())
}

#[test]
fn intervals_admit_only_explicit_events_and_complete_support() -> fixture::Result {
    let batch = fixture::batch("inclusive_thresholds_and_band_hold", 1)?;
    for end in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        rejects(
            batch.trial_to(end, &[]),
            StructureInputErrorKind::Definition,
            None,
        )?;
    }
    rejects(
        batch.trial_to(1.5, &[]),
        StructureInputErrorKind::UnsupportedRegime,
        None,
    )?;
    rejects(
        batch.trial_to(0.5, &[fixture::observe(0.5, 1.5, 0)]),
        StructureInputErrorKind::OperatingPoint,
        None,
    )?;
    let initial_states = batch.states()?;
    let between = batch.trial_to(0.5, &[])?.into_next();
    assert_eq!(between.states()?, initial_states);
    assert_eq!(
        between.next_event(),
        Some(ControlEvent {
            time_s: 1.0,
            decision_index: Some(0),
            terminal: false
        })
    );
    let mut terminal = between;
    for time in [1.0, 2.0, 3.0, 4.0, 5.0] {
        terminal = fixture::accept(terminal, time, 1.5)?;
    }
    assert_eq!(terminal.next_event(), None);
    rejects(
        terminal.trial_to(6.0, &[]),
        StructureInputErrorKind::UnsupportedRegime,
        None,
    )
}

#[test]
fn empty_calendars_hold_explicit_initial_commands_to_the_terminal_boundary() -> fixture::Result {
    let case = fixture::case("inclusive_thresholds_and_band_hold")?;
    let structures = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![reference_case::pump(&reference_case::device(0))?],
    )?;
    let batch = structures.with_operating_controls(
        vec![reference_case::definition(
            case,
            0,
            &reference_case::device(0),
        )],
        vec![reference_case::initial(case, 0)],
        ControlDecisionSchedule {
            initial_time_s: 0.0,
            decision_times_s: Vec::new(),
            end_s: 0.5,
        },
        reference_case::timing(case),
    )?;
    assert_eq!(
        batch.next_event(),
        Some(ControlEvent {
            time_s: 0.5,
            decision_index: None,
            terminal: true
        })
    );
    let states = batch.states()?;
    let next = batch.trial_to(0.5, &[])?.into_next();
    assert_eq!(next.states()?, states);
    assert_eq!(
        next.commands()?
            .first()
            .ok_or("Missing held initial command")?
            .command,
        ControlCommand::PumpStoppedIsolated
    );
    assert_eq!(next.next_event(), None);
    Ok(())
}
