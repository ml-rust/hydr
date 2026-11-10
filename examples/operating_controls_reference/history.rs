use super::{
    check,
    reference_case::*,
    reference_inputs::Case,
    row::{self, Row},
};
use hydr::structures::*;
use std::io::Write;

pub fn run(
    writer: &mut impl Write,
    case: &Case,
    width: usize,
    mut batch: OperatingControlBatch<crate::runtime::Runtime>,
) -> Result<Vec<String>> {
    let mut errors = Vec::new();
    for (index, expected) in case.decisions.iter().enumerate() {
        let start = batch.time();
        let before = batch.snapshot_data()?;
        let left = batch.commands()?;
        let before_states = batch.states()?;
        let samples = (0..width)
            .map(|lane| sample(expected.sample, lane))
            .collect::<Vec<_>>();
        match batch.trial_to(expected.sample.observed, &samples) {
            Err(error) => {
                if let Err(error) = check_error(&error, expected.expected, Some(0)) {
                    errors.push(error.to_string());
                }
                if batch.snapshot_data()? != before {
                    errors.push(format!(
                        "{}: rejected trial changes accepted package",
                        case.id
                    ));
                }
                for (lane, ((left, state), sample)) in
                    left.iter().zip(&before_states).zip(&samples).enumerate()
                {
                    if let Err(error) = check::state(case, expected, state, lane) {
                        errors.push(error.to_string());
                    }
                    row::write(
                        writer,
                        Row {
                            case,
                            width,
                            lane,
                            phase: "decision",
                            decision_index: Some(index),
                            status: category(&error),
                            element: error.element(),
                            start,
                            end: expected.sample.observed,
                            accepted_time: Some(batch.time()),
                            accepted_steps: Some(batch.accepted_steps()),
                            left: Some(left),
                            right: Some(left),
                            state: Some(state),
                            receipt: None,
                            sample: Some(sample),
                        },
                    )?;
                }
            }
            Ok(trial) => {
                if expected.expected != "success" {
                    errors.push(format!(
                        "{} decision={index}: trial unexpectedly succeeds",
                        case.id
                    ));
                }
                let diagnostics = trial.diagnostics()?;
                let states = trial.states()?;
                let commands = trial.commands()?;
                if diagnostics.decisions.len() != width
                    || diagnostics.left_commands.len() != width
                    || diagnostics.right_commands.len() != width
                    || states.len() != width
                    || commands.len() != width
                {
                    return Err("Decision control projection shape differs".into());
                }
                if diagnostics.start_s.to_bits() != start.to_bits()
                    || diagnostics.end_s.to_bits() != expected.sample.observed.to_bits()
                    || diagnostics.decision_index != Some(index)
                {
                    errors.push(format!("{}: decision support metadata differs", case.id));
                }
                for lane in 0..width {
                    let left = &diagnostics.left_commands[lane];
                    let right = &diagnostics.right_commands[lane];
                    let receipt = &diagnostics.decisions[lane];
                    if let Err(error) = check::command(case, left, start, lane) {
                        errors.push(error.to_string());
                    }
                    if let Err(error) = check::command(case, right, expected.sample.observed, lane)
                    {
                        errors.push(error.to_string());
                    }
                    if let Err(error) = check::decision(case, expected, receipt, lane) {
                        errors.push(error.to_string());
                    }
                    if left.latch != before_states[lane].latch
                        || right.latch != states[lane].latch
                        || receipt.before != before_states[lane]
                        || receipt.after != states[lane]
                        || commands[lane] != *right
                    {
                        errors.push(format!(
                            "{} lane={lane}: left/right state bridge differs",
                            case.id
                        ));
                    }
                    row::write(
                        writer,
                        Row {
                            case,
                            width,
                            lane,
                            phase: "decision",
                            decision_index: diagnostics.decision_index,
                            status: "success".into(),
                            element: Some(receipt.element),
                            start,
                            end: diagnostics.end_s,
                            accepted_time: Some(diagnostics.end_s),
                            accepted_steps: Some(before.accepted_steps + 1),
                            left: Some(left),
                            right: Some(right),
                            state: Some(&states[lane]),
                            receipt: Some(receipt),
                            sample: Some(&samples[lane]),
                        },
                    )?;
                }
                if batch.snapshot_data()? != before {
                    errors.push(format!(
                        "{}: candidate creation changes accepted package",
                        case.id
                    ));
                }
                batch = trial.into_next();
                if batch.time().to_bits() != expected.accepted_time.to_bits()
                    || batch.accepted_steps() != before.accepted_steps + 1
                {
                    errors.push(format!("{}: accepted time or counter differs", case.id));
                }
                if batch.last_step()?.as_ref() != Some(&diagnostics) {
                    errors.push(format!("{}: retained step differs", case.id));
                }
            }
        }
    }
    if batch.time() < case.end
        && batch
            .next_event()
            .is_some_and(|event| event.decision_index.is_none())
    {
        let before = batch.snapshot_data()?;
        let start = batch.time();
        let trial = batch.trial_to(case.end, &[])?;
        let diagnostics = trial.diagnostics()?;
        let commands = trial.commands()?;
        let states = trial.states()?;
        if diagnostics.decision_index.is_some()
            || !diagnostics.decisions.is_empty()
            || states != before.states
        {
            errors.push(format!(
                "{}: terminal-only interval changes relay state",
                case.id
            ));
        }
        for lane in 0..width {
            row::write(
                writer,
                Row {
                    case,
                    width,
                    lane,
                    phase: "terminal",
                    decision_index: None,
                    status: "success".into(),
                    element: Some(commands[lane].element),
                    start,
                    end: case.end,
                    accepted_time: Some(case.end),
                    accepted_steps: Some(before.accepted_steps + 1),
                    left: Some(&diagnostics.left_commands[lane]),
                    right: Some(&commands[lane]),
                    state: Some(&states[lane]),
                    receipt: None,
                    sample: None,
                },
            )?;
        }
        batch = trial.into_next();
        if batch.next_event().is_some() {
            errors.push(format!(
                "{}: terminal support retains a future event",
                case.id
            ));
        }
    }
    Ok(errors)
}
