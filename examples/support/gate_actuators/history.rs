use super::{
    context::*,
    inputs::{Case, Query},
    row::{self, Frame, Observation},
};
use hydr::structures::*;
use std::io::Write;

pub fn run(
    writer: &mut impl Write,
    case: &Case,
    width: usize,
    query_index: usize,
    query: &Query,
) -> Result<Vec<String>> {
    let mut errors = Vec::new();
    let mut frame = Frame {
        case,
        width,
        query: query_index,
        record: 0,
    };
    let mut batch = match build(case, width) {
        Ok(batch) => batch,
        Err(error) => {
            let status = category(&error);
            if status != case.binding_expected {
                errors.push(format!(
                    "{}: construction {status}, expected {}",
                    case.id, case.binding_expected
                ));
            }
            if status != query.expected {
                errors.push(format!(
                    "{}: binding {status}, expected {}",
                    case.id, query.expected
                ));
            }
            row::rows(
                writer,
                &frame,
                None,
                Observation {
                    phase: "result",
                    stage: "bind",
                    status: &status,
                    expected: query.expected,
                    element: error.element(),
                    start: case.start,
                    end: query.time,
                    rollback: true,
                    receipts: None,
                    event: None,
                    error: error.to_string(),
                },
            )?;
            return Ok(errors);
        }
    };
    let initial_event = match batch.next_event() {
        Ok(event) => event,
        Err(error) => {
            return rejected(
                writer,
                &mut frame,
                &batch,
                query,
                &error,
                "next_event",
                &mut errors,
            );
        }
    };
    row::rows(
        writer,
        &frame,
        Some(&batch),
        Observation {
            phase: "initial",
            stage: "observed",
            status: "Success",
            expected: "Success",
            element: None,
            start: case.start,
            end: case.start,
            rollback: true,
            receipts: None,
            event: initial_event,
            error: String::new(),
        },
    )?;
    frame.record += 1;
    let budget = case
        .updates
        .len()
        .checked_add(1)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(4))
        .ok_or("Actuator event-count bound overflows")?;
    let mut count = 0usize;
    while batch.time() < query.time {
        count = count
            .checked_add(1)
            .ok_or("Actuator event iteration counter overflows")?;
        if count > budget {
            return Err(format!(
                "{}: event progression exceeds finite command/episode bound",
                case.id
            )
            .into());
        }
        let event = match batch.next_event() {
            Ok(event) => event,
            Err(error) => {
                return rejected(
                    writer,
                    &mut frame,
                    &batch,
                    query,
                    &error,
                    "next_event",
                    &mut errors,
                );
            }
        };
        let end = event
            .as_ref()
            .map_or(query.time, |e| e.time_s.min(query.time));
        if !end.is_finite() || end <= batch.time() {
            return Err(format!("{}: nonprogressing actuator event", case.id).into());
        }
        let packets = updates(case, width, end);
        let before = batch.snapshot_data()?;
        let start = batch.time();
        match batch.trial_to(end, &packets) {
            Err(error) => {
                let rollback = batch.snapshot_data()? == before;
                if !rollback {
                    errors.push(format!(
                        "{}: rejected candidate changes accepted package",
                        case.id
                    ));
                }
                let status = category(&error);
                row::rows(
                    writer,
                    &frame,
                    Some(&batch),
                    Observation {
                        phase: "rejected",
                        stage: "trial",
                        status: &status,
                        expected: query.expected,
                        element: error.element(),
                        start,
                        end,
                        rollback,
                        receipts: None,
                        event,
                        error: error.to_string(),
                    },
                )?;
                frame.record += 1;
                return rejected(
                    writer,
                    &mut frame,
                    &batch,
                    query,
                    &error,
                    "trial",
                    &mut errors,
                );
            }
            Ok(trial) => {
                let diagnostics = trial.diagnostics()?;
                let rollback = batch.snapshot_data()? == before;
                if !rollback {
                    errors.push(format!(
                        "{}: candidate creation changes accepted package",
                        case.id
                    ));
                }
                let candidate = trial.into_next();
                let expected_steps = before
                    .accepted_steps
                    .checked_add(1)
                    .ok_or("Actuator accepted interval counter overflows")?;
                if candidate.time().to_bits() != end.to_bits()
                    || candidate.accepted_steps() != expected_steps
                {
                    errors.push(format!(
                        "{}: accepted time or interval count differs",
                        case.id
                    ));
                }
                row::rows(
                    writer,
                    &frame,
                    Some(&candidate),
                    Observation {
                        phase: "accepted",
                        stage: "trial",
                        status: "Success",
                        expected: "Success",
                        element: None,
                        start,
                        end,
                        rollback,
                        receipts: Some(&diagnostics.receipts),
                        event: diagnostics.event,
                        error: String::new(),
                    },
                )?;
                frame.record += 1;
                batch = candidate;
            }
        }
    }
    if query.expected != "Success" {
        errors.push(format!(
            "{}: query unexpectedly succeeds, expected {}",
            case.id, query.expected
        ));
    }
    row::rows(
        writer,
        &frame,
        Some(&batch),
        Observation {
            phase: "result",
            stage: "observed",
            status: "Success",
            expected: query.expected,
            element: None,
            start: batch.time(),
            end: query.time,
            rollback: true,
            receipts: None,
            event: batch.next_event()?,
            error: String::new(),
        },
    )?;
    Ok(errors)
}

fn rejected(
    writer: &mut impl Write,
    frame: &mut Frame<'_>,
    batch: &GateActuatorBatch,
    query: &Query,
    error: &StructureError,
    stage: &str,
    errors: &mut Vec<String>,
) -> Result<Vec<String>> {
    let status = category(error);
    if status != query.expected {
        errors.push(format!(
            "{}: query {status}, expected {}",
            frame.case.id, query.expected
        ));
    }
    row::rows(
        writer,
        frame,
        Some(batch),
        Observation {
            phase: "result",
            stage,
            status: &status,
            expected: query.expected,
            element: error.element(),
            start: batch.time(),
            end: query.time,
            rollback: true,
            receipts: None,
            event: None,
            error: error.to_string(),
        },
    )?;
    Ok(std::mem::take(errors))
}
