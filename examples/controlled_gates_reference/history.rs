use super::{
    context::*,
    inputs::Case,
    row::{self, Frame, Observation},
};
use std::io::Write;

pub fn run(writer: &mut impl Write, case: &Case, width: usize) -> Result<Vec<String>> {
    let mut errors = Vec::new();
    let mut frame = Frame {
        case,
        width,
        record: 0,
    };
    let mut batch = match build(case, width) {
        Ok(batch) => batch,
        Err(error) => {
            let status = category(&error);
            row::rows(
                writer,
                &frame,
                None,
                Observation {
                    phase: "bind",
                    query: None,
                    status: &status,
                    element: error.element(),
                    start: case.start,
                    end: case.end,
                    rollback: true,
                    controls: None,
                    actuators: None,
                    event: None,
                    error: error.to_string(),
                },
            )?;
            errors.push(format!("{}: construction rejects: {error}", case.id));
            return Ok(errors);
        }
    };
    row::rows(
        writer,
        &frame,
        Some(&batch),
        Observation {
            phase: "initial",
            query: None,
            status: "Success",
            element: None,
            start: case.start,
            end: case.start,
            rollback: true,
            controls: None,
            actuators: None,
            event: batch.next_event()?,
            error: String::new(),
        },
    )?;
    frame.record += 1;
    let budget = case
        .decisions
        .len()
        .checked_add(1)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(case.queries.len()))
        .ok_or("Controlled gate event-count bound overflows")?;
    let mut intervals = 0usize;
    for (index, query) in case.queries.iter().enumerate() {
        while batch.time() < query.time {
            intervals = intervals
                .checked_add(1)
                .ok_or("Controlled gate interval counter overflows")?;
            if intervals > budget {
                return Err(format!(
                    "{}: event progression exceeds finite calendar bound",
                    case.id
                )
                .into());
            }
            let event = batch
                .next_event()?
                .ok_or("Missing controlled gate event before query")?;
            let end = event.time_s.min(query.time);
            let start = batch.time();
            if !end.is_finite() || end <= start {
                return Err("Controlled gate event does not advance time".into());
            }
            let before = batch.snapshot_data()?;
            let before_bits = format!("{before:?}");
            let supplied = samples(case, width, end);
            match batch.trial_to(end, &supplied) {
                Err(error) => {
                    let rollback = format!("{:?}", batch.snapshot_data()?) == before_bits;
                    let status = category(&error);
                    if !rollback || status != query.expected {
                        errors.push(format!(
                            "{}: rejection={status}, rollback={rollback}, expected={}",
                            case.id, query.expected
                        ));
                    }
                    row::rows(
                        writer,
                        &frame,
                        Some(&batch),
                        Observation {
                            phase: "rejected",
                            query: Some(index),
                            status: &status,
                            element: error.element(),
                            start,
                            end,
                            rollback,
                            controls: None,
                            actuators: None,
                            event: Some(event),
                            error: error.to_string(),
                        },
                    )?;
                    frame.record += 1;
                    row::rows(
                        writer,
                        &frame,
                        Some(&batch),
                        Observation {
                            phase: "query",
                            query: Some(index),
                            status: &status,
                            element: error.element(),
                            start: batch.time(),
                            end: query.time,
                            rollback,
                            controls: None,
                            actuators: None,
                            event: None,
                            error: error.to_string(),
                        },
                    )?;
                    return Ok(errors);
                }
                Ok(trial) => {
                    let controls = trial
                        .controls()
                        .last_step()?
                        .ok_or("Missing candidate control step")?;
                    let actors = trial
                        .actuators()
                        .last_step()?
                        .ok_or("Missing candidate actuator step")?;
                    let rollback = format!("{:?}", batch.snapshot_data()?) == before_bits;
                    let next = trial.into_next();
                    let steps = before
                        .controls
                        .accepted_steps
                        .checked_add(1)
                        .ok_or("Controlled gate accepted counter overflows")?;
                    if !rollback
                        || next.time().to_bits() != end.to_bits()
                        || next.controls().time().to_bits() != next.actuators().time().to_bits()
                        || next.accepted_steps() != steps
                        || next.actuators().accepted_steps() != steps
                    {
                        errors.push(format!("{}: candidate changes accepted state or child clock/count correspondence",case.id));
                    }
                    row::rows(
                        writer,
                        &frame,
                        Some(&next),
                        Observation {
                            phase: "accepted",
                            query: Some(index),
                            status: "Success",
                            element: None,
                            start,
                            end,
                            rollback,
                            controls: Some(&controls),
                            actuators: Some(&actors),
                            event: if event.time_s == end {
                                Some(event)
                            } else {
                                None
                            },
                            error: String::new(),
                        },
                    )?;
                    frame.record += 1;
                    batch = next;
                }
            }
        }
        if query.expected != "Success" {
            errors.push(format!("{}: query unexpectedly succeeds", case.id));
        }
        row::rows(
            writer,
            &frame,
            Some(&batch),
            Observation {
                phase: "query",
                query: Some(index),
                status: "Success",
                element: None,
                start: batch.time(),
                end: query.time,
                rollback: true,
                controls: None,
                actuators: None,
                event: batch.next_event()?,
                error: String::new(),
            },
        )?;
        frame.record += 1;
    }
    Ok(errors)
}
