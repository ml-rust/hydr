use super::{
    context::*,
    inputs::{Case, Support},
    point,
    row::{self, Frame, Observation},
};
use hydr::structures::*;
use std::io::Write;

pub fn run(writer: &mut impl Write, case: &Case, width: usize) -> Result<Vec<String>> {
    let mut errors = Vec::new();
    let (structures, mut pump) = build(case, width)?;
    let registry =
        structures.with_operating_sources(vec![OperatingSourceBinding::ControlledPump {
            source_id: SOURCE.into(),
            pump: &pump,
        }])?;
    let mut frame = Frame {
        case,
        family: "history",
        width,
        record: 0,
    };
    row::rows(
        writer,
        &frame,
        &pump,
        Observation {
            phase: "initial",
            query: None,
            status: "Success",
            element: None,
            mode: "Direct",
            restart_equal: true,
            unchanged: true,
            control_step: None,
            heads: None,
            point: None,
            operations: None,
            error: String::new(),
        },
    )?;
    frame.record += 1;
    for (index, time) in case.queries.iter().copied().enumerate() {
        while pump.time() < time {
            let event = pump
                .next_event()
                .ok_or("Missing sampled pump event before query")?;
            let end = event.time_s.min(time);
            if !end.is_finite() || end <= pump.time() {
                return Err("Sampled pump event fails to advance time".into());
            }
            let before = pump.snapshot_data()?;
            let trial = pump.trial_to(end, &samples(case, width, end))?;
            let unchanged = format!("{:?}", pump.snapshot_data()?) == format!("{before:?}");
            let next = trial.into_next();
            let step = next
                .controls()
                .last_step()?
                .ok_or("Missing sampled pump control step")?;
            let expected = before
                .controls
                .accepted_steps
                .checked_add(1)
                .ok_or("Sampled pump interval count overflows")?;
            if !unchanged
                || next.accepted_steps() != expected
                || next.time().to_bits() != end.to_bits()
            {
                errors.push(format!(
                    "{}: candidate mutates accepted state or interval metadata",
                    case.id
                ));
            }
            row::rows(
                writer,
                &frame,
                &next,
                Observation {
                    phase: "accepted",
                    query: Some(index),
                    status: "Success",
                    element: None,
                    mode: "Direct",
                    restart_equal: true,
                    unchanged,
                    control_step: Some(&step),
                    heads: None,
                    point: None,
                    operations: None,
                    error: String::new(),
                },
            )?;
            frame.record += 1;
            pump = next;
        }
        let (restored, mode, equal) = point::restart(case, width, pump, index)?;
        pump = restored;
        if !equal {
            errors.push(format!(
                "{}: restored response differs from accepted continuation",
                case.id
            ));
        }
        errors.extend(point::observe(
            writer,
            &frame,
            &pump,
            &registry,
            point::Query {
                index,
                expected: "Success",
                donor: Support::Wet(2.0),
                receiver: Support::Wet(3.0),
                mode,
                equal,
                phase: "query",
            },
        )?);
        frame.record += 1;
    }
    Ok(errors)
}
