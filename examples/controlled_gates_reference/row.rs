use super::{actuator_state, context::Result, control, inputs::Case, proof::*};
use hydr::structures::*;
use std::io::Write;

pub const HEADER: &str = "case_id,layout,lane,phase,query_index,record_index,status,element,start_bits,end_bits,accepted_time_bits,accepted_steps,control_steps,actuator_steps,rollback_unchanged,input_json,command_json,control_state_json,control_step_json,actuator_output_json,actuator_state_json,actuator_receipt_json,event_json,error";
pub struct Frame<'a> {
    pub case: &'a Case,
    pub width: usize,
    pub record: usize,
}
pub struct Observation<'a> {
    pub phase: &'a str,
    pub query: Option<usize>,
    pub status: &'a str,
    pub element: Option<usize>,
    pub start: f64,
    pub end: f64,
    pub rollback: bool,
    pub controls: Option<&'a ControlStepDiagnostics>,
    pub actuators: Option<&'a GateActuatorStepDiagnostics>,
    pub event: Option<ControlledGateEvent>,
    pub error: String,
}
fn combined(e: &ControlledGateEvent) -> String {
    object(&[
        ("time", bits(e.time_s)),
        (
            "decision",
            e.decision_index.map_or_else(|| "null".into(), number),
        ),
        ("lower", e.arrival_lower_boundary.to_string()),
        ("upper", e.arrival_upper_boundary.to_string()),
        ("terminal", e.terminal.to_string()),
    ])
}
fn input(c: &Case) -> String {
    object(&[
        ("start", bits(c.start)),
        ("end", bits(c.end)),
        ("initial", bits(c.initial)),
        ("target", bits(c.target)),
        ("opening", bits(c.opening_rate)),
        ("closing", bits(c.closing_rate)),
        ("low", bits(c.low)),
        ("high", bits(c.high)),
        ("lower_stage", bits(c.lower_stage)),
        ("upper_stage", bits(c.upper_stage)),
        ("high_delay", bits(c.high_delay)),
        ("low_delay", bits(c.low_delay)),
        ("gap", bits(c.gap)),
        ("lateness", bits(c.lateness)),
        (
            "absolute",
            bits(c.accuracy.maximum_absolute_opening_error_m),
        ),
        ("relative", bits(c.accuracy.maximum_relative_opening_error)),
        ("window", bits(c.accuracy.maximum_arrival_window_s)),
        (
            "decisions",
            format!(
                "[{}]",
                c.decisions
                    .iter()
                    .map(|d| object(&[
                        ("time", bits(d.time)),
                        ("available", bits(d.available)),
                        ("stage", bits(d.stage)),
                        ("admitted", d.admitted.to_string())
                    ]))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
    ])
}
fn csv(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}
pub fn rows(
    writer: &mut impl Write,
    frame: &Frame<'_>,
    batch: Option<&ControlledGateBatch<crate::runtime::Runtime>>,
    o: Observation<'_>,
) -> Result {
    let commands = batch.map(|b| b.controls().commands()).transpose()?;
    let controls = batch.map(|b| b.controls().states()).transpose()?;
    let actors = batch.map(|b| b.actuators().states()).transpose()?;
    let outputs = batch.map(|b| b.actuators().outputs()).transpose()?;
    if commands.as_ref().is_some_and(|v| v.len() != frame.width)
        || controls.as_ref().is_some_and(|v| v.len() != frame.width)
        || actors.as_ref().is_some_and(|v| v.len() != frame.width)
        || outputs.as_ref().is_some_and(|v| v.len() != frame.width)
        || o.controls.is_some_and(|s| {
            s.left_commands.len() != frame.width
                || s.right_commands.len() != frame.width
                || (!s.decisions.is_empty() && s.decisions.len() != frame.width)
        })
        || o.actuators.is_some_and(|s| s.receipts.len() != frame.width)
    {
        return Err("Controlled gate projection shape differs".into());
    }
    let inputs = input(frame.case);
    let event = object(&[
        ("combined", option(o.event.as_ref(), combined)),
        (
            "actuator",
            option(o.actuators.and_then(|a| a.event.as_ref()), event),
        ),
    ]);
    for lane in 0..frame.width {
        let command = commands.as_ref().and_then(|v| v.get(lane));
        let control = controls.as_ref().and_then(|v| v.get(lane));
        let actor = actors.as_ref().and_then(|v| v.get(lane));
        let output = outputs.as_ref().and_then(|v| v.get(lane));
        let receipt = o.actuators.and_then(|s| s.receipts.get(lane));
        let fields = [
            frame.case.id.to_string(),
            frame.width.to_string(),
            lane.to_string(),
            o.phase.into(),
            o.query.map_or_else(String::new, |v| v.to_string()),
            frame.record.to_string(),
            o.status.into(),
            (if o.status == "Success" {
                output.map(|v| v.element)
            } else {
                o.element
            })
            .map_or_else(String::new, |v| v.to_string()),
            format!("{:016x}", o.start.to_bits()),
            format!("{:016x}", o.end.to_bits()),
            batch.map_or_else(String::new, |b| format!("{:016x}", b.time().to_bits())),
            batch.map_or_else(String::new, |b| b.accepted_steps().to_string()),
            batch.map_or_else(String::new, |b| b.controls().accepted_steps().to_string()),
            batch.map_or_else(String::new, |b| b.actuators().accepted_steps().to_string()),
            o.rollback.to_string(),
            inputs.clone(),
            option(command, control::command),
            option(control, control::state),
            o.controls
                .map_or_else(|| "null".into(), |s| control::step(s, lane)),
            option(output, actuator_state::output),
            option(actor, actuator_state::state),
            option(receipt, actuator_state::receipt),
            event.clone(),
            o.error.clone(),
        ];
        writeln!(
            writer,
            "{}",
            fields.iter().map(|v| csv(v)).collect::<Vec<_>>().join(",")
        )?;
    }
    Ok(())
}
