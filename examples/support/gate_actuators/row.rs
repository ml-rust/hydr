use super::{context::Result, inputs::Case, proof::*, state};
use hydr::structures::*;
use std::io::Write;

pub const HEADER: &str = "family,case_id,policy,layout,lane,query_index,record_index,phase,stage,status,expected,element,start_bits,end_bits,accepted_time_bits,accepted_steps,rollback_unchanged,definition_json,state_json,output_json,receipt_json,event_json,error";
pub struct Frame<'a> {
    pub case: &'a Case,
    pub width: usize,
    pub query: usize,
    pub record: usize,
}
pub struct Observation<'a> {
    pub phase: &'a str,
    pub stage: &'a str,
    pub status: &'a str,
    pub expected: &'a str,
    pub element: Option<usize>,
    pub start: f64,
    pub end: f64,
    pub rollback: bool,
    pub receipts: Option<&'a [GateActuatorStepReceipt]>,
    pub event: Option<GateActuatorEvent>,
    pub error: String,
}
pub fn definition(case: &Case) -> String {
    object(&[
        ("start", bits(case.start)),
        ("end", bits(case.end)),
        ("initial", bits(case.initial)),
        ("target", bits(case.target)),
        ("opening_rate", bits(case.opening_rate)),
        ("closing_rate", bits(case.closing_rate)),
        ("maximum", bits(case.maximum)),
        (
            "absolute",
            bits(case.accuracy.maximum_absolute_opening_error_m),
        ),
        (
            "relative",
            bits(case.accuracy.maximum_relative_opening_error),
        ),
        ("window", bits(case.accuracy.maximum_arrival_window_s)),
        (
            "updates",
            format!(
                "[{}]",
                case.updates
                    .iter()
                    .map(|u| object(&[
                        ("time", bits(u.time)),
                        ("target", bits(u.target)),
                        ("available", bits(u.available))
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
    batch: Option<&GateActuatorBatch<crate::runtime::Runtime>>,
    obs: Observation<'_>,
) -> Result {
    let states = batch
        .map(GateActuatorBatch::<crate::runtime::Runtime>::states)
        .transpose()?;
    let outputs = batch
        .map(GateActuatorBatch::<crate::runtime::Runtime>::outputs)
        .transpose()?;
    if states.as_ref().is_some_and(|s| s.len() != frame.width)
        || outputs.as_ref().is_some_and(|o| o.len() != frame.width)
        || obs.receipts.is_some_and(|r| r.len() != frame.width)
    {
        return Err("Actuator observation projection shape differs".into());
    }
    let input = definition(frame.case);
    for lane in 0..frame.width {
        let state = states.as_ref().and_then(|s| s.get(lane));
        let output = outputs.as_ref().and_then(|o| o.get(lane));
        let receipt = obs.receipts.and_then(|r| r.get(lane));
        let element = if obs.status == "Success" {
            output.map(|o| o.element)
        } else {
            obs.element
        };
        let fields = [
            frame.case.family.to_string(),
            frame.case.id.to_string(),
            frame.case.policy.to_string(),
            frame.width.to_string(),
            lane.to_string(),
            frame.query.to_string(),
            frame.record.to_string(),
            obs.phase.into(),
            obs.stage.into(),
            obs.status.into(),
            obs.expected.into(),
            element.map_or_else(String::new, |v| v.to_string()),
            format!("{:016x}", obs.start.to_bits()),
            format!("{:016x}", obs.end.to_bits()),
            batch.map_or_else(String::new, |b| format!("{:016x}", b.time().to_bits())),
            batch.map_or_else(String::new, |b| b.accepted_steps().to_string()),
            obs.rollback.to_string(),
            input.clone(),
            option(state, state::state),
            option(output, state::output),
            option(receipt, state::receipt),
            option(obs.event.as_ref(), event),
            obs.error.clone(),
        ];
        writeln!(
            writer,
            "{}",
            fields.iter().map(|f| csv(f)).collect::<Vec<_>>().join(",")
        )?;
    }
    Ok(())
}
