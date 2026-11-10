use super::{context::Result, encoding::*, inputs::Case, projection};
use hydr::structures::*;
use std::io::Write;

pub const HEADER: &str = "family,case_id,layout,lane,phase,query_index,record_index,time_bits,accepted_steps,status,element,restart_mode,restart_equal,unchanged,input_json,heads_json,command_json,control_state_json,control_step_json,wrapper_json,wrapper_operation_json,proposal_operation_json,source_json,hydraulic_json,error";
pub struct Frame<'a> {
    pub case: &'a Case,
    pub family: &'a str,
    pub width: usize,
    pub record: usize,
}
pub struct Observation<'a> {
    pub phase: &'a str,
    pub query: Option<usize>,
    pub status: &'a str,
    pub element: Option<usize>,
    pub mode: &'a str,
    pub restart_equal: bool,
    pub unchanged: bool,
    pub control_step: Option<&'a ControlStepDiagnostics>,
    pub heads: Option<&'a [StructureHeads]>,
    pub point: Option<&'a [OperatingSourceOutput]>,
    pub operations: Option<&'a [StructureOperation]>,
    pub error: String,
}
fn input(c: &Case) -> String {
    object(&[
        ("end", bits(c.end)),
        ("initial_latch", text(format!("{:?}", c.initial_latch))),
        ("low", projection::command_value(&c.low)),
        ("high", projection::command_value(&c.high)),
        ("high_delay", bits(c.high_delay)),
        ("low_delay", bits(c.low_delay)),
        ("gap", bits(c.gap)),
        ("affinity", c.affinity.to_string()),
        ("adoption", bits(c.adoption)),
        (
            "decisions",
            format!(
                "[{}]",
                c.decisions
                    .iter()
                    .map(|d| object(&[
                        ("time", bits(d.time)),
                        ("available", bits(d.available)),
                        ("stage", bits(d.stage))
                    ]))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
        (
            "curve",
            format!(
                "[{},{}]",
                object(&[("head", bits(0.0)), ("q", bits(4.0))]),
                object(&[("head", bits(4.0)), ("q", bits(0.0))])
            ),
        ),
        ("suction", bits(0.0)),
        (
            "minimum_speed",
            if c.affinity { bits(0.5) } else { "null".into() },
        ),
        (
            "maximum_speed",
            if c.affinity { bits(2.0) } else { "null".into() },
        ),
    ])
}
fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
pub fn rows(
    writer: &mut impl Write,
    frame: &Frame<'_>,
    batch: &ControlledPumpBatch,
    o: Observation<'_>,
) -> Result {
    let commands = batch.controls().commands()?;
    let states = batch.controls().states()?;
    let response = batch.outputs()?;
    let actual = batch.operations()?;
    if commands.len() != frame.width
        || states.len() != frame.width
        || response.len() != frame.width
        || actual.len() != frame.width
        || batch.definitions().len() != frame.width
        || o.heads.is_some_and(|v| v.len() != frame.width)
        || o.point.is_some_and(|v| v.len() != frame.width)
        || o.operations.is_some_and(|v| v.len() != frame.width)
        || o.control_step.is_some_and(|s| {
            s.left_commands.len() != frame.width
                || s.right_commands.len() != frame.width
                || (!s.decisions.is_empty() && s.decisions.len() != frame.width)
        })
    {
        return Err("Controlled pump projection shape differs".into());
    }
    let input = input(frame.case);
    for lane in 0..frame.width {
        let command = commands.get(lane).ok_or("Missing command row")?;
        let state = states.get(lane).ok_or("Missing control state row")?;
        let response = response.get(lane).ok_or("Missing actual response row")?;
        let definition = batch
            .definitions()
            .get(lane)
            .ok_or("Missing response definition row")?;
        let actual = actual.get(lane).ok_or("Missing actual operation row")?;
        let point = o.point.and_then(|v| v.get(lane));
        let fields = [
            frame.family.to_string(),
            frame.case.id.to_string(),
            frame.width.to_string(),
            lane.to_string(),
            o.phase.into(),
            o.query.map_or_else(String::new, |v| v.to_string()),
            frame.record.to_string(),
            format!("{:016x}", batch.time().to_bits()),
            batch.accepted_steps().to_string(),
            o.status.into(),
            (if o.status == "Success" {
                Some(response.element)
            } else {
                o.element
            })
            .map_or_else(String::new, |v| v.to_string()),
            o.mode.into(),
            o.restart_equal.to_string(),
            o.unchanged.to_string(),
            input.clone(),
            option(o.heads.and_then(|v| v.get(lane)), projection::heads),
            projection::command(command),
            projection::state(state),
            o.control_step
                .map_or_else(|| "null".into(), |s| projection::step(s, lane)),
            projection::response(response, definition),
            projection::operation(&actual.operating_point),
            option(o.operations.and_then(|v| v.get(lane)), |p| {
                projection::operation(&p.operating_point)
            }),
            option(point, |p| projection::source(&p.source)),
            option(point, |p| projection::hydraulic(&p.hydraulic)),
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
