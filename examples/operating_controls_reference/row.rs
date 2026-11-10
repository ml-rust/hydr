use super::{reference_case::*, reference_inputs::Case};
use hydr::structures::*;
use std::io::Write;

pub const HEADER: &str = "family,case_id,layout,lane,controller_id,structure_id,phase,decision_index,status,element,start_bits,end_bits,accepted_time_bits,accepted_steps,left_latch,right_latch,left_command_kind,left_command_bits,right_command_kind,right_command_bits,pending_target,pending_since_bits,last_sample_time_bits,last_sample_available_bits,last_sample_stage_bits,last_transition_time_bits,transition_count,qualifies,transitioned,timer_start_bits,timer_evaluated_bits,timer_delay_bits,elapsed_high_bits,elapsed_low_bits,ready,allowed_high_bits,allowed_low_bits,allowed_overflow,within_lateness,sample_time_bits,sample_available_bits,sample_stage_bits,threshold_lower_bits,threshold_upper_bits,high_delay_bits,low_delay_bits,cadence_bits,lateness_bits";

pub struct Row<'a> {
    pub case: &'a Case,
    pub width: usize,
    pub lane: usize,
    pub phase: &'static str,
    pub decision_index: Option<usize>,
    pub status: String,
    pub element: Option<usize>,
    pub start: f64,
    pub end: f64,
    pub accepted_time: Option<f64>,
    pub accepted_steps: Option<usize>,
    pub left: Option<&'a ControlCommandOutput>,
    pub right: Option<&'a ControlCommandOutput>,
    pub state: Option<&'a ControlState>,
    pub receipt: Option<&'a ControlDecisionReceipt>,
    pub sample: Option<&'a ControlSample>,
}

fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}
fn optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, bits)
}
fn index(value: Option<usize>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}
fn boolean(value: Option<bool>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}
fn latch(value: Option<ControlLatch>) -> String {
    value.map_or_else(String::new, |value| format!("{value:?}"))
}

fn command(value: Option<&ControlCommandOutput>) -> (String, String) {
    match value.map(|value| value.command) {
        None => (String::new(), String::new()),
        Some(ControlCommand::PumpStoppedIsolated) => ("PumpStoppedIsolated".into(), String::new()),
        Some(ControlCommand::PumpRunning { speed_ratio }) => {
            ("PumpRunning".into(), bits(speed_ratio))
        }
        Some(ControlCommand::GateTarget { opening_m }) => ("GateTarget".into(), bits(opening_m)),
    }
}

pub fn write(writer: &mut impl Write, row: Row<'_>) -> Result {
    let (left_kind, left_value) = command(row.left);
    let (right_kind, right_value) = command(row.right);
    let pending = row.state.and_then(|state| state.pending);
    let last_sample = row.state.and_then(|state| state.last_sample.as_ref());
    let timer = row.receipt.and_then(|receipt| receipt.timer.as_ref());
    let allowed = timer.and_then(|timer| timer.allowed_elapsed.as_ref());
    let case = row.case;
    let fields = [
        case.family.into(),
        case.id.into(),
        row.width.to_string(),
        row.lane.to_string(),
        controller(row.lane),
        device(row.lane),
        row.phase.into(),
        index(row.decision_index),
        row.status,
        index(row.element),
        bits(row.start),
        bits(row.end),
        optional(row.accepted_time),
        index(row.accepted_steps),
        latch(row.left.map(|value| value.latch)),
        latch(row.right.map(|value| value.latch)),
        left_kind,
        left_value,
        right_kind,
        right_value,
        latch(pending.map(|value| value.target)),
        optional(pending.map(|value| value.since_s)),
        optional(last_sample.map(|value| value.observed_at_s)),
        optional(last_sample.map(|value| value.available_at_s)),
        optional(last_sample.map(|value| value.stage_m)),
        optional(row.state.and_then(|state| state.last_transition_at_s)),
        index(row.state.map(|state| state.transition_count)),
        boolean(row.receipt.map(|value| value.qualifies)),
        boolean(row.receipt.map(|value| value.transitioned)),
        optional(timer.map(|value| value.started_at_s)),
        optional(timer.map(|value| value.evaluated_at_s)),
        optional(timer.map(|value| value.delay_s)),
        optional(timer.map(|value| value.elapsed_high_s)),
        optional(timer.map(|value| value.elapsed_low_s)),
        boolean(timer.map(|value| value.ready)),
        optional(allowed.map(|value| value.high_s)),
        optional(allowed.map(|value| value.low_s)),
        boolean(timer.map(|value| value.allowed_elapsed_overflow)),
        boolean(timer.map(|value| value.within_lateness)),
        optional(row.sample.map(|value| value.observed_at_s)),
        optional(row.sample.map(|value| value.available_at_s)),
        optional(row.sample.map(|value| value.stage_m)),
        bits(case.lower),
        bits(case.upper),
        bits(case.high_delay),
        bits(case.low_delay),
        bits(case.maximum_gap),
        bits(case.lateness),
    ];
    writeln!(writer, "{}", fields.join(","))?;
    Ok(())
}
