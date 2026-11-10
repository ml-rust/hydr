use super::{
    context::*,
    inputs::{Case, Support},
    row::{self, Frame, Observation},
};
use hydr::structures::*;
use std::io::Write;

pub struct Query<'a> {
    pub index: usize,
    pub expected: &'a str,
    pub donor: Support,
    pub receiver: Support,
    pub mode: &'a str,
    pub equal: bool,
    pub phase: &'a str,
}
pub fn observe(
    writer: &mut impl Write,
    frame: &Frame<'_>,
    pump: &ControlledPumpBatch<crate::runtime::Runtime>,
    registry: &OperatingSourceBatch<crate::runtime::Runtime>,
    query: Query<'_>,
) -> Result<Vec<String>> {
    let mut errors = Vec::new();
    let heads = heads(frame.width, query.donor, query.receiver);
    let before = format!("{:?}", pump.snapshot_data()?);
    let proposal = registry.propose_at(
        pump.time(),
        &heads,
        &[],
        &[OperatingSourceFrame::ControlledPump {
            source_id: SOURCE,
            pump,
        }],
    );
    let unchanged = format!("{:?}", pump.snapshot_data()?) == before;
    if !unchanged {
        errors.push(format!(
            "{}: source proposal changes accepted response",
            frame.case.id
        ));
    }
    match proposal {
        Err(error) => {
            let status = category(&error);
            if status != query.expected {
                errors.push(format!(
                    "{}: source category={status}, expected={}",
                    frame.case.id, query.expected
                ));
            }
            row::rows(
                writer,
                frame,
                pump,
                Observation {
                    phase: query.phase,
                    query: Some(query.index),
                    status: &status,
                    element: error.element(),
                    mode: query.mode,
                    restart_equal: query.equal,
                    unchanged,
                    control_step: None,
                    heads: Some(&heads),
                    point: None,
                    operations: None,
                    error: error.to_string(),
                },
            )?;
        }
        Ok(proposal) => {
            if query.expected != "Success" {
                errors.push(format!(
                    "{}: source proposal unexpectedly succeeds",
                    frame.case.id
                ));
            }
            let outputs = proposal.outputs()?;
            let operations = proposal.operations()?;
            let actual = pump.operations()?;
            if operations != actual
                || outputs.len() != actual.len()
                || outputs
                    .iter()
                    .zip(&actual)
                    .any(|(o, a)| o.hydraulic.operation != a.operating_point)
            {
                errors.push(format!(
                    "{}: source operation projection differs from actual response",
                    frame.case.id
                ));
            }
            row::rows(
                writer,
                frame,
                pump,
                Observation {
                    phase: query.phase,
                    query: Some(query.index),
                    status: "Success",
                    element: None,
                    mode: query.mode,
                    restart_equal: query.equal,
                    unchanged,
                    control_step: None,
                    heads: Some(&heads),
                    point: Some(&outputs),
                    operations: Some(&operations),
                    error: String::new(),
                },
            )?;
        }
    }
    Ok(errors)
}
pub fn restart(
    case: &Case,
    width: usize,
    pump: ControlledPumpBatch<crate::runtime::Runtime>,
    index: usize,
) -> Result<(
    ControlledPumpBatch<crate::runtime::Runtime>,
    &'static str,
    bool,
)> {
    if index.is_multiple_of(3) {
        return Ok((pump, "Direct", true));
    }
    let before = pump.snapshot_data()?;
    let (_, mut restored) = build(case, width)?;
    let mode = if index % 3 == 1 {
        restored.restore(&pump.snapshot())?;
        "Trusted"
    } else {
        restored.restore_data(&before)?;
        "Cold"
    };
    let equal = format!("{:?}", restored.snapshot_data()?) == format!("{before:?}");
    Ok((restored, mode, equal))
}
