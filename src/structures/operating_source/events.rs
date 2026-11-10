//! Exact static-group event aggregation from current fresh source frames.

use super::{
    ActuatorSourceFrame, OperatingSourceBatch, OperatingSourceEvent, OperatingSourceEventKind,
    OperatingSourceEventOrigin,
};
use crate::cpu::structures::operating_source::{CpuOperatingSourceKind, frames};
use crate::structures::StructureError;

impl OperatingSourceBatch {
    /// Return the next strictly later common-supported event using only scalar actuator event reductions.
    ///
    /// Fresh frames must exist at this query. The result predicts no complete future actuator sequence.
    pub fn next_event_after(
        &self,
        time_s: f64,
        actuator_frames: &[ActuatorSourceFrame<'_>],
    ) -> Result<Option<OperatingSourceEvent>, StructureError> {
        self.check_time(time_s)?;
        let frames = frames(&self.context, time_s, actuator_frames)?;
        let Some(support) = self.support() else {
            return Ok(None);
        };
        if time_s == support.end_s {
            return Ok(None);
        }
        let mut next_time = support.end_s;
        let mut origins = Vec::new();
        let mut next_frame = frames.into_iter();
        for group in &self.context.groups {
            let candidate = match &group.kind {
                CpuOperatingSourceKind::Manual => None,
                CpuOperatingSourceKind::Prescribed(source) => {
                    if time_s >= source.support.end_s {
                        None
                    } else {
                        let index = source.event_times.partition_point(|event| *event <= time_s);
                        source
                            .event_times
                            .get(index)
                            .map(|&time_s| (time_s, OperatingSourceEventKind::PrescribedKnot))
                    }
                }
                CpuOperatingSourceKind::Actuated(_) => {
                    let frame = next_frame.next().ok_or_else(|| {
                        StructureError::numerical(
                            "checked operating-source frame vector is incomplete",
                        )
                    })?;
                    frame.next_event()?.map(|event| {
                        (
                            event.time_s,
                            OperatingSourceEventKind::Actuator {
                                target_index: event.target_index,
                                arrival_lower_boundary: event.arrival_lower_boundary,
                                arrival_upper_boundary: event.arrival_upper_boundary,
                                terminal: event.terminal,
                            },
                        )
                    })
                }
            };
            if let Some((candidate, kind)) = candidate {
                if candidate <= time_s || candidate > support.end_s {
                    continue;
                }
                if candidate < next_time {
                    next_time = candidate;
                    origins.clear();
                }
                if candidate == next_time {
                    origins.push(OperatingSourceEventOrigin {
                        source_id: group.id.clone(),
                        kind,
                    });
                }
            }
        }
        Ok(Some(OperatingSourceEvent {
            time_s: next_time,
            terminal: next_time == support.end_s,
            origins,
        }))
    }
}
