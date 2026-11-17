//! Exact static-group event aggregation from current fresh source frames.

use super::{
    OperatingSourceBatch, OperatingSourceEvent, OperatingSourceEventKind,
    OperatingSourceEventOrigin, OperatingSourceFrame,
};
use crate::structures::StructureError;
use crate::structures::tensor::operating_source::{TensorOperatingSourceKind, frames};
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> OperatingSourceBatch<R> {
    /// Return the next strictly later common-supported event using only scalar actuator event reductions.
    ///
    /// Fresh frames must exist at this query. The result predicts no complete future actuator sequence.
    pub fn next_event_after(
        &self,
        time_s: f64,
        source_frames: &[OperatingSourceFrame<'_, R>],
    ) -> Result<Option<OperatingSourceEvent>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>,
    {
        self.check_time(time_s)?;
        let frames = frames(&self.context, time_s, source_frames)?;
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
                TensorOperatingSourceKind::Manual => None,
                TensorOperatingSourceKind::Prescribed(source) => {
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
                TensorOperatingSourceKind::Actuated(_) => {
                    let Some(OperatingSourceFrame::ActuatedGate { actuator, .. }) =
                        next_frame.next()
                    else {
                        return Err(StructureError::numerical(
                            "checked actuator event frame is incomplete or has another kind",
                        ));
                    };
                    actuator.next_event()?.map(|event| {
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
                TensorOperatingSourceKind::ControlledPump(_) => {
                    let Some(OperatingSourceFrame::ControlledPump { pump, .. }) = next_frame.next()
                    else {
                        return Err(StructureError::numerical(
                            "checked pump event frame is incomplete or has another kind",
                        ));
                    };
                    pump.next_event().map(|event| {
                        (
                            event.time_s,
                            OperatingSourceEventKind::PumpControl {
                                decision_index: event.decision_index,
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
