//! Exact original support, knot events, and temporal segment metadata.

use super::GateTrajectoryAssignment;
use crate::structures::{StructureError, StructureInputErrorKind};

/// Closed common time support in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateTrajectorySupport {
    /// Inclusive first supported time.
    pub start_s: f64,
    /// Inclusive last supported time.
    pub end_s: f64,
}
/// One original knot event, without epsilon merging.
#[derive(Clone, Debug, PartialEq)]
pub struct GateTrajectoryEvent {
    /// Original event time in seconds.
    pub time_s: f64,
    /// Original-order gates with a knot at this time.
    pub structure_ids: Vec<String>,
}
/// Checked original interval and exact internal/terminal events.
#[derive(Clone, Debug, PartialEq)]
pub struct GateTrajectoryInterval {
    /// Original interval start in seconds.
    pub start_s: f64,
    /// Original interval end in seconds.
    pub end_s: f64,
    /// Events strictly inside the interval.
    pub interior_events: Vec<GateTrajectoryEvent>,
    /// Event exactly at the interval end.
    pub end_event: Option<GateTrajectoryEvent>,
}
pub(super) fn build(
    assignments: &[GateTrajectoryAssignment],
    allow_point: bool,
) -> Result<(GateTrajectorySupport, Vec<GateTrajectoryEvent>), StructureError> {
    let mut start_s: f64 = 0.0;
    let mut end_s = f64::MAX;
    for assignment in assignments {
        let knots = assignment.trajectory.knots();
        let first = knots
            .first()
            .ok_or_else(|| StructureError::numerical("checked trajectory lacks first knot"))?;
        let last = knots
            .last()
            .ok_or_else(|| StructureError::numerical("checked trajectory lacks last knot"))?;
        start_s = start_s.max(first.time_s);
        end_s = end_s.min(last.time_s);
    }
    if end_s < start_s || (!allow_point && end_s == start_s) {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            if allow_point {
                format!(
                    "common support=[{start_s},{end_s}]: supply a nonempty closed trajectory intersection"
                )
            } else {
                format!(
                    "common support=[{start_s},{end_s}]: supply positive overlapping trajectory duration"
                )
            },
        ));
    }
    let mut events = std::collections::BTreeMap::<u64, GateTrajectoryEvent>::new();
    for assignment in assignments {
        for knot in assignment.trajectory.knots() {
            if knot.time_s >= start_s && knot.time_s <= end_s {
                let key = if knot.time_s == 0.0 {
                    0
                } else {
                    knot.time_s.to_bits()
                };
                events
                    .entry(key)
                    .or_insert_with(|| GateTrajectoryEvent {
                        time_s: knot.time_s,
                        structure_ids: Vec::new(),
                    })
                    .structure_ids
                    .push(assignment.structure_id.clone());
            }
        }
    }
    let events = events.into_values().collect();
    Ok((GateTrajectorySupport { start_s, end_s }, events))
}
pub(super) fn check_query(
    support: GateTrajectorySupport,
    time_s: f64,
) -> Result<(), StructureError> {
    if !time_s.is_finite() {
        return Err(StructureError::input(
            StructureInputErrorKind::NonfiniteInput,
            format!("trajectory query={time_s}: supply finite original time"),
        ));
    }
    if time_s < support.start_s || time_s > support.end_s {
        return Err(StructureError::input(
            StructureInputErrorKind::UnsupportedRegime,
            format!(
                "trajectory query={time_s},support={support:?}: query within original closed support"
            ),
        ));
    }
    Ok(())
}
