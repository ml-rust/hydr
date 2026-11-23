//! Event-time queries and epoch derivation for structure geometry schedules.

use super::StructureGeometrySchedule;

/// Earliest knot time strictly after `start_s` across `schedules`, or `None` when none remains.
pub fn next_geometry_event_after(
    schedules: &[StructureGeometrySchedule],
    start_s: f64,
) -> Option<f64> {
    schedules
        .iter()
        .filter_map(|schedule| {
            schedule
                .track
                .knot_times()
                .into_iter()
                .find(|&time| time > start_s)
        })
        .reduce(f64::min)
}

/// Active epoch index of every schedule at `time_s`, in schedule order.
///
/// The epoch is the number of knots with `knot.time_s <= time_s`. It is purely a function of
/// `time_s`, so checkpoint import recomputes it and rejects any mismatch.
pub fn active_geometry_epochs(schedules: &[StructureGeometrySchedule], time_s: f64) -> Vec<usize> {
    schedules
        .iter()
        .map(|schedule| {
            let times = schedule.track.knot_times();
            times.partition_point(|&t| t <= time_s)
        })
        .collect()
}
