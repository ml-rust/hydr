//! Structure geometry schedules in the surface/channel coordinator.
//!
//! Knot times join the coordinator event set so no interval straddles a knot. Each interval
//! evaluates its midpoint geometry on every scheduled structure, rebuilds the interval's link
//! operations (re-certifying floored culvert rows if sediment depth changed), and records a
//! receipt with bit-equal storage-before and storage-after volume witnesses.

use super::SurfaceChannelError;
use super::context::Context;
use super::structures::{LinkOperations, SurfaceChannelStructure};
use crate::structures::{
    CulvertClient, StructureElement, StructureGeometrySchedule, StructureOperatingPoint,
    active_geometry_epochs, check_schedules, evaluate_element_interval, next_geometry_event_after,
    track_value_bits,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Receipt of one structure geometry schedule applied on an accepted interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelGeometryReceipt {
    /// Schedule identity.
    pub schedule_id: String,
    /// Target structure identity.
    pub structure_id: String,
    /// Interval start time (knot time when starting at an event), in seconds.
    pub time_s: f64,
    /// Active epoch index of this schedule at `time_s`.
    pub epoch: usize,
    /// Parameter value bits immediately before `time_s`.
    pub old_value_bits: u64,
    /// Parameter value bits applied on this interval's midpoint.
    pub new_value_bits: u64,
    /// Total system storage immediately before applying the geometry event, in m³.
    pub storage_before_m3: f64,
    /// Total system storage immediately after applying the geometry event, in m³ (bit-equal).
    pub storage_after_m3: f64,
}

/// Checks `schedules` against `structures`, rejecting unknown targets, mismatched laws, duplicate
/// parameter writers, invalid knot values, and any knot whose resulting control elevation falls
/// below its attachment beds.
pub(super) fn check_geometry_schedules(
    structures: &[SurfaceChannelStructure],
    schedules: &[StructureGeometrySchedule],
    beds: &[[f64; 2]],
    gravity: f64,
) -> Result<(), SurfaceChannelError> {
    if schedules.is_empty() {
        return Ok(());
    }
    let elements: Vec<StructureElement> = structures.iter().map(|s| s.element.clone()).collect();
    check_schedules(&elements, schedules)
        .map_err(|error| SurfaceChannelError::input(format!("geometry_schedules: {error}")))?;
    // Verify that at every knot's left and right limits, each scheduled structure still satisfies
    // attachment-bed and gravity checks.
    let mut times: Vec<f64> = schedules
        .iter()
        .flat_map(|s| s.track.knot_times())
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| a.to_bits() == b.to_bits());
    for (structure, &link_beds) in structures.iter().zip(beds) {
        let mine: Vec<&StructureGeometrySchedule> = schedules
            .iter()
            .filter(|s| s.structure_id == structure.element.id)
            .collect();
        if mine.is_empty() {
            continue;
        }
        for &t in &times {
            for dt in [-1e-9, 1e-9] {
                let (start, end) = if dt < 0.0 { (t + dt, t) } else { (t, t + dt) };
                let evaluated = evaluate_element_interval(&structure.element, &mine, start, end)
                    .map_err(|error| {
                        SurfaceChannelError::input(format!("geometry_schedules: {error}"))
                    })?;
                let mut probe = structure.clone();
                probe.element = evaluated;
                super::structures::check_law_for_geometry(&probe, link_beds, gravity)?;
            }
        }
    }
    Ok(())
}

/// Earliest geometry knot time strictly after `start_s`, or `None`.
pub(super) fn next_event(schedules: &[StructureGeometrySchedule], start_s: f64) -> Option<f64> {
    next_geometry_event_after(schedules, start_s)
}

/// Active epoch per schedule at `time_s`.
pub(super) fn epochs_at(schedules: &[StructureGeometrySchedule], time_s: f64) -> Vec<usize> {
    active_geometry_epochs(schedules, time_s)
}

/// Evaluates scheduled geometry for `[start_s, end_s]` and rebuilds `LinkOperations` with updated
/// laws, cutoffs, and culvert stops, plus pending receipts (whose storage witness is filled on
/// acceptance).
pub(super) fn evaluate_interval<R: Runtime<DType = DType>>(
    context: &Context<R>,
    schedules: &[StructureGeometrySchedule],
    points: Vec<StructureOperatingPoint>,
    start_s: f64,
    end_s: f64,
) -> Result<(Arc<LinkOperations<R>>, Vec<SurfaceChannelGeometryReceipt>), SurfaceChannelError>
where
    R::Client: CulvertClient<R>,
{
    let mid = start_s + 0.5 * (end_s - start_s);
    let mut elements = Vec::with_capacity(context.link_elements.len());
    for element in &context.link_elements {
        let mine: Vec<&StructureGeometrySchedule> = schedules
            .iter()
            .filter(|s| s.structure_id == element.id)
            .collect();
        if mine.is_empty() {
            elements.push(element.clone());
        } else {
            let evaluated =
                evaluate_element_interval(element, &mine, start_s, end_s).map_err(|error| {
                    SurfaceChannelError::input(format!(
                        "geometry_schedules at [{start_s}, {end_s}]: {error}"
                    ))
                })?;
            elements.push(evaluated);
        }
    }
    let beds: Vec<[f64; 2]> = context.operations.laws.iter().map(|law| law.bed).collect();
    let allowance = (context.head_absolute_m, context.head_relative);
    let mut ops = LinkOperations::declared(&context.client, &elements, points, &beds, allowance)?;
    let stops = context.build_culvert_stops(&ops.laws)?;
    ops.culverts = stops;
    let epochs = active_geometry_epochs(schedules, start_s);
    let mut receipts = Vec::new();
    for (schedule, epoch) in schedules.iter().zip(epochs) {
        let old_bits = track_value_bits(&schedule.track, start_s, false);
        let new_bits = track_value_bits(&schedule.track, mid, true);
        let at_knot = schedule
            .track
            .knot_times()
            .iter()
            .any(|&t| t.to_bits() == start_s.to_bits());
        if at_knot || old_bits != new_bits {
            receipts.push(SurfaceChannelGeometryReceipt {
                schedule_id: schedule.schedule_id.clone(),
                structure_id: schedule.structure_id.clone(),
                time_s: start_s,
                epoch,
                old_value_bits: old_bits,
                new_value_bits: new_bits,
                storage_before_m3: 0.0,
                storage_after_m3: 0.0,
            });
        }
    }
    Ok((Arc::new(ops), receipts))
}
