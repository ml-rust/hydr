//! Structure geometry schedules and pool curve epochs in the surface/channel coordinator.
//!
//! Knot and epoch times join the coordinator event set so no interval straddles an event.
//! Structure schedules evaluate midpoint geometry and rebuild interval link operations. Pool curve
//! epochs replace the pool curves at the interval start while keeping every stored volume and
//! counter bit-identical.

mod channel;
mod pools;
mod surface;

pub(super) use channel::{
    apply_channel_sections, channel_section_epochs_at, check_channel_section_schedules,
    import_reach, next_channel_section_event, restore_reach,
};
pub(super) use pools::{
    apply_pool_curves, check_pool_curve_schedules, next_pool_curve_event, pool_curve_epochs_at,
    restored_pool_curves,
};
pub(super) use surface::{
    apply_surface_bed, check_surface_bed_schedules, next_surface_bed_event, restore_surface_data,
    surface_bed_epochs_at,
};

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

/// Receipt of one geometry schedule event (structure or pool curve) applied on an accepted
/// interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelGeometryReceipt {
    /// Schedule identity.
    pub schedule_id: String,
    /// Target structure or pool identity.
    pub element_id: String,
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

/// Prepared start-of-interval geometry overrides (surface bed, channel section, and pool curve
/// epochs).
pub(super) struct StartGeometry<R: Runtime<DType = DType>> {
    pub surface: Option<crate::solver::Solver<R>>,
    pub reaches: Option<Vec<crate::channel::ChannelReach<R>>>,
    pub pools: Option<super::pools::PoolChild<R>>,
    pub receipts: Vec<SurfaceChannelGeometryReceipt>,
}

/// Applies any surface bed, channel section, and pool curve epochs active at `start_s`.
pub(super) fn prepare_start_geometry<R: Runtime<DType = DType>>(
    context: &Context<R>,
    definition: &super::SurfaceChannelDefinition,
    surface: &crate::solver::Solver<R>,
    reaches: &[crate::channel::ChannelReach<R>],
    pools: Option<&super::pools::PoolChild<R>>,
    start_s: f64,
) -> Result<StartGeometry<R>, SurfaceChannelError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + numr::ops::CumulativeOps<R>
        + numr::ops::HypotOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::MidpointOps<R>
        + numr::ops::NextRepresentableOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TensorOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>
        + solvr::optimize::roots::BatchedBisectionAlgorithms<R>
        + solvr::optimize::roots::BatchedNewtonAlgorithms<R>,
{
    let setup = definition.setup();
    let (active_surface, mut receipts) = if setup.surface_bed_schedules.is_empty() {
        (None, Vec::new())
    } else {
        let (next, r) = apply_surface_bed(
            setup.surface.domain.bed(),
            &setup.surface_bed_schedules,
            surface,
            start_s,
        )?;
        (Some(next), r)
    };
    let (active_reaches, channel_receipts) = apply_channel_sections(setup, reaches, start_s)?;
    receipts.extend(channel_receipts);
    let (active_pools, pool_receipts) = match (pools, definition.pools.as_ref()) {
        (Some(child), Some(declared)) if !setup.pool_curve_schedules.is_empty() => {
            let (next, r) = apply_pool_curves(
                context,
                declared,
                &setup.pool_curve_schedules,
                child,
                start_s,
            )?;
            (Some(next), r)
        }
        (Some(child), _) => (Some(child.clone()), Vec::new()),
        _ => (None, Vec::new()),
    };
    receipts.extend(pool_receipts);
    Ok(StartGeometry {
        surface: active_surface,
        reaches: active_reaches,
        pools: active_pools,
        receipts,
    })
}

/// Active epoch per schedule at `time_s`.
pub(super) fn epochs_at(schedules: &[StructureGeometrySchedule], time_s: f64) -> Vec<usize> {
    active_geometry_epochs(schedules, time_s)
}

/// Checks `geometry_epochs` and `pool_curve_epochs` of `data` at `data.counters.time_s`.
pub(super) fn check_snapshot_epochs(
    data: &super::snapshot::SurfaceChannelSnapshotData,
) -> Result<(), SurfaceChannelError> {
    if data.geometry_epochs != epochs_at(&data.setup.geometry_schedules, data.counters.time_s) {
        return Err(super::frames::rejected(
            "geometry_epochs",
            "differs from the active geometry schedule epochs at counters.time_s",
        ));
    }
    if data.pool_curve_epochs
        != pool_curve_epochs_at(&data.setup.pool_curve_schedules, data.counters.time_s)
    {
        return Err(super::frames::rejected(
            "pool_curve_epochs",
            "differs from the active pool curve schedule epochs at counters.time_s",
        ));
    }
    if data.surface_bed_epochs
        != surface_bed_epochs_at(&data.setup.surface_bed_schedules, data.counters.time_s)
    {
        return Err(super::frames::rejected(
            "surface_bed_epochs",
            "differs from the active surface bed schedule epochs at counters.time_s",
        ));
    }
    if data.channel_section_epochs
        != channel_section_epochs_at(&data.setup.channel_section_schedules, data.counters.time_s)
    {
        return Err(super::frames::rejected(
            "channel_section_epochs",
            "differs from the active channel section schedule epochs at counters.time_s",
        ));
    }
    Ok(())
}

/// Interval link operations, pending receipts, and evaluated structure elements.
pub(super) type EvaluatedInterval<R> = (
    Arc<LinkOperations<R>>,
    Vec<SurfaceChannelGeometryReceipt>,
    Vec<StructureElement>,
);

/// Evaluates all link elements at the midpoint of `[start_s, end_s]` (`start_s == end_s` gives
/// the exact right limit at `start_s`).
fn evaluate_elements<R: Runtime<DType = DType>>(
    context: &Context<R>,
    schedules: &[StructureGeometrySchedule],
    start_s: f64,
    end_s: f64,
) -> Result<Vec<StructureElement>, SurfaceChannelError> {
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
    Ok(elements)
}

/// Evaluates scheduled geometry for `[start_s, end_s]` and rebuilds `LinkOperations` with updated
/// laws, cutoffs, and culvert stops, plus pending receipts (whose storage witness is filled on
/// acceptance) and the evaluated elements.
pub(super) fn evaluate_interval<R: Runtime<DType = DType>>(
    context: &Context<R>,
    schedules: &[StructureGeometrySchedule],
    points: Vec<StructureOperatingPoint>,
    start_s: f64,
    end_s: f64,
) -> Result<EvaluatedInterval<R>, SurfaceChannelError>
where
    R::Client: CulvertClient<R>,
{
    let mid = start_s + 0.5 * (end_s - start_s);
    let elements = evaluate_elements(context, schedules, start_s, end_s)?;
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
                element_id: schedule.structure_id.clone(),
                time_s: start_s,
                epoch,
                old_value_bits: old_bits,
                new_value_bits: new_bits,
                storage_before_m3: 0.0,
                storage_after_m3: 0.0,
            });
        }
    }
    Ok((Arc::new(ops), receipts, elements))
}
