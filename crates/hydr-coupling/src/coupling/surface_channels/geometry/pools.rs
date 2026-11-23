//! Pool stage-volume curve epochs in the surface/channel coordinator.
//!
//! Epoch times join the coordinator event set so no interval straddles a curve change. At the
//! start of each interval whose start time reaches a new epoch, the pool child switches its
//! curves while keeping every stored volume and cumulative counter bit-identical. Volume above
//! the new top knot rejects without retry.

use super::SurfaceChannelGeometryReceipt;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::Context;
use crate::coupling::surface_channels::pool_ends::CheckedPoolEnd;
use crate::coupling::surface_channels::pools::{
    CheckedPools, PoolChild, PoolCurveState, pool_error,
};
use crate::coupling::surface_channels::structures::{CheckedNode, CheckedStructure};
use crate::storage::{
    LevelPoolCurve, PoolCurveSchedule, active_curve_at, active_pool_curve_epochs,
    check_pool_curve_schedules as check_storage_schedules, curve_digest_bits,
    next_pool_curve_event_after,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use std::sync::Arc;

/// Checks `schedules` against declared `pools`, structure attachments, and pool ends.
///
/// Rejects schedules without pools, invalid schedule definitions, and any epoch whose new curve
/// origin rises above an attached structure bed or pool-end cell bed.
pub(in super::super) fn check_pool_curve_schedules(
    pools: Option<&CheckedPools>,
    structures: &[CheckedStructure],
    ends: &[CheckedPoolEnd],
    schedules: &[PoolCurveSchedule],
) -> Result<(), SurfaceChannelError> {
    if schedules.is_empty() {
        return Ok(());
    }
    let Some(pools) = pools else {
        return Err(SurfaceChannelError::input(
            "pool_curve_schedules: declared schedules without coordinator pools",
        ));
    };
    let elements = pools.elements();
    check_storage_schedules(&elements, schedules)
        .map_err(|error| SurfaceChannelError::input(format!("pool_curve_schedules: {error}")))?;
    for schedule in schedules {
        let Some(pool_index) = pools.ids.iter().position(|id| id == &schedule.pool_id) else {
            return Err(SurfaceChannelError::input(format!(
                "pool_curve_schedules {:?}: pool_id={:?} names no declared pool",
                schedule.schedule_id, schedule.pool_id
            )));
        };
        for (epoch_index, epoch) in schedule.epochs.iter().enumerate() {
            let origin = epoch.curve.origin();
            for structure in structures {
                for (side, node) in structure.nodes.iter().enumerate() {
                    if *node == CheckedNode::Pool(pool_index) && structure.beds[side] < origin {
                        return Err(SurfaceChannelError::input(format!(
                            "unobstructed-pool-connection: pool_curve_schedules {:?} epoch {epoch_index} at time_s={}: pool={:?} new curve origin={origin} exceeds structure={:?} attachment bed={}",
                            schedule.schedule_id,
                            epoch.time_s,
                            schedule.pool_id,
                            structure.id(),
                            structure.beds[side]
                        )));
                    }
                }
            }
            for end in ends {
                if end.pool == pool_index && end.bed_m < origin {
                    return Err(SurfaceChannelError::input(format!(
                        "unobstructed-pool-connection: pool_curve_schedules {:?} epoch {epoch_index} at time_s={}: pool={:?} new curve origin={origin} exceeds pool end={:?} bed={}",
                        schedule.schedule_id, epoch.time_s, schedule.pool_id, end.id, end.bed_m
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Earliest pool curve epoch time strictly after `start_s`, or `None`.
pub(in super::super) fn next_pool_curve_event(
    schedules: &[PoolCurveSchedule],
    start_s: f64,
) -> Option<f64> {
    next_pool_curve_event_after(schedules, start_s)
}

/// Curve-dependent pool state for a restored pool batch `batch`.
pub(in super::super) fn restored_pool_curves<R: Runtime<DType = DType>>(
    context: &Context<R>,
    batch: &crate::storage::LevelPoolBatch<R>,
    has_schedules: bool,
) -> Result<Arc<PoolCurveState<R>>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let topology = context.pool_topology()?;
    if !has_schedules {
        return Ok(Arc::clone(&topology.curves));
    }
    let offset = context.surface_count + context.slot_count;
    Ok(Arc::new(PoolCurveState::from_batch(
        &context.client,
        topology,
        offset,
        batch,
    )?))
}

/// Active pool curve epoch per schedule at `time_s`.
pub(in super::super) fn pool_curve_epochs_at(
    schedules: &[PoolCurveSchedule],
    time_s: f64,
) -> Vec<usize> {
    active_pool_curve_epochs(schedules, time_s)
}

/// Applies any pool curve epochs active at `start_s` to `child`, returning the updated pool child
/// (with bit-identical volumes and counters) and receipts for every schedule whose epoch
/// transitions at `start_s`.
///
/// Rejects with `pool-capacity-exceeded` when a pool's current volume exceeds its new curve's top
/// knot.
pub(in super::super) fn apply_pool_curves<R: Runtime<DType = DType>>(
    context: &Context<R>,
    declared: &CheckedPools,
    schedules: &[PoolCurveSchedule],
    child: &PoolChild<R>,
    start_s: f64,
) -> Result<(PoolChild<R>, Vec<SurfaceChannelGeometryReceipt>), SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    if schedules.is_empty() {
        return Ok((child.clone(), Vec::new()));
    }
    let topology = context.pool_topology()?;
    let declared_elements = declared.elements();
    let current_elements = child.batch.elements();
    let epochs = active_pool_curve_epochs(schedules, start_s);
    let mut target_curves: Vec<LevelPoolCurve> = Vec::with_capacity(current_elements.len());
    let mut changed = false;
    let mut receipts = Vec::new();
    for (pool_index, (decl, curr)) in declared_elements.iter().zip(current_elements).enumerate() {
        let scheduled = schedules
            .iter()
            .zip(&epochs)
            .find(|(s, _)| s.pool_id == decl.id);
        let Some((schedule, &epoch)) = scheduled else {
            target_curves.push(curr.curve.clone());
            continue;
        };
        let active = active_curve_at(&decl.curve, schedule, start_s);
        let old_bits = curve_digest_bits(&curr.curve);
        let new_bits = curve_digest_bits(active);
        let at_epoch = schedule
            .epochs
            .iter()
            .any(|e| e.time_s.to_bits() == start_s.to_bits());
        if old_bits != new_bits {
            changed = true;
            // Re-check current volume against new top knot for an explicit capacity error.
            let volumes = child.batch.volumes_tensor().try_to_vec::<f64>()?;
            let vol = volumes.get(pool_index).copied().unwrap_or(0.0);
            let top = active.maximum_volume_m3();
            if vol > top {
                return Err(SurfaceChannelError::input(format!(
                    "pool-capacity-exceeded: pool_curve_schedules {:?} at time_s={start_s}: pool={:?} volume={vol} exceeds new top curve knot volume={top}: lower pool volume before the epoch or extend the curve",
                    schedule.schedule_id, schedule.pool_id
                )));
            }
        }
        if at_epoch || old_bits != new_bits {
            receipts.push(SurfaceChannelGeometryReceipt {
                schedule_id: schedule.schedule_id.clone(),
                element_id: schedule.pool_id.clone(),
                time_s: start_s,
                epoch,
                old_value_bits: old_bits,
                new_value_bits: new_bits,
                storage_before_m3: 0.0,
                storage_after_m3: 0.0,
            });
        }
        target_curves.push(active.clone());
    }
    if !changed {
        return Ok((child.clone(), receipts));
    }
    let next_batch =
        (topology.fns.with_curves)(&child.batch, &target_curves).map_err(pool_error)?;
    let offset = context.surface_count + context.slot_count;
    let curves = Arc::new(PoolCurveState::from_batch(
        &context.client,
        topology,
        offset,
        &next_batch,
    )?);
    Ok((
        PoolChild {
            batch: next_batch,
            external: child.external.clone(),
            curves,
        },
        receipts,
    ))
}
