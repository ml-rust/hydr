//! Caller-owned pool stage-volume curve schedules and epoch queries.

use crate::storage::{LevelPoolCurve, LevelPoolElement, LevelPoolError};
use std::collections::{BTreeMap, BTreeSet};

/// One scheduled stage-volume curve epoch for a pool.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolCurveEpoch {
    /// Event time in seconds, finite and nonnegative.
    pub time_s: f64,
    /// Stage-volume curve active from `time_s` until the next epoch.
    pub curve: LevelPoolCurve,
    /// Caller-supplied provenance or engineering evidence for this epoch.
    pub evidence: String,
}

/// Caller-owned sequence of stage-volume curve epochs for one declared pool.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolCurveSchedule {
    /// Unique non-empty schedule identity.
    pub schedule_id: String,
    /// Identity of the target declared pool.
    pub pool_id: String,
    /// Strictly increasing epoch times.
    pub epochs: Vec<PoolCurveEpoch>,
}

/// Checks every schedule against the declared `elements`.
///
/// Rejects empty or duplicate schedule identities, duplicate pool targets, unknown pools,
/// empty epoch lists, nonfinite or negative or non-increasing epoch times, empty evidence,
/// and epoch curves whose vertical reference differs from the target pool's curve.
pub fn check_pool_curve_schedules(
    elements: &[LevelPoolElement],
    schedules: &[PoolCurveSchedule],
) -> Result<(), LevelPoolError> {
    let by_id: BTreeMap<&str, &LevelPoolElement> =
        elements.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut schedule_ids = BTreeSet::new();
    let mut pool_ids = BTreeSet::new();
    for schedule in schedules {
        if schedule.schedule_id.trim().is_empty() {
            return Err(LevelPoolError::input(
                "pool curve schedule_id must be non-empty",
            ));
        }
        if !schedule_ids.insert(schedule.schedule_id.as_str()) {
            return Err(LevelPoolError::input(format!(
                "duplicate pool curve schedule_id={:?}: give each schedule a unique identity",
                schedule.schedule_id
            )));
        }
        let Some(element) = by_id.get(schedule.pool_id.as_str()) else {
            return Err(LevelPoolError::input(format!(
                "pool curve schedule {:?}: pool_id={:?} names no declared pool",
                schedule.schedule_id, schedule.pool_id
            )));
        };
        if !pool_ids.insert(schedule.pool_id.as_str()) {
            return Err(LevelPoolError::input(format!(
                "pool curve schedule {:?}: pool_id={:?} already has a curve schedule",
                schedule.schedule_id, schedule.pool_id
            )));
        }
        if schedule.epochs.is_empty() {
            return Err(LevelPoolError::input(format!(
                "pool curve schedule {:?}: supply at least one curve epoch",
                schedule.schedule_id
            )));
        }
        let expected_ref = element.curve.vertical_reference();
        let mut previous_time: Option<f64> = None;
        for (index, epoch) in schedule.epochs.iter().enumerate() {
            if !epoch.time_s.is_finite() || epoch.time_s < 0.0 {
                return Err(LevelPoolError::input(format!(
                    "pool curve schedule {:?} epoch {index}: time_s={} must be finite and nonnegative",
                    schedule.schedule_id, epoch.time_s
                )));
            }
            if let Some(prev) = previous_time
                && epoch.time_s <= prev
            {
                return Err(LevelPoolError::input(format!(
                    "pool curve schedule {:?} epoch {index}: time_s={} must strictly exceed previous time_s={prev}",
                    schedule.schedule_id, epoch.time_s
                )));
            }
            if epoch.evidence.trim().is_empty() {
                return Err(LevelPoolError::input(format!(
                    "pool curve schedule {:?} epoch {index}: supply non-empty evidence",
                    schedule.schedule_id
                )));
            }
            if epoch.curve.vertical_reference() != expected_ref {
                return Err(LevelPoolError::input(format!(
                    "pool curve schedule {:?} epoch {index}: curve vertical_reference={:?} differs from pool vertical_reference={expected_ref:?}",
                    schedule.schedule_id,
                    epoch.curve.vertical_reference()
                )));
            }
            previous_time = Some(epoch.time_s);
        }
    }
    Ok(())
}

/// Earliest pool curve epoch time strictly greater than `time_s`, if any.
pub fn next_pool_curve_event_after(schedules: &[PoolCurveSchedule], time_s: f64) -> Option<f64> {
    schedules
        .iter()
        .filter_map(|schedule| {
            let index = schedule
                .epochs
                .partition_point(|epoch| epoch.time_s <= time_s);
            schedule.epochs.get(index).map(|epoch| epoch.time_s)
        })
        .reduce(f64::min)
}

/// Active epoch count (`0..=epochs.len()`) per schedule at `time_s`.
///
/// Zero means no epoch has activated yet; `k > 0` means `epochs[k - 1]` is active.
pub fn active_pool_curve_epochs(schedules: &[PoolCurveSchedule], time_s: f64) -> Vec<usize> {
    schedules
        .iter()
        .map(|schedule| {
            schedule
                .epochs
                .partition_point(|epoch| epoch.time_s <= time_s)
        })
        .collect()
}

/// Active curve of one pool at `time_s`: the latest epoch with `time_s <= time_s`, else `declared`.
pub fn active_curve_at<'a>(
    declared: &'a LevelPoolCurve,
    schedule: &'a PoolCurveSchedule,
    time_s: f64,
) -> &'a LevelPoolCurve {
    let count = schedule
        .epochs
        .partition_point(|epoch| epoch.time_s <= time_s);
    if count == 0 {
        declared
    } else {
        &schedule.epochs[count - 1].curve
    }
}

/// 64-bit FNV-1a digest over the exact `(volume_m3, stage_m)` bit patterns of `curve`.
pub fn curve_digest_bits(curve: &LevelPoolCurve) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for knot in curve.knots() {
        for word in [knot.volume_m3.to_bits(), knot.stage_m.to_bits()] {
            for byte in word.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    hash
}
