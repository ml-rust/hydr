//! Caller-owned channel cross-section epochs and schedule queries.

use crate::channel::{ChannelError, ChannelGeometry, ChannelReachDefinition};
use std::collections::{BTreeMap, BTreeSet};

/// One scheduled cross-section geometry epoch for a channel reach.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSectionEpoch {
    /// Event time in seconds, finite and nonnegative.
    pub time_s: f64,
    /// New cross-section geometry active from `time_s` until the next epoch.
    pub geometry: ChannelGeometry,
    /// Caller-supplied provenance or engineering evidence for this epoch.
    pub evidence: String,
}

/// Caller-owned sequence of cross-section epochs for one declared reach.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSectionSchedule {
    /// Unique non-empty schedule identity.
    pub schedule_id: String,
    /// Identity of the target declared reach.
    pub reach_id: String,
    /// Strictly increasing epoch times.
    pub epochs: Vec<ChannelSectionEpoch>,
}

/// Checks every channel section schedule against the declared `reaches`.
///
/// Rejects empty or duplicate schedule identities, duplicate reach targets, unknown reaches,
/// empty epochs, nonfinite or negative or non-increasing times, empty evidence, invalid section
/// definitions, changed cell counts, changed cell identities or lengths, and surveyed/non-surveyed
/// variant switches.
pub fn check_channel_section_schedules(
    reaches: &[(&str, &ChannelReachDefinition)],
    schedules: &[ChannelSectionSchedule],
) -> Result<(), ChannelError> {
    let by_id: BTreeMap<&str, &ChannelReachDefinition> = reaches.iter().copied().collect();
    let mut schedule_ids = BTreeSet::new();
    let mut reach_ids = BTreeSet::new();
    for schedule in schedules {
        if schedule.schedule_id.trim().is_empty() {
            return Err(ChannelError::input(
                "channel section schedule_id must be non-empty",
            ));
        }
        if !schedule_ids.insert(schedule.schedule_id.as_str()) {
            return Err(ChannelError::input(format!(
                "duplicate channel section schedule_id={:?}: give each schedule a unique identity",
                schedule.schedule_id
            )));
        }
        let Some(&declared) = by_id.get(schedule.reach_id.as_str()) else {
            return Err(ChannelError::input(format!(
                "channel section schedule {:?}: reach_id={:?} names no declared reach",
                schedule.schedule_id, schedule.reach_id
            )));
        };
        if !reach_ids.insert(schedule.reach_id.as_str()) {
            return Err(ChannelError::input(format!(
                "channel section schedule {:?}: reach_id={:?} already has a section schedule",
                schedule.schedule_id, schedule.reach_id
            )));
        }
        if schedule.epochs.is_empty() {
            return Err(ChannelError::input(format!(
                "channel section schedule {:?}: supply at least one section epoch",
                schedule.schedule_id
            )));
        }
        let declared_surveyed =
            matches!(declared.geometry, ChannelGeometry::PrismaticSurveyed { .. });
        let mut previous_time: Option<f64> = None;
        for (index, epoch) in schedule.epochs.iter().enumerate() {
            if !epoch.time_s.is_finite() || epoch.time_s < 0.0 {
                return Err(ChannelError::input(format!(
                    "channel section schedule {:?} epoch {index}: time_s={} must be finite and nonnegative",
                    schedule.schedule_id, epoch.time_s
                )));
            }
            if let Some(prev) = previous_time
                && epoch.time_s <= prev
            {
                return Err(ChannelError::input(format!(
                    "channel section schedule {:?} epoch {index}: time_s={} must strictly exceed previous time_s={prev}",
                    schedule.schedule_id, epoch.time_s
                )));
            }
            if epoch.evidence.trim().is_empty() {
                return Err(ChannelError::input(format!(
                    "channel section schedule {:?} epoch {index}: supply non-empty evidence",
                    schedule.schedule_id
                )));
            }
            let epoch_surveyed =
                matches!(epoch.geometry, ChannelGeometry::PrismaticSurveyed { .. });
            if epoch_surveyed != declared_surveyed {
                return Err(ChannelError::input(format!(
                    "channel section schedule {:?} epoch {index}: cannot switch between surveyed and analytic channel geometry",
                    schedule.schedule_id
                )));
            }
            let mut probe = declared.clone();
            probe.geometry = epoch.geometry.clone();
            probe.check(true).map_err(|error| {
                ChannelError::input(format!(
                    "channel section schedule {:?} epoch {index}: {error}",
                    schedule.schedule_id
                ))
            })?;
            if epoch.geometry.len() != declared.geometry.len() {
                return Err(ChannelError::input(format!(
                    "channel section schedule {:?} epoch {index}: cell count={} differs from declared {}",
                    schedule.schedule_id,
                    epoch.geometry.len(),
                    declared.geometry.len()
                )));
            }
            for (cell, (a, b)) in declared
                .geometry
                .cells()
                .zip(epoch.geometry.cells())
                .enumerate()
            {
                if a.id != b.id || a.length_m.to_bits() != b.length_m.to_bits() {
                    return Err(ChannelError::input(format!(
                        "channel section schedule {:?} epoch {index} cell {cell}: id/length_m ({:?}, {}) differs from declared ({:?}, {})",
                        schedule.schedule_id, b.id, b.length_m, a.id, a.length_m
                    )));
                }
            }
            previous_time = Some(epoch.time_s);
        }
    }
    Ok(())
}

/// Earliest channel section epoch time strictly greater than `time_s`, if any.
pub fn next_channel_section_event_after(
    schedules: &[ChannelSectionSchedule],
    time_s: f64,
) -> Option<f64> {
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
pub fn active_channel_section_epochs(
    schedules: &[ChannelSectionSchedule],
    time_s: f64,
) -> Vec<usize> {
    schedules
        .iter()
        .map(|schedule| {
            schedule
                .epochs
                .partition_point(|epoch| epoch.time_s <= time_s)
        })
        .collect()
}

/// Active geometry of one reach at `time_s`: the latest epoch with `time_s <= time_s`, else
/// `declared`.
pub fn active_geometry_at<'a>(
    declared: &'a ChannelGeometry,
    schedule: &'a ChannelSectionSchedule,
    time_s: f64,
) -> &'a ChannelGeometry {
    let count = schedule
        .epochs
        .partition_point(|epoch| epoch.time_s <= time_s);
    if count == 0 {
        declared
    } else {
        &schedule.epochs[count - 1].geometry
    }
}

/// 64-bit FNV-1a digest over the exact Debug representation of `geometry`.
pub fn geometry_digest_bits(geometry: &ChannelGeometry) -> u64 {
    format!("{geometry:?}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
}
