//! Caller-owned surface bed elevation epochs and schedule queries.

use crate::contract::EngineError;
use crate::model::Domain;
use std::collections::BTreeSet;

/// One cell bed elevation update inside a [`SurfaceBedEpoch`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceBedUpdate {
    /// Active row-major cell index.
    pub cell: usize,
    /// New finite bed elevation in metres on the domain's vertical datum.
    pub bed_elevation_m: f64,
}

/// One scheduled surface bed elevation step epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceBedEpoch {
    /// Event time in seconds, finite and nonnegative.
    pub time_s: f64,
    /// Non-empty list of unique active cell bed updates applied at `time_s`.
    pub updates: Vec<SurfaceBedUpdate>,
    /// Caller-supplied provenance or engineering evidence for this epoch.
    pub evidence: String,
}

/// Caller-owned sequence of surface bed elevation epochs.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceBedSchedule {
    /// Unique non-empty schedule identity.
    pub schedule_id: String,
    /// Strictly increasing epoch times.
    pub epochs: Vec<SurfaceBedEpoch>,
}

/// Checks every surface bed schedule against `domain`.
///
/// Rejects empty or duplicate schedule identities, empty epochs or updates, nonfinite or negative
/// or non-increasing times, empty evidence, out-of-range or inactive cells, nonfinite bed
/// elevations, duplicate cells within an epoch, and cells targeted by more than one schedule.
pub fn check_surface_bed_schedules(
    domain: &Domain,
    schedules: &[SurfaceBedSchedule],
) -> Result<(), EngineError> {
    let cell_count = domain.grid().cell_count();
    let active = domain.active();
    let mut schedule_ids = BTreeSet::new();
    let mut owned_cells = BTreeSet::new();
    for schedule in schedules {
        if schedule.schedule_id.trim().is_empty() {
            return Err(EngineError::rejected(
                "surface_bed_schedules",
                "schedule_id must be non-empty",
            ));
        }
        if !schedule_ids.insert(schedule.schedule_id.as_str()) {
            return Err(EngineError::rejected(
                "surface_bed_schedules",
                format!(
                    "duplicate schedule_id={:?}: give each schedule a unique identity",
                    schedule.schedule_id
                ),
            ));
        }
        if schedule.epochs.is_empty() {
            return Err(EngineError::rejected(
                "surface_bed_schedules",
                format!(
                    "schedule {:?}: supply at least one bed epoch",
                    schedule.schedule_id
                ),
            ));
        }
        let mut schedule_cells = BTreeSet::new();
        let mut previous_time: Option<f64> = None;
        for (index, epoch) in schedule.epochs.iter().enumerate() {
            if !epoch.time_s.is_finite() || epoch.time_s < 0.0 {
                return Err(EngineError::rejected(
                    "surface_bed_schedules",
                    format!(
                        "schedule {:?} epoch {index}: time_s={} must be finite and nonnegative",
                        schedule.schedule_id, epoch.time_s
                    ),
                ));
            }
            if let Some(prev) = previous_time
                && epoch.time_s <= prev
            {
                return Err(EngineError::rejected(
                    "surface_bed_schedules",
                    format!(
                        "schedule {:?} epoch {index}: time_s={} must strictly exceed previous time_s={prev}",
                        schedule.schedule_id, epoch.time_s
                    ),
                ));
            }
            if epoch.evidence.trim().is_empty() {
                return Err(EngineError::rejected(
                    "surface_bed_schedules",
                    format!(
                        "schedule {:?} epoch {index}: supply non-empty evidence",
                        schedule.schedule_id
                    ),
                ));
            }
            if epoch.updates.is_empty() {
                return Err(EngineError::rejected(
                    "surface_bed_schedules",
                    format!(
                        "schedule {:?} epoch {index}: supply at least one cell bed update",
                        schedule.schedule_id
                    ),
                ));
            }
            let mut epoch_cells = BTreeSet::new();
            for update in &epoch.updates {
                if update.cell >= cell_count || !active[update.cell] {
                    return Err(EngineError::rejected(
                        "surface_bed_schedules",
                        format!(
                            "schedule {:?} epoch {index}: cell={} is out of range or inactive",
                            schedule.schedule_id, update.cell
                        ),
                    ));
                }
                if !update.bed_elevation_m.is_finite() {
                    return Err(EngineError::rejected(
                        "surface_bed_schedules",
                        format!(
                            "schedule {:?} epoch {index} cell={}: bed_elevation_m={} must be finite",
                            schedule.schedule_id, update.cell, update.bed_elevation_m
                        ),
                    ));
                }
                if !epoch_cells.insert(update.cell) {
                    return Err(EngineError::rejected(
                        "surface_bed_schedules",
                        format!(
                            "schedule {:?} epoch {index}: duplicate update for cell={}",
                            schedule.schedule_id, update.cell
                        ),
                    ));
                }
                schedule_cells.insert(update.cell);
            }
            previous_time = Some(epoch.time_s);
        }
        for cell in schedule_cells {
            if !owned_cells.insert(cell) {
                return Err(EngineError::rejected(
                    "surface_bed_schedules",
                    format!(
                        "schedule {:?}: cell={cell} is already targeted by another bed schedule",
                        schedule.schedule_id
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Earliest surface bed epoch time strictly greater than `time_s`, if any.
pub fn next_surface_bed_event_after(schedules: &[SurfaceBedSchedule], time_s: f64) -> Option<f64> {
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
pub fn active_surface_bed_epochs(schedules: &[SurfaceBedSchedule], time_s: f64) -> Vec<usize> {
    schedules
        .iter()
        .map(|schedule| {
            schedule
                .epochs
                .partition_point(|epoch| epoch.time_s <= time_s)
        })
        .collect()
}

/// Full bed elevation vector at `time_s`, starting from `declared_bed` and applying every epoch
/// with `epoch.time_s <= time_s` in order.
pub fn active_bed_at(
    declared_bed: &[f64],
    schedules: &[SurfaceBedSchedule],
    time_s: f64,
) -> Vec<f64> {
    let mut bed = declared_bed.to_vec();
    for schedule in schedules {
        let count = schedule
            .epochs
            .partition_point(|epoch| epoch.time_s <= time_s);
        for epoch in &schedule.epochs[..count] {
            for update in &epoch.updates {
                if let Some(slot) = bed.get_mut(update.cell) {
                    *slot = update.bed_elevation_m;
                }
            }
        }
    }
    bed
}

/// 64-bit FNV-1a digest over the exact bit patterns of `bed`.
pub fn bed_digest_bits(bed: &[f64]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in bed {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}
