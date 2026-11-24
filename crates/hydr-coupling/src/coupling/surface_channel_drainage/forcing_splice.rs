//! Conservative surface-forcing splice for injecting inlet capture/surcharge rates over `[start_s, end_s]`.

use super::error::SurfaceChannelDrainageError;
use crate::coupling::surface_drainage::exchange::ResolvedInlet;
use crate::solver::{BoundaryConditions, ForcingInterval, ForcingSchedule};
use hydr_core::model::Domain;

/// Builds a spliced [`ForcingSchedule`] starting at `start_s` that adds drainage inlet capture
/// (as `withdrawal`) and pressurized surcharge (as `rain`) over `[start_s, end_s]`, preserving
/// `base` verbatim after `end_s`.
#[allow(clippy::too_many_arguments)]
pub(super) fn splice_drainage_forcing(
    base: &ForcingSchedule,
    domain: &Domain,
    boundaries: &BoundaryConditions,
    start_s: f64,
    end_s: f64,
    inlets: &[ResolvedInlet],
    transfers_m3: &[f64],
) -> Result<ForcingSchedule, SurfaceChannelDrainageError> {
    let dt_s = end_s - start_s;
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(SurfaceChannelDrainageError::input(format!(
            "invalid splice interval [{start_s}, {end_s}]"
        )));
    }
    let cell_area = domain.grid().cell_area();
    let intervals = base.intervals();
    let Some(cover_idx) = intervals
        .iter()
        .position(|iv| iv.start <= start_s && start_s < iv.end)
    else {
        return Err(SurfaceChannelDrainageError::input(format!(
            "base surface forcing schedule does not cover start_s={start_s}"
        )));
    };
    let cover = &intervals[cover_idx];
    if end_s > cover.end + 1e-12 {
        return Err(SurfaceChannelDrainageError::input(format!(
            "splice end_s={end_s} exceeds covering base forcing interval end {}",
            cover.end
        )));
    }
    let clamped_end = end_s.min(cover.end);

    let mut active = cover.clone();
    active.start = start_s;
    active.end = clamped_end;
    for (k, inlet) in inlets.iter().enumerate() {
        let vol = transfers_m3[k];
        if vol > 0.0 {
            // Positive transfer: surface -> drainage capture (withdrawal on surface cell).
            active.withdrawal[inlet.spec.surface_cell] += vol / (cell_area * dt_s);
        } else if vol < 0.0 {
            // Negative transfer: drainage -> surface surcharge (inflow on surface cell).
            active.rain[inlet.spec.surface_cell] += (-vol) / (cell_area * dt_s);
        }
    }

    let mut out = Vec::with_capacity(intervals.len() - cover_idx + 1);
    out.push(active);
    if clamped_end < cover.end - 1e-12 {
        let mut remainder = cover.clone();
        remainder.start = clamped_end;
        out.push(remainder);
    }
    for iv in &intervals[cover_idx + 1..] {
        out.push(iv.clone());
    }
    ForcingSchedule::new(domain, boundaries, out).map_err(|e| {
        SurfaceChannelDrainageError::Numerical(format!("spliced forcing rejected: {e}"))
    })
}

/// Returns the end time of the base forcing interval covering `time_s`.
pub(super) fn covering_forcing_end(
    base: &ForcingSchedule,
    time_s: f64,
) -> Result<f64, SurfaceChannelDrainageError> {
    base.intervals()
        .iter()
        .find(|iv| iv.start <= time_s && time_s < iv.end)
        .map(|iv| iv.end)
        .ok_or_else(|| {
            SurfaceChannelDrainageError::input(format!(
                "base surface forcing schedule does not cover time_s={time_s}"
            ))
        })
}

/// Re-builds a [`ForcingInterval`] slice from `base` starting at `time_s` (for rollback/restore).
#[allow(dead_code)]
pub(super) fn _check_interval_type(_: &ForcingInterval) {}
