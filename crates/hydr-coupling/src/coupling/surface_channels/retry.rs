//! Shortening rules of a rejected common attempt.

use super::SurfaceChannelError;
use crate::channel::ChannelError;

/// Smallest fraction of a stalled duration the next attempt keeps.
const STALL_FLOOR: f64 = 0.125;

/// Relative time tolerance of certified bisection when locating a cutoff crossing.
pub(super) const CROSSING_TIME_RTOL: f64 = 1e-5;

/// Suggested shorter endpoint from a retryable `RecomputeInterval` rejection, if any.
pub(super) fn suggested_end(error: &SurfaceChannelError) -> Option<f64> {
    match error {
        SurfaceChannelError::RecomputeInterval {
            suggested_end_s, ..
        }
        | SurfaceChannelError::Channel {
            source:
                ChannelError::RecomputeInterval {
                    suggested_end_s, ..
                },
            ..
        } => Some(*suggested_end_s),
        _ => None,
    }
}

/// Linear interpolation estimate of a zero crossing in `(start, end)` from `start_margin > 0`
/// and `end_margin <= 0`, clamped strictly inside `(start, end)`.
pub(super) fn linear_crossing_estimate(
    start: f64,
    end: f64,
    start_margin: f64,
    end_margin: f64,
) -> f64 {
    let dt = end - start;
    if start_margin.is_finite() && end_margin.is_finite() && start_margin > 0.0 && end_margin <= 0.0
    {
        let denom = start_margin - end_margin;
        if denom > 0.0 && denom.is_finite() {
            let frac = (start_margin / denom).clamp(0.05, 0.95);
            return start + frac * dt;
        }
    }
    start + 0.5 * dt
}

/// Fraction of a stalled duration the next attempt keeps.
///
/// The largest of `1/2`, `1/4`, `1/8` whose product with `contraction` is at most `target`,
/// else `1/8`. Powers of two keep the rule exact.
pub(super) fn shortening(contraction: f64, target: f64) -> f64 {
    let mut fraction = 0.5;
    while fraction > STALL_FLOOR && contraction * fraction > target {
        fraction *= 0.5;
    }
    fraction
}

/// Computes the next shorter end time for a non-crossing retryable rejection, or returns
/// `RecomputeInterval` / `Unresolved`.
#[allow(clippy::too_many_arguments)]
pub(super) fn next_retry_end(
    start: f64,
    end: f64,
    requested: f64,
    minimum: f64,
    retries: usize,
    retry_limit: usize,
    contraction_target: f64,
    stalled: Option<super::iteration::Stall>,
    requires_exact_span: bool,
    error: &SurfaceChannelError,
) -> Result<f64, SurfaceChannelError> {
    let dt = end - start;
    let shorter = suggested_end(error)
        .filter(|&t| t.is_finite() && t > start && t < end)
        .unwrap_or_else(|| {
            let fraction = stalled.map_or(0.5, |s| shortening(s.contraction, contraction_target));
            start + fraction * dt
        });
    if requires_exact_span {
        return Err(recompute(
            start,
            requested,
            shorter,
            minimum,
            retries,
            error.to_string(),
        ));
    }
    if retries >= retry_limit || shorter - start < minimum || shorter <= start || shorter >= end {
        return Err(SurfaceChannelError::Unresolved {
            dt_s: dt,
            retries,
            reason: error.to_string(),
        });
    }
    Ok(shorter)
}

/// Recompute request for a shorter integrated span, or `Unresolved` when none is admissible.
pub(super) fn recompute(
    start: f64,
    requested: f64,
    suggested: f64,
    minimum: f64,
    retries: usize,
    reason: String,
) -> SurfaceChannelError {
    if !suggested.is_finite()
        || suggested <= start
        || suggested >= requested
        || suggested - start < minimum
    {
        SurfaceChannelError::Unresolved {
            dt_s: suggested - start,
            retries,
            reason,
        }
    } else {
        SurfaceChannelError::RecomputeInterval {
            start_s: start,
            requested_end_s: requested,
            suggested_end_s: suggested,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortening_keeps_the_largest_power_of_two_under_the_target() {
        assert_eq!(shortening(0.9, 0.5), 0.5);
        assert_eq!(shortening(1.0, 0.5), 0.5);
        assert_eq!(shortening(1.2, 0.5), 0.25);
        assert_eq!(shortening(3.0, 0.5), 0.125);
        assert_eq!(shortening(100.0, 0.5), 0.125);
    }
}
