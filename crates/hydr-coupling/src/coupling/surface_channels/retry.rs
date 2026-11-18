//! Shortening rules of a rejected common attempt.

use super::SurfaceChannelError;

/// Smallest fraction of a stalled duration the next attempt keeps.
const STALL_FLOOR: f64 = 0.125;

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
