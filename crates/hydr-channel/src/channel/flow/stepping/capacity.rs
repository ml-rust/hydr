//! Interval retries after a candidate scalar trace rejects near its sonic outflow capacity.
//!
//! A leaving `Discharge` trace exists only while the target `q` stays below the sonic
//! outflow of the end cell. The margin `m(t) = A u_b (d_s) - q` moves smoothly with the end
//! cell state. When the accepted state holds `m > 0` and the candidate after `dt` holds
//! `m < 0`, the margin crosses zero near `dt m_s / (m_s - m_c)`. A shorter interval up to that
//! crossing fixes the candidate. Halving instead approaches the crossing in ever shorter
//! steps. Once the accepted margin falls inside its own enclosure width, or the minimum
//! step already loses capacity, no admissible interval keeps the trace: it rejects.
//!
//! Near the sonic depth `dG/dd` vanishes, so the root depth enclosure widens like
//! `1 / (c - u)`. A trace whose enclosure cannot meet a control at any quadrature level sits
//! at its precision floor. When the candidate margin falls certainly below the accepted
//! margin and stays above half of it, the floor lies within the first half of the remaining
//! capacity. Shorter intervals then only approach the floor: the trace rejects.

use crate::channel::flow::{ChannelError, boundary::boundary_rejection, surveyed_boundary::Margin};

/// Fraction of the estimated crossing taken by the next interval.
const REACH: f64 = 15.0 / 16.0;

/// The next action after a candidate scalar trace rejects at interval `dt`.
#[derive(Debug, PartialEq)]
pub(super) enum CapacityRetry {
    /// A shorter interval can fix the rejection: halve the interval.
    Halve,
    /// Retry at this shorter interval.
    Shorten(f64),
    /// No admissible interval keeps the trace.
    Lost(ChannelError),
}

/// The retry rule for accepted margin `start` and candidate margin `candidate` at `dt`.
///
/// `exhausted` marks a candidate rejection that no quadrature level resolves.
pub(super) fn capacity_retry(
    start: Option<Margin>,
    candidate: Option<Margin>,
    exhausted: bool,
    dt: f64,
    min_step: f64,
) -> CapacityRetry {
    let (Some(start), Some(candidate)) = (start, candidate) else {
        return CapacityRetry::Halve;
    };
    let lost = |cause: &str| {
        CapacityRetry::Lost(boundary_rejection(&format!(
            "requested discharge exceeds the sonic outflow of the end cell: {cause}, accepted margin={:e} within {:e}, candidate margin={:e} after dt={dt:e}",
            start.value, start.width, candidate.value
        )))
    };
    if !candidate.lost() {
        let falling = start.value - candidate.value > 0.5 * (start.width + candidate.width);
        if exhausted && falling && candidate.value > 0.5 * start.value {
            return lost("the trace reaches its precision floor while the margin falls");
        }
        return CapacityRetry::Halve;
    }
    if !start.holds() {
        return lost("the accepted margin lies inside its enclosure");
    }
    if dt <= min_step {
        return lost("the minimum step loses capacity");
    }
    let crossing = dt * (start.value / (start.value - candidate.value));
    let next = (REACH * crossing).min(REACH * dt).max(min_step);
    if next.is_finite() && next < dt {
        CapacityRetry::Shorten(next)
    } else {
        CapacityRetry::Halve
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn margin(value: f64, width: f64) -> Option<Margin> {
        Some(Margin { value, width })
    }

    fn rejects(retry: &CapacityRetry) -> bool {
        matches!(retry, CapacityRetry::Lost(error)
            if error.to_string().contains("exceeds the sonic outflow"))
    }

    /// Frozen criteria of the retry rule:
    /// - a missing margin or an uncertain candidate loss halves
    /// - a certain loss shortens to 15/16 of the linear crossing
    /// - an accepted margin inside its width, or a lost minimum step, rejects
    /// - a precision floor rejects only with a falling margin above half the accepted one
    #[test]
    fn capacity_retry_separates_shortening_from_loss() {
        let halve = CapacityRetry::Halve;
        assert_eq!(
            capacity_retry(None, margin(-1.0, 0.0), false, 1.0, 1e-12),
            halve
        );
        assert_eq!(
            capacity_retry(margin(1.0, 0.0), margin(-1e-3, 4e-3), false, 1.0, 1e-12),
            halve
        );
        assert_eq!(
            capacity_retry(margin(1.0, 0.0), margin(-3.0, 0.0), false, 1.0, 1e-12),
            CapacityRetry::Shorten(REACH * 0.25)
        );
        for retry in [
            capacity_retry(margin(1e-12, 4e-12), margin(-1.0, 0.0), false, 1.0, 1e-12),
            capacity_retry(margin(1.0, 0.0), margin(-1.0, 0.0), false, 1e-12, 1e-12),
            capacity_retry(margin(1.0, 0.0), margin(0.6, 0.0), true, 1.0, 1e-12),
        ] {
            assert!(rejects(&retry), "{retry:?}");
        }
        for (candidate, exhausted) in [(0.6, false), (0.4, true), (1.0, true)] {
            assert_eq!(
                capacity_retry(
                    margin(1.0, 0.0),
                    margin(candidate, 0.0),
                    exhausted,
                    1.0,
                    1e-12
                ),
                halve
            );
        }
    }
}
