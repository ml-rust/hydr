//! Chained identity of forcing intervals that precede a retained forcing window.

use super::{Axis, BoundaryFace, BoundaryForcing, ExteriorState, ForcingInterval};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

const ORIGIN_TAG: &[u8] = b"hydr-forcing-prefix-1";

/// Folded identity of the forcing intervals that precede a retained window.
///
/// The digest checks consistency between a package and an engine schedule.
/// It establishes no authenticity. Callers own integrity protection.
#[derive(Clone, Debug, PartialEq)]
pub struct ForcingPrefix {
    /// Number of folded intervals.
    pub intervals: usize,
    /// Start of the first folded interval in seconds. Equals `end` for an empty prefix.
    pub start: f64,
    /// End of the last folded interval in seconds. Equals the first retained start.
    pub end: f64,
    /// Chained SHA-256 digest over every folded interval, in schedule order.
    pub digest: [u8; 32],
}

impl ForcingPrefix {
    /// Returns the empty prefix that starts and ends at `start`.
    pub(super) fn empty(start: f64) -> Self {
        Self {
            intervals: 0,
            start,
            end: start,
            digest: origin(),
        }
    }

    /// Checks internal shape against the first retained interval start.
    ///
    /// An empty prefix has equal start and end bits and the origin digest.
    pub(super) fn consistent(&self, first_start: f64) -> bool {
        let shape = if self.intervals == 0 {
            self.start.to_bits() == self.end.to_bits() && self.digest == origin()
        } else {
            self.start < self.end && self.digest != origin()
        };
        shape
            && self.start.is_finite()
            && self.end.is_finite()
            && self.end.to_bits() == first_start.to_bits()
    }
}

/// Returns `D_0 = SHA-256("hydr-forcing-prefix-1")`.
pub(super) fn origin() -> [u8; 32] {
    static ORIGIN: OnceLock<[u8; 32]> = OnceLock::new();
    *ORIGIN.get_or_init(|| Sha256::digest(ORIGIN_TAG).into())
}

/// Returns the digest after each interval, chained from `base`.
pub(super) fn chain(base: &[u8; 32], intervals: &[ForcingInterval]) -> Vec<[u8; 32]> {
    let mut previous = *base;
    intervals
        .iter()
        .map(|interval| {
            previous = link(&previous, interval);
            previous
        })
        .collect()
}

/// Folds the first `count` intervals onto `prefix`.
///
/// A cached `chain` over the same intervals avoids rehashing. Returns `None` past the slice end.
pub(super) fn folded(
    prefix: &ForcingPrefix,
    intervals: &[ForcingInterval],
    chain: Option<&[[u8; 32]]>,
    count: usize,
) -> Option<ForcingPrefix> {
    if count == 0 {
        return Some(prefix.clone());
    }
    let folded = intervals.get(..count)?;
    let digest = match chain {
        Some(chain) => *chain.get(count - 1)?,
        None => folded.iter().fold(prefix.digest, |previous, interval| {
            link(&previous, interval)
        }),
    };
    Some(ForcingPrefix {
        intervals: prefix.intervals.checked_add(count)?,
        start: prefix.start,
        end: folded.last()?.end,
        digest,
    })
}

fn link(previous: &[u8; 32], interval: &ForcingInterval) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(previous);
    encode(&mut hasher, interval);
    hasher.finalize().into()
}

fn encode(hasher: &mut Sha256, interval: &ForcingInterval) {
    real(hasher, interval.start);
    real(hasher, interval.end);
    for rates in [
        &interval.rain,
        &interval.potential_evaporation,
        &interval.withdrawal,
    ] {
        length(hasher, rates.len());
        rates.iter().for_each(|rate| real(hasher, *rate));
    }
    length(hasher, interval.boundaries.len());
    for boundary in &interval.boundaries {
        match boundary {
            BoundaryForcing::Flux(port) => {
                hasher.update([0]);
                face(hasher, port.face);
                real(hasher, port.water_rate);
                real(hasher, port.momentum_rate[0]);
                real(hasher, port.momentum_rate[1]);
                real(hasher, port.signal_speed);
                length(hasher, port.provenance.len());
                hasher.update(port.provenance.as_bytes());
            }
            BoundaryForcing::Stage(stage) => {
                hasher.update([1]);
                face(hasher, stage.face);
                real(hasher, stage.elevation);
                optional(hasher, stage.inflow_tangential_velocity);
            }
            BoundaryForcing::Discharge(discharge) => {
                hasher.update([2]);
                face(hasher, discharge.face);
                real(hasher, discharge.outward_rate);
                optional(hasher, discharge.inflow_tangential_velocity);
            }
            BoundaryForcing::Exterior(exterior) => {
                hasher.update([3]);
                face(hasher, exterior.face);
                match exterior.state {
                    ExteriorState::Dry => hasher.update([0]),
                    ExteriorState::Wet {
                        elevation,
                        outward_velocity,
                        tangential_velocity,
                    } => {
                        hasher.update([1]);
                        real(hasher, elevation);
                        real(hasher, outward_velocity);
                        real(hasher, tangential_velocity);
                    }
                }
            }
            BoundaryForcing::Transmissive(value) => {
                hasher.update([4]);
                face(hasher, *value);
            }
        }
    }
}

fn real(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_bits().to_le_bytes());
}

fn length(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn optional(hasher: &mut Sha256, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            real(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn face(hasher: &mut Sha256, value: BoundaryFace) {
    hasher.update([match value.axis {
        Axis::X => 0,
        Axis::Y => 1,
    }]);
    length(hasher, value.column);
    length(hasher, value.row);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::FluxPort;

    fn interval(start: f64, rain: f64) -> ForcingInterval {
        ForcingInterval {
            start,
            end: start + 1.0,
            rain: vec![rain, 0.0],
            potential_evaporation: vec![0.0; 2],
            withdrawal: vec![0.0; 2],
            boundaries: vec![BoundaryForcing::Flux(FluxPort {
                face: BoundaryFace {
                    axis: Axis::X,
                    column: 0,
                    row: 0,
                },
                water_rate: 0.0,
                momentum_rate: [0.0; 2],
                signal_speed: 1.0,
                provenance: "gauge".into(),
            })],
        }
    }

    fn schedule() -> Vec<ForcingInterval> {
        (0..5)
            .map(|k| interval(k as f64, 1e-3 * k as f64))
            .collect()
    }

    #[test]
    fn fold_in_two_parts_equals_one_fold() -> Result<(), &'static str> {
        let intervals = schedule();
        let empty = ForcingPrefix::empty(0.0);
        let whole = folded(&empty, &intervals, None, 5).ok_or("whole fold")?;
        let head = folded(&empty, &intervals, None, 2).ok_or("head fold")?;
        let tail = folded(&head, &intervals[2..], None, 3).ok_or("tail fold")?;
        assert_eq!(whole, tail);
        let cached = chain(&empty.digest, &intervals);
        assert_eq!(folded(&empty, &intervals, Some(&cached), 5), Some(whole));
        assert_eq!(folded(&empty, &intervals, None, 6), None);
        assert!(tail.consistent(5.0));
        assert!(!tail.consistent(4.0));
        Ok(())
    }

    #[test]
    fn encoding_distinguishes_signed_zero_and_provenance() {
        let base = origin();
        let positive = interval(0.0, 0.0);
        let mut negative = positive.clone();
        negative.rain[1] = -0.0;
        assert_ne!(link(&base, &positive), link(&base, &negative));
        let mut renamed = positive.clone();
        if let BoundaryForcing::Flux(port) = &mut renamed.boundaries[0] {
            port.provenance = "gauge2".into();
        }
        assert_ne!(link(&base, &positive), link(&base, &renamed));
        assert_eq!(link(&base, &positive), link(&base, &positive.clone()));
    }

    #[test]
    fn empty_prefix_requires_origin_digest() {
        let empty = ForcingPrefix::empty(2.0);
        assert!(empty.consistent(2.0));
        let mut forged = empty.clone();
        forged.digest[0] ^= 1;
        assert!(!forged.consistent(2.0));
        forged = empty;
        forged.intervals = 1;
        assert!(!forged.consistent(2.0));
    }
}
