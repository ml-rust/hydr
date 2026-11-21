//! Shared helpers of the standalone surveyed endpoint tests.
//!
//! `Section` holds the exact band polynomials of a frozen reference profile. The mass checks
//! apply the frozen "Mass, standalone" criterion to every accepted interval.

use super::support::{Reach, TestResult, check_water, forcing, profile, reach};
use super::surveyed_profiles::Profile;
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelFluxPort, ChannelSurveyedConfig, CrossSectionDefinition,
};

/// Simpson panels of the invariant `φ`.
const PHI_PANELS: usize = 2000;

/// Exact geometry of one frozen profile with its bed at elevation zero.
#[derive(Clone, Copy)]
pub struct Section {
    pub profile: Profile,
    pub gravity: f64,
}

impl Section {
    pub fn new(profile: Profile, gravity: f64) -> Self {
        Self { profile, gravity }
    }

    fn points(&self) -> Vec<(f64, f64)> {
        self.profile
            .points
            .iter()
            .map(|p| (f64::from_bits(p[0]), f64::from_bits(p[1])))
            .collect()
    }

    /// Station where one monotone bank reaches elevation `y`.
    fn station(bank: &[(f64, f64)], y: f64) -> f64 {
        for pair in bank.windows(2) {
            let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
            let (low, high) = (y0.min(y1), y0.max(y1));
            if y >= low && y <= high && high > low {
                return x0 + (x1 - x0) * (y - y0) / (y1 - y0);
            }
        }
        bank[0].0
    }

    /// Top width at depth `y`.
    pub fn width(&self, y: f64) -> f64 {
        let points = self.points();
        let first = points.iter().position(|p| p.1 == 0.0).unwrap_or(0);
        let last = points.iter().rposition(|p| p.1 == 0.0).unwrap_or(0);
        Self::station(&points[last..], y) - Self::station(&points[..=first], y)
    }

    /// Depth intervals between band knots, cut at `d`.
    fn pieces(&self, d: f64) -> Vec<(f64, f64)> {
        self.profile
            .knots
            .windows(2)
            .filter(|k| k[0] < d)
            .map(|k| (k[0], k[1].min(d)))
            .collect()
    }

    /// Flow area. Width is linear on each band, so the trapezoid rule is exact.
    pub fn area(&self, d: f64) -> f64 {
        self.pieces(d)
            .into_iter()
            .map(|(a, b)| 0.5 * (self.width(a) + self.width(b)) * (b - a))
            .sum()
    }

    /// Hydrostatic integral `I(d)`. Its integrand is quadratic on each band, so Simpson is exact.
    pub fn hydrostatic(&self, d: f64) -> f64 {
        let f = |y: f64| (d - y) * self.width(y);
        self.pieces(d)
            .into_iter()
            .map(|(a, b)| (b - a) / 6.0 * (f(a) + 4.0 * f(0.5 * (a + b)) + f(b)))
            .sum()
    }

    /// Sound speed `sqrt(g A / T)`.
    pub fn sound(&self, d: f64) -> f64 {
        (self.gravity * self.area(d) / self.width(d)).sqrt()
    }

    /// Riemann invariant `φ(d)`, with `y = d x²` to remove the bed singularity.
    pub fn invariant(&self, d: f64) -> f64 {
        let f = |x: f64| {
            if x == 0.0 {
                return 2.0 * (self.gravity * d).sqrt();
            }
            let y = d * x * x;
            (self.gravity * self.width(y) / self.area(y)).sqrt() * 2.0 * d * x
        };
        let h = 1.0 / PHI_PANELS as f64;
        let mut sum = f(0.0) + f(1.0);
        for i in 1..PHI_PANELS {
            sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    pub fn definition(&self) -> CrossSectionDefinition {
        profile(self.profile.name, self.profile.points, self.profile.cap)
    }

    /// Default reach config at the section gravity.
    pub fn config(&self) -> ChannelConfig {
        ChannelConfig {
            gravity: self.gravity,
            ..ChannelConfig::default()
        }
    }

    /// Reach of `count` cells of `length` with every cell at area `area` and discharge `q`.
    pub fn reach(&self, count: usize, length: f64, area: f64, discharge: f64) -> TestResult<Reach> {
        self.reach_with(count, length, area, discharge, self.config())
    }

    /// [`Section::reach`] with an explicit reach config.
    pub fn reach_with(
        &self,
        count: usize,
        length: f64,
        area: f64,
        discharge: f64,
        config: ChannelConfig,
    ) -> TestResult<Reach> {
        reach::<crate::runtime::Runtime>(
            crate::runtime::client(),
            self.definition(),
            &vec![length; count],
            &vec![0.0; count],
            &vec![area; count],
            &vec![discharge; count],
            config,
            ChannelSurveyedConfig::default(),
        )
    }

    /// Port with zero water, the exact pressure `g I(A)` of the rest state at depth `d`, and a
    /// speed two ulps above the end cell characteristic speed.
    pub fn rest_port(&self, d: f64) -> TestResult<ChannelBoundary> {
        let speed = self.sound(d).next_up().next_up();
        Ok(ChannelBoundary::FluxPort(ChannelFluxPort::new(
            0.0,
            self.gravity * self.hydrostatic(d),
            speed,
            Some(self.invariant(d)),
            "test provider of the exact surveyed rest flux",
        )?))
    }
}

/// Advances `current` by one interval of at most `target - time`, then checks mass.
pub fn step_checked(
    current: &Reach,
    target: f64,
    ends: &[ChannelBoundary; 2],
) -> TestResult<Reach> {
    let start = current.time();
    let next = current
        .step(target, &forcing(start, target, ends.clone()))?
        .next;
    assert!(next.time() > start && next.time() <= target);
    check_interval(&next, ends)?;
    Ok(next)
}

/// Frozen "Mass, standalone" criterion for the last accepted interval of `reach`.
///
/// - The ledger residual and its charges stay within the volume tolerance.
/// - A `Discharge` end face volume equals `RN(σ Q dt)` bit for bit.
/// - A `FluxPort` end face volume and momentum equal `RN(rate · dt)` bit for bit.
/// - Both end face water volume low parts equal `+0.0`.
/// - The end face transfer bound covers the product tail.
pub fn check_interval(reach: &Reach, ends: &[ChannelBoundary; 2]) -> TestResult {
    check_water(reach)?;
    let receipt = reach.last_step()?.ok_or("missing interval receipt")?;
    let dt = receipt.end_s - receipt.start_s;
    let last = receipt.face_volume_m3.len() - 1;
    for (side, face) in [(0, 0), (1, last)] {
        // A wall face is a Riemann face with a two-part hydrostatic momentum, not a direct face.
        if matches!(ends[side], ChannelBoundary::Wall) {
            continue;
        }
        assert_eq!(
            receipt.face_volume_low_m3[face].to_bits(),
            0.0_f64.to_bits(),
            "face {face} volume low part, end {:?}",
            ends[side]
        );
        // Momentum low parts stay unchecked. A proven-rest scalar end face takes the two-part
        // Riemann momentum, so exact rest holds.
        // Global rate: toward +x at either face.
        let sigma = if side == 0 { -1.0 } else { 1.0 };
        let (water, momentum) = match &ends[side] {
            ChannelBoundary::Discharge {
                discharge_m3_per_s, ..
            } => (*discharge_m3_per_s, None),
            ChannelBoundary::FluxPort(port) => (
                sigma * port.outward_water_rate_m3_per_s(),
                Some(port.outward_momentum_rate_m4_per_s2()),
            ),
            _ => continue,
        };
        let volume = receipt.face_volume_m3[face];
        // A port face takes `σ` with the rate, so a zero rate keeps no sign. Adding `+0.0`
        // maps `-0.0` to `+0.0` and leaves every other value unchanged.
        let (expected, actual) = match momentum {
            Some(_) => (water * dt + 0.0, volume + 0.0),
            None => (water * dt, volume),
        };
        assert_eq!(actual.to_bits(), expected.to_bits(), "face {face}");
        if let Some(rate) = momentum {
            assert_eq!(
                receipt.face_momentum_m4_per_s[face].to_bits(),
                (rate * dt).to_bits(),
                "face {face}"
            );
        }
        let tail = water.mul_add(dt, -volume).abs();
        assert!(
            receipt.face_transfer_underflow_bound_m3[face] >= tail,
            "face {face}: bound={:e}, tail={tail:e}",
            receipt.face_transfer_underflow_bound_m3[face]
        );
    }
    Ok(())
}
