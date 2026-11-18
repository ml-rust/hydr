//! Law class, endpoint cutoffs, and response bounds of every link in the table.
//!
//! Each link keeps the cutoffs of [`endpoint_cutoffs`]: a weir's crest, a gate's sill, an
//! orifice's crown, and a pump donor's suction bound. Bank links are weirs with both cutoffs at
//! the crest.

use crate::coupling::links::cutoffs::endpoint_cutoffs;
use crate::structures::{DirectionPolicy, StructureElement, StructureLaw, StructureOperatingPoint};

/// Margin of the suction stop above the suction cutoff, in units of `|cutoff| f64::EPSILON`.
const SUCTION_MARGIN_ULPS: f64 = 16.0;

/// Margin of the largest kept lift below the curve's zero-flow head, in units of
/// `|head| f64::EPSILON`.
const LIFT_MARGIN_ULPS: f64 = 64.0;

/// Pump operands of the suction cap and the response bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in super::super) struct PumpLimits {
    /// Donor side: `0` for A, `1` for B.
    pub donor: usize,
    /// The pump runs over every interval.
    pub running: bool,
    /// Stage at which the suction cap stops the donor, in metres.
    ///
    /// It lies above the donor cutoff by half the head allowance at the cutoff, plus
    /// `SUCTION_MARGIN_ULPS` ulps, so the donor ends strictly above its cutoff.
    pub stop_m: f64,
    /// Largest lift a response probe keeps, in metres: just below the first zero-flow head of the
    /// curve at the running speed, or below its last head.
    pub lift_limit_m: f64,
    /// Free outlet elevation, in metres.
    pub outlet_m: Option<f64>,
}

/// Law class of one link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in super::super) enum LawKind {
    /// Villemonte weir. A receiver at or below the crest keeps free flow.
    Weir,
    /// Rectangular gate aperture. A receiver at or below the sill keeps free flow.
    Gate,
    /// Fully submerged orifice. Both sides stay above the crown.
    Orifice,
    /// Pump on its curve, or isolated.
    Pump(PumpLimits),
}

/// Law class and endpoint cutoffs of one link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in super::super) struct LinkLaw {
    pub kind: LawKind,
    /// Attachment bed of endpoints `[A, B]`, in metres.
    pub bed: [f64; 2],
    /// Cutoff stage of endpoints `[A, B]`, in metres.
    pub cut: [f64; 2],
}

impl LinkLaw {
    /// Law of `element` under `operation` with attachment `beds`.
    ///
    /// `allowance` is the head allowance `(absolute, relative)` of the coordinator.
    pub fn new(
        element: &StructureElement,
        operation: StructureOperatingPoint,
        beds: [f64; 2],
        (absolute, relative): (f64, f64),
    ) -> Self {
        let cut = endpoint_cutoffs(element, beds);
        let kind = match &element.law {
            StructureLaw::SuppressedRectangularWeir(_) => LawKind::Weir,
            StructureLaw::HydrostaticRectangularGate(_) => LawKind::Gate,
            StructureLaw::SubmergedOrifice(_) => LawKind::Orifice,
            StructureLaw::Pump(p) => {
                let donor = usize::from(element.direction == DirectionPolicy::BtoAOnly);
                let cutoff = cut[donor];
                let stop_m = cutoff
                    + 0.5 * (absolute + relative * cutoff.abs())
                    + SUCTION_MARGIN_ULPS * f64::EPSILON * cutoff.abs();
                let knots = p.curve().knots();
                let zero = knots
                    .iter()
                    .find(|k| k.discharge_m3_per_s == 0.0)
                    .or(knots.last())
                    .map_or(0.0, |k| k.head_m);
                let (running, speed) = match operation {
                    StructureOperatingPoint::PumpRunning { speed_ratio } => (true, speed_ratio),
                    _ => (false, 0.0),
                };
                let head = zero - LIFT_MARGIN_ULPS * f64::EPSILON * zero.abs();
                LawKind::Pump(PumpLimits {
                    donor,
                    running,
                    stop_m,
                    lift_limit_m: head * speed * speed,
                    outlet_m: p.free_outlet().map(|o| o.elevation_m),
                })
            }
        };
        Self {
            kind,
            bed: beds,
            cut,
        }
    }

    /// The law moves water only down its head difference.
    pub fn passive(&self) -> bool {
        !matches!(self.kind, LawKind::Pump(_))
    }

    /// The law keeps free flow while the receiver stays at or below its cutoff.
    pub fn has_below(&self) -> bool {
        matches!(self.kind, LawKind::Weir | LawKind::Gate)
    }

    /// Pump limits, or `None` for a passive law.
    pub fn pump(&self) -> Option<&PumpLimits> {
        match &self.kind {
            LawKind::Pump(limits) => Some(limits),
            _ => None,
        }
    }

    /// Largest volume each direction keeps, `[A to B, B to A]`, each nonnegative.
    ///
    /// `heads` are pre-exchange end stages `[A, B]`. `area(side, stage)` is the plan area of a side
    /// at `stage`. `shares` split each side among its links. `available` holds each side's donor
    /// water. Every bound stays inside the law's admitted head range in the plan-area model:
    /// - Weir and gate: the receiver stays at or below its cutoff, or below the donor.
    /// - Orifice: the receiver stays below the donor, and the donor above the crown.
    /// - Pump: its donor stays at or above the suction stop, and the lift below the curve's
    ///   zero-flow head. The other direction keeps nothing.
    pub fn bounds(
        &self,
        heads: [f64; 2],
        area: impl Fn(usize, f64) -> f64,
        shares: [f64; 2],
        available: [f64; 2],
    ) -> [f64; 2] {
        let directional = |donor: usize| -> f64 {
            let receiver = 1 - donor;
            let [h_d, h_r] = [heads[donor], heads[receiver]];
            let [s_d, s_r] = [shares[donor], shares[receiver]];
            let z_r = self.cut[receiver];
            let floor = h_r.max(z_r);
            let level = (h_d - floor) / (s_d / area(donor, floor) + s_r / area(receiver, floor));
            let kept = match &self.kind {
                LawKind::Weir | LawKind::Gate => ((z_r - h_r) * area(receiver, h_r) / s_r)
                    .max(level)
                    .max(0.0),
                LawKind::Orifice => {
                    let z_d = self.cut[donor];
                    level.min((h_d - z_d) * area(donor, z_d) / s_d).max(0.0)
                }
                LawKind::Pump(p) => {
                    if !p.running || p.donor != donor {
                        return 0.0;
                    }
                    let lift = p.outlet_m.map_or(h_r, |outlet| h_r.max(outlet)) - h_d;
                    let suction = (h_d - p.stop_m) * area(donor, p.stop_m) / s_d;
                    let domain = (p.lift_limit_m - lift)
                        / (s_d / area(donor, p.stop_m) + s_r / area(receiver, h_r));
                    suction.min(domain).max(0.0)
                }
            };
            kept.min(available[donor] / s_d)
        };
        [directional(0), directional(1)]
    }
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{gate_law, orifice_law, pump_law, weir_law};
    use super::*;
    use crate::structures::{DirectionPolicy, StructureElement};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn element(law: StructureLaw, direction: DirectionPolicy) -> StructureElement {
        StructureElement {
            id: "s".into(),
            endpoint_a: "surface-cell:0".into(),
            endpoint_b: "reach:r:cell:0".into(),
            datum: "d".into(),
            direction,
            law,
        }
    }

    /// Unit plan area on side A and twice that on side B.
    fn area(side: usize, _stage: f64) -> f64 {
        if side == 0 { 4.0 } else { 8.0 }
    }

    #[test]
    fn a_weir_bound_matches_the_bank_cap_terms() -> TestResult {
        let law = LinkLaw::new(
            &element(weir_law(1.2)?, DirectionPolicy::Bidirectional),
            StructureOperatingPoint::Passive,
            [1.0, 0.5],
            (1e-12, 0.0),
        );
        assert_eq!(law.cut, [1.2, 1.2]);
        // Receiver below the crest: it may fill to the crest.
        let [forward, backward] = law.bounds([1.6, 0.9], area, [1.0, 1.0], [10.0, 10.0]);
        assert_eq!(forward, (1.2 - 0.9) * 8.0);
        // The receiver A stands above the crest, so the backward cap is the level term.
        assert_eq!(backward, 0.0);
        Ok(())
    }

    #[test]
    fn a_gate_keeps_free_flow_up_to_its_sill() -> TestResult {
        let law = LinkLaw::new(
            &element(gate_law(0.8)?, DirectionPolicy::Bidirectional),
            StructureOperatingPoint::GateOpening { opening_m: 0.3 },
            [0.5, 0.5],
            (1e-12, 0.0),
        );
        assert_eq!(law.cut, [0.8, 0.8]);
        assert!(law.has_below() && law.passive());
        let [forward, _] = law.bounds([1.4, 0.6], area, [1.0, 1.0], [10.0, 10.0]);
        let level = (1.4 - 0.8) / (1.0 / 4.0 + 1.0 / 8.0);
        assert_eq!(forward, ((0.8 - 0.6) * 8.0_f64).max(level));
        Ok(())
    }

    #[test]
    fn an_orifice_donor_stays_above_its_crown() -> TestResult {
        let law = LinkLaw::new(
            &element(orifice_law(0.5, 0.7)?, DirectionPolicy::Bidirectional),
            StructureOperatingPoint::Passive,
            [0.5, 0.5],
            (1e-12, 0.0),
        );
        assert_eq!(law.cut, [0.7, 0.7]);
        assert!(!law.has_below());
        // Level heads lie above the crown: the level term binds.
        let [forward, backward] = law.bounds([1.2, 0.8], area, [1.0, 1.0], [10.0, 10.0]);
        assert_eq!(forward, (1.2 - 0.8) / (1.0 / 4.0 + 1.0 / 8.0));
        assert_eq!(backward, 0.0);
        // A deep receiver: the donor reaches the crown first.
        let [forward, _] = law.bounds([0.9, 3.0], |_, _| 4.0, [1.0, 1.0], [10.0, 10.0]);
        assert_eq!(forward, 0.0);
        let [_, backward] = law.bounds([0.9, 3.0], |_, _| 4.0, [1.0, 1.0], [100.0, 100.0]);
        assert_eq!(backward, ((3.0 - 0.9) / 0.5_f64).min((3.0 - 0.7) * 4.0));
        // The donor limit caps every direction.
        let [forward, _] = law.bounds([1.2, 0.8], area, [1.0, 1.0], [0.25, 0.25]);
        assert_eq!(forward, 0.25);
        Ok(())
    }

    #[test]
    fn a_pump_bound_holds_suction_and_its_curve_domain() -> TestResult {
        let law = LinkLaw::new(
            &element(pump_law(0.4, None)?, DirectionPolicy::AtoBOnly),
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            [0.2, 0.5],
            (1e-12, 0.0),
        );
        let Some(pump) = law.pump() else {
            return Err("a pump law".into());
        };
        assert_eq!(pump.donor, 0);
        assert!(pump.stop_m > 0.4 && pump.stop_m < 0.4 + 1e-12);
        assert!(!law.passive());
        // Lift stays below the curve's 3 m zero-flow head.
        let [forward, backward] = law.bounds([1.0, 1.5], area, [1.0, 1.0], [100.0, 100.0]);
        let suction = (1.0 - pump.stop_m) * 4.0;
        let domain = (pump.lift_limit_m - 0.5) / (1.0 / 4.0 + 1.0 / 8.0);
        assert_eq!(forward, suction.min(domain));
        assert_eq!(backward, 0.0);
        // A donor at the stop keeps nothing.
        let [forward, _] = law.bounds([pump.stop_m, 1.5], area, [1.0, 1.0], [100.0, 100.0]);
        assert_eq!(forward, 0.0);
        // A stopped pump keeps nothing either way.
        let stopped = LinkLaw::new(
            &element(pump_law(0.4, None)?, DirectionPolicy::BtoAOnly),
            StructureOperatingPoint::PumpStoppedIsolated,
            [0.2, 0.5],
            (1e-12, 0.0),
        );
        assert_eq!(
            stopped.pump().map(|p| (p.donor, p.running)),
            Some((1, false))
        );
        assert_eq!(
            stopped.bounds([1.0, 1.5], area, [1.0, 1.0], [100.0, 100.0]),
            [0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn a_free_outlet_sets_the_lift_of_a_low_receiver() -> TestResult {
        let law = LinkLaw::new(
            &element(pump_law(0.4, Some(2.0))?, DirectionPolicy::AtoBOnly),
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            [0.2, 0.5],
            (1e-12, 0.0),
        );
        let Some(pump) = law.pump() else {
            return Err("a pump law".into());
        };
        assert_eq!(pump.outlet_m, Some(2.0));
        let [forward, _] = law.bounds([1.0, 0.5], area, [1.0, 1.0], [100.0, 100.0]);
        let domain = (pump.lift_limit_m - (2.0 - 1.0)) / (1.0 / 4.0 + 1.0 / 8.0);
        assert_eq!(forward, ((1.0 - pump.stop_m) * 4.0).min(domain));
        Ok(())
    }
}
