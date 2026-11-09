//! Closed monotone reference-speed pump curves with explicit provenance.

use crate::structures::{StructureError, StructureInputErrorKind};

/// One supplied quiescent endpoint-stage reference-curve knot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PumpCurveKnot {
    /// Receiving-stage minus donor-stage head, in metres.
    pub head_m: f64,
    /// Nonnegative reference-speed discharge magnitude, in m³/s.
    pub discharge_m3_per_s: f64,
}

pub(crate) type PumpCurveIdentity = (String, Vec<(u64, u64)>);

/// Immutable single-valued bounded pump curve. No extrapolation or implicit shutoff extension applies.
#[derive(Clone, Debug, PartialEq)]
pub struct PumpCurve {
    knots: Vec<PumpCurveKnot>,
    provenance: String,
}
impl PumpCurve {
    /// Check increasing finite heads, nonincreasing nonnegative flow, and explicit curve provenance.
    pub fn new(knots: Vec<PumpCurveKnot>, provenance: String) -> Result<Self, StructureError> {
        crate::structures::definition::text(&provenance, "pump curve provenance")?;
        if knots.len() < 2 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                "pump reference curve requires at least two explicit knots",
            ));
        }
        for (index, knot) in knots.iter().enumerate() {
            if !knot.head_m.is_finite()
                || !knot.discharge_m3_per_s.is_finite()
                || knot.discharge_m3_per_s < 0.0
            {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    format!(
                        "pump knot={index},value={knot:?}: supply finite head and nonnegative finite discharge"
                    ),
                ));
            }
        }
        for (index, pair) in knots.windows(2).enumerate() {
            let width = pair[1].head_m - pair[0].head_m;
            if !width.is_finite()
                || width <= 0.0
                || pair[1].discharge_m3_per_s > pair[0].discharge_m3_per_s
            {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    format!(
                        "pump segment={index},width={width}: require increasing representable head and nonincreasing discharge"
                    ),
                ));
            }
        }
        if !knots.iter().any(|k| k.discharge_m3_per_s > 0.0) {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                "pump curve requires at least one positive discharge knot",
            ));
        }
        Ok(Self { knots, provenance })
    }
    /// Complete exact supplied reference knots.
    pub fn knots(&self) -> &[PumpCurveKnot] {
        &self.knots
    }
    /// Source/revision, SI units, reference speed, fluid applicability, and valid curve envelope evidence.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
    pub(crate) fn key(&self) -> PumpCurveIdentity {
        (
            self.provenance.clone(),
            self.knots
                .iter()
                .map(|k| (k.head_m.to_bits(), k.discharge_m3_per_s.to_bits()))
                .collect(),
        )
    }
}
