//! Immutable closed-range storage curves with explicit datum and provenance.

use super::LevelPoolError;

/// One exact supplied volume/elevation knot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelPoolKnot {
    /// Finite nonnegative storage volume in m³.
    pub volume_m3: f64,
    /// Finite elevation in metres on the supplied vertical reference.
    pub stage_m: f64,
}

pub(crate) type CurveIdentity = (String, String, Vec<(u64, u64)>);

/// Checked piecewise-linear stage as a function of authoritative volume.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolCurve {
    knots: Vec<LevelPoolKnot>,
    vertical_reference: String,
    provenance: String,
}
impl LevelPoolCurve {
    /// Check strictly increasing finite knots, positive secant areas, and explicit source metadata.
    pub fn new(
        knots: Vec<LevelPoolKnot>,
        vertical_reference: String,
        provenance: String,
    ) -> Result<Self, LevelPoolError> {
        super::definition::text(&vertical_reference, "vertical_reference")?;
        super::definition::text(&provenance, "curve.provenance")?;
        if knots.len() < 2 || knots.first().is_none_or(|k| k.volume_m3 != 0.0) {
            return Err(LevelPoolError::input(
                "curve requires at least two knots with first volume exactly zero",
            ));
        }
        for (index, knot) in knots.iter().enumerate() {
            if !knot.volume_m3.is_finite() || !knot.stage_m.is_finite() || knot.volume_m3 < 0.0 {
                return Err(LevelPoolError::input(format!(
                    "knot={index},value={knot:?}: supply finite nonnegative volume and finite stage"
                )));
            }
        }
        for (index, pair) in knots.windows(2).enumerate() {
            let dv = pair[1].volume_m3 - pair[0].volume_m3;
            let dh = pair[1].stage_m - pair[0].stage_m;
            let area = dv / dh;
            if !dv.is_finite()
                || dv <= 0.0
                || !dh.is_finite()
                || dh <= 0.0
                || !area.is_finite()
                || area <= 0.0
            {
                return Err(LevelPoolError::input(format!(
                    "segment={index},delta_volume={dv},delta_stage={dh},secant_area={area}: require strictly increasing representable knots and positive finite area"
                )));
            }
        }
        Ok(Self {
            knots,
            vertical_reference,
            provenance,
        })
    }
    /// Exact supplied knots in increasing volume order.
    pub fn knots(&self) -> &[LevelPoolKnot] {
        &self.knots
    }
    /// Explicit elevation reference. No datum transformation occurs.
    pub fn vertical_reference(&self) -> &str {
        &self.vertical_reference
    }
    /// Immutable source evidence.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
    /// Maximum supported storage volume in m³.
    pub fn maximum_volume_m3(&self) -> f64 {
        self.knots[self.knots.len() - 1].volume_m3
    }
    #[doc(hidden)]
    pub fn origin(&self) -> f64 {
        self.knots[0].stage_m
    }
    pub(crate) fn key(&self) -> CurveIdentity {
        (
            self.vertical_reference.clone(),
            self.provenance.clone(),
            self.knots
                .iter()
                .map(|k| (k.volume_m3.to_bits(), k.stage_m.to_bits()))
                .collect(),
        )
    }
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.vertical_reference == other.vertical_reference
            && self.provenance == other.provenance
            && self.knots.len() == other.knots.len()
            && self.knots.iter().zip(&other.knots).all(|(a, b)| {
                a.volume_m3.to_bits() == b.volume_m3.to_bits()
                    && a.stage_m.to_bits() == b.stage_m.to_bits()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::coupling_cuts::fixture::batch;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn nonfinite_static_segment_difference_returns_typed_input_error() -> TestResult {
        let result = batch(&[0.0, 1.0e308], &[-1.0e308, 1.0e308], 0.0);
        assert!(matches!(result, Err(LevelPoolError::InvalidInput { .. })));
        Ok(())
    }
}
