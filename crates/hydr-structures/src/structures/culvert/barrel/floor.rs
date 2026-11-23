//! Flat beds in barrels: sediment beds and the own floors of boxes and flat-floor arches.
//!
//! A bed at depth `h` leaves the open section above it, of rise `D' = D - h`. The admitted
//! depth is `D - fl(D - h)`, so `D'`, `h` and `h + D' = D` are exact F64 values.

use super::{CulvertBarrel, CulvertBarrelShape, FLOOR_FRACTION, OVERFLOW_MARGIN};
use crate::structures::StructureError;

/// Flat bed of a barrel: its depth above the invert and its Manning `n` against the walls'.
///
/// A sediment bed fills every barrel to one depth along its whole length. A box or a
/// flat-floor arch also takes depth zero: its own floor with another roughness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertFloor {
    /// Bed depth `h` above the invert, in metres, with `0 <= h < D`.
    pub sediment_depth_m: f64,
    /// Bed-to-wall Manning ratio `rho = n_bed / n_wall`, dimensionless.
    pub roughness_ratio: f64,
}

impl CulvertBarrel {
    /// The barrel with `floor`, replacing any floor it had.
    ///
    /// Stores the depth `D - fl(D - h)`, which differs from `h` by at most half an F64 step of
    /// `D - h`. Rejects a nonfinite or negative depth, a depth at or above the rise, which is a
    /// closed barrel, and a zero depth on a barrel without a floor. Rejects a nonfinite or
    /// nonpositive ratio, a ratio whose `rho^(3/2)` times the span leaves the F64 range, and an
    /// open rise whose `2^-30` fraction is subnormal.
    pub fn with_floor(self, floor: CulvertFloor) -> Result<Self, StructureError> {
        let base = self.base();
        let CulvertFloor {
            sediment_depth_m: depth,
            roughness_ratio: ratio,
        } = floor;
        let rise = base.rise_m;
        if !depth.is_finite() {
            return Err(StructureError::definition(format!(
                "sediment_depth_m={depth:?}: supply a finite sediment depth in metres"
            )));
        }
        if depth < 0.0 {
            return Err(StructureError::definition(format!(
                "sediment_depth_m={depth:?}: supply a nonnegative sediment depth in metres"
            )));
        }
        if depth >= rise {
            return Err(StructureError::definition(format!(
                "sediment_depth_m={depth:?},rise_m={rise:?}: the barrel is full. Declare StructureCondition::Closed"
            )));
        }
        let open = rise - depth;
        // Sterbenz: rise - open is exact for open in [rise/2, rise], and depth >= rise/2 is
        // itself exact, so this depth makes D - h exact.
        let bed = rise - open;
        if bed == 0.0
            && !matches!(
                base.shape,
                CulvertBarrelShape::Box | CulvertBarrelShape::Arch
            )
        {
            let what = shape_name(base.shape);
            return Err(StructureError::definition(if depth == 0.0 {
                format!(
                    "sediment_depth_m=0: {what} has no floor. Supply a positive sediment depth or omit the floor"
                )
            } else {
                format!(
                    "sediment_depth_m={depth:?},rise_m={rise:?}: the sediment depth rounds to zero against the rise. Supply a larger sediment depth or omit the floor"
                )
            }));
        }
        if !(ratio.is_finite() && ratio > 0.0) {
            return Err(StructureError::definition(format!(
                "roughness_ratio={ratio:?}: supply a finite positive bed-to-wall Manning n ratio"
            )));
        }
        let weight = ratio * ratio.sqrt();
        if !(weight.is_normal() && (weight * base.span_m * OVERFLOW_MARGIN).is_finite()) {
            return Err(StructureError::definition(format!(
                "roughness_ratio={ratio:?},span_m={:?}: ratio^(3/2) times the span leaves the F64 range. Supply a ratio nearer 1",
                base.span_m
            )));
        }
        let smallest = f64::MIN_POSITIVE / FLOOR_FRACTION;
        if open < smallest {
            return Err(StructureError::definition(format!(
                "sediment_depth_m={depth:?},rise_m={rise:?}: the open rise times 2^-30 is subnormal. Supply a smaller sediment depth or declare StructureCondition::Closed"
            )));
        }
        Ok(Self {
            floor: Some(CulvertFloor {
                sediment_depth_m: bed,
                roughness_ratio: ratio,
            }),
            ..base
        })
    }

    /// The declared floor, with its admitted depth.
    pub fn floor(&self) -> Option<CulvertFloor> {
        self.floor
    }

    /// The barrel without its floor.
    pub fn base(&self) -> CulvertBarrel {
        Self {
            floor: None,
            ..*self
        }
    }

    /// Rise of the open section above the bed, `D - h` in metres. `D` without a floor.
    pub fn open_rise_m(&self) -> f64 {
        match self.floor {
            Some(floor) => self.rise_m - floor.sediment_depth_m,
            None => self.rise_m,
        }
    }
}

fn shape_name(shape: CulvertBarrelShape) -> &'static str {
    match shape {
        CulvertBarrelShape::Circular => "a circular barrel",
        CulvertBarrelShape::Box => "a box barrel",
        CulvertBarrelShape::HorizontalEllipse => "a horizontal ellipse",
        CulvertBarrelShape::VerticalEllipse => "a vertical ellipse",
        CulvertBarrelShape::PipeArch => "a pipe-arch",
        CulvertBarrelShape::Arch => "a flat-floor arch",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;

    fn floor(sediment_depth_m: f64, roughness_ratio: f64) -> CulvertFloor {
        CulvertFloor {
            sediment_depth_m,
            roughness_ratio,
        }
    }

    fn message(result: Result<CulvertBarrel, StructureError>) -> Option<String> {
        match result {
            Err(StructureError::InvalidInput {
                kind: StructureInputErrorKind::Definition,
                context,
                element: None,
            }) => Some(context),
            _ => None,
        }
    }

    #[test]
    fn every_floor_rule_rejects_with_its_values_and_action() -> Result<(), StructureError> {
        let circle = CulvertBarrel::circular(1.2)?;
        let pipe = CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?;
        let boxed = CulvertBarrel::rectangular_box(2.0, 1.5)?;
        let cases = [
            (
                circle.with_floor(floor(1.2, 2.5)),
                "sediment_depth_m=1.2,rise_m=1.2: the barrel is full. Declare StructureCondition::Closed",
            ),
            (circle.with_floor(floor(3.0, 2.5)), "the barrel is full"),
            (
                circle.with_floor(floor(0.0, 2.5)),
                "sediment_depth_m=0: a circular barrel has no floor. Supply a positive sediment depth or omit the floor",
            ),
            (
                pipe.with_floor(floor(-0.0, 2.5)),
                "sediment_depth_m=0: a pipe-arch has no floor",
            ),
            (
                circle.with_floor(floor(1e-300, 2.5)),
                "the sediment depth rounds to zero against the rise",
            ),
            (
                circle.with_floor(floor(f64::NAN, 2.5)),
                "sediment_depth_m=NaN: supply a finite sediment depth in metres",
            ),
            (
                circle.with_floor(floor(-0.1, 2.5)),
                "sediment_depth_m=-0.1: supply a nonnegative sediment depth in metres",
            ),
            (
                circle.with_floor(floor(0.2, 0.0)),
                "roughness_ratio=0.0: supply a finite positive bed-to-wall Manning n ratio",
            ),
            (
                boxed.with_floor(floor(0.0, f64::NAN)),
                "roughness_ratio=NaN: supply a finite positive bed-to-wall Manning n ratio",
            ),
            (
                boxed.with_floor(floor(0.0, 1e250)),
                "ratio^(3/2) times the span leaves the F64 range",
            ),
            (
                CulvertBarrel::circular(2f64.powi(-990))?
                    .with_floor(floor(2f64.powi(-990) * 0.875, 1.0)),
                "the open rise times 2^-30 is subnormal",
            ),
        ];
        for (result, expected) in cases {
            let text = message(result);
            assert!(
                text.as_deref().is_some_and(|t| t.contains(expected)),
                "{text:?} lacks {expected:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn admitted_floor_keeps_an_exact_open_rise() -> Result<(), StructureError> {
        let circle = CulvertBarrel::circular(1.2)?;
        let floored = circle.with_floor(floor(0.3, 2.5))?;
        let Some(admitted) = floored.floor() else {
            return Err(StructureError::definition("the floor was lost"));
        };
        let (bed, open) = (admitted.sediment_depth_m, floored.open_rise_m());
        assert_eq!(bed + open, 1.2);
        assert_eq!(1.2 - open, bed);
        assert!((bed - 0.3).abs() <= 0.5 * (open.next_up() - open));
        assert_eq!(floored.base(), circle);
        assert_eq!(circle.open_rise_m(), 1.2);
        // A box and an arch take their own floor at depth zero, and -0 admits as +0.
        let boxed = CulvertBarrel::rectangular_box(2.0, 1.5)?.with_floor(floor(-0.0, 2.5))?;
        let depth = boxed.floor().map(|f| f.sediment_depth_m.to_bits());
        assert_eq!(depth, Some(0));
        assert_eq!(boxed.open_rise_m(), 1.5);
        CulvertBarrel::arch(3.048, 1.6002)?.with_floor(floor(0.0, 0.5))?;
        // A new floor replaces the old one.
        let again = floored.with_floor(floor(0.6, 1.0))?;
        assert_eq!(again.floor().map(|f| f.sediment_depth_m), Some(0.6));
        assert_eq!(again.base(), circle);
        Ok(())
    }
}
