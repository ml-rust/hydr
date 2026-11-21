//! Closed-conduit barrel shapes and their admission.

use crate::structures::StructureError;

/// Cross-section shape of a culvert barrel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CulvertBarrelShape {
    /// Circular pipe. Its span and rise both equal the diameter.
    Circular,
    /// Rectangular box of span `B` and rise `D`.
    Box,
}

/// Admitted culvert barrel cross-section, with dimensions in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertBarrel {
    shape: CulvertBarrelShape,
    span_m: f64,
    rise_m: f64,
}

/// Dimensions scaled by this factor stay finite, so area, perimeter and moment never overflow.
const OVERFLOW_MARGIN: f64 = 64.0;

/// `2^-30`. The depth `D 2^-30` must stay a normal F64.
const FLOOR_FRACTION: f64 = 1.0 / 1_073_741_824.0;

impl CulvertBarrel {
    /// Admits a circular barrel of diameter `diameter_m`.
    ///
    /// Rejects a nonfinite or nonpositive diameter, a diameter whose `2^-30` fraction is subnormal,
    /// and a diameter whose cube, scaled by 64, overflows F64.
    pub fn circular(diameter_m: f64) -> Result<Self, StructureError> {
        dimension("diameter_m", "diameter", diameter_m)?;
        let fields = format!("diameter_m={diameter_m:?}");
        admit(
            CulvertBarrelShape::Circular,
            diameter_m,
            diameter_m,
            &fields,
        )
    }

    /// Admits a box barrel of span `span_m` and rise `rise_m`.
    ///
    /// Rejects a nonfinite or nonpositive span or rise, a rise whose `2^-30` fraction is
    /// subnormal, and dimensions whose `B D`, `2 (B + D)` or `B D^2`, scaled by 64, overflow F64.
    pub fn rectangular_box(span_m: f64, rise_m: f64) -> Result<Self, StructureError> {
        dimension("span_m", "span", span_m)?;
        dimension("rise_m", "rise", rise_m)?;
        let fields = format!("span_m={span_m:?},rise_m={rise_m:?}");
        admit(CulvertBarrelShape::Box, span_m, rise_m, &fields)
    }

    /// Barrel shape.
    pub fn shape(&self) -> CulvertBarrelShape {
        self.shape
    }

    /// Horizontal width `B` in metres. A circle reports its diameter.
    pub fn span_m(&self) -> f64 {
        self.span_m
    }

    /// Vertical height `D` from invert to crown in metres. A circle reports its diameter.
    pub fn rise_m(&self) -> f64 {
        self.rise_m
    }
}

fn dimension(field: &str, what: &str, value: f64) -> Result<(), StructureError> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(StructureError::definition(format!(
        "{field}={value:?}: supply a finite positive barrel {what} in metres"
    )))
}

fn admit(
    shape: CulvertBarrelShape,
    span_m: f64,
    rise_m: f64,
    fields: &str,
) -> Result<CulvertBarrel, StructureError> {
    // 2^-992 = MIN_POSITIVE 2^30 is exact. Below it, D 2^-30 is subnormal or rounds.
    let smallest = f64::MIN_POSITIVE / FLOOR_FRACTION;
    if rise_m < smallest {
        return Err(StructureError::definition(format!(
            "{fields}: rise times 2^-30 is subnormal. Supply a rise of at least {smallest:?} m"
        )));
    }
    let scales = [
        ("B D", span_m * rise_m),
        ("2 (B + D)", 2.0 * (span_m + rise_m)),
        ("B D^2", span_m * rise_m * rise_m),
    ];
    for (name, scale) in scales {
        if !(scale * OVERFLOW_MARGIN).is_finite() {
            return Err(StructureError::definition(format!(
                "{fields}: {name} times 64 overflows F64. Supply smaller barrel dimensions in metres"
            )));
        }
    }
    Ok(CulvertBarrel {
        shape,
        span_m,
        rise_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;

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
    fn admission_names_the_field_value_and_action() {
        let cases = [
            (
                CulvertBarrel::rectangular_box(1.0, -0.5),
                "rise_m=-0.5: supply a finite positive barrel rise in metres",
            ),
            (
                CulvertBarrel::rectangular_box(f64::NAN, 1.0),
                "span_m=NaN: supply a finite positive barrel span in metres",
            ),
            (
                CulvertBarrel::circular(0.0),
                "diameter_m=0.0: supply a finite positive barrel diameter in metres",
            ),
            (
                CulvertBarrel::circular(f64::INFINITY),
                "diameter_m=inf: supply a finite positive barrel diameter in metres",
            ),
            (
                CulvertBarrel::circular(1e-300),
                "diameter_m=1e-300: rise times 2^-30 is subnormal",
            ),
            (
                CulvertBarrel::rectangular_box(1e300, 1e10),
                "span_m=1e300,rise_m=10000000000.0: B D times 64 overflows F64",
            ),
            (
                CulvertBarrel::circular(1e103),
                "diameter_m=1e103: B D^2 times 64 overflows F64",
            ),
        ];
        for (result, expected) in cases {
            let text = message(result);
            assert!(
                text.as_deref().is_some_and(|t| t.contains(expected)),
                "{text:?} lacks {expected:?}"
            );
        }
    }

    #[test]
    fn admitted_barrels_report_their_dimensions() -> Result<(), StructureError> {
        let circle = CulvertBarrel::circular(0.3)?;
        assert_eq!(circle.shape(), CulvertBarrelShape::Circular);
        assert_eq!((circle.span_m(), circle.rise_m()), (0.3, 0.3));
        let rectangle = CulvertBarrel::rectangular_box(2.5, 0.8)?;
        assert_eq!(rectangle.shape(), CulvertBarrelShape::Box);
        assert_eq!((rectangle.span_m(), rectangle.rise_m()), (2.5, 0.8));
        // The smallest admitted rise keeps D 2^-30 normal.
        let smallest = f64::MIN_POSITIVE / FLOOR_FRACTION;
        assert!(CulvertBarrel::circular(smallest).is_ok());
        assert!(CulvertBarrel::circular(smallest.next_down()).is_err());
        Ok(())
    }
}
