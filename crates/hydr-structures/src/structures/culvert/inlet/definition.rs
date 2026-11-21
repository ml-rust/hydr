//! HDS-5 inlet-control definitions: equation form, coefficients and the mitered flag.

use super::codes::CODES;
use crate::structures::{CulvertBarrelShape, StructureError};

/// Unsubmerged HDS-5 inlet-control equation, with `X = Ku Q / (A D^0.5)` in foot units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CulvertInletForm {
    /// `HW/D = Hc/D + K X^M + Ks S`, with critical head `Hc = dc + A(dc) / (2 T(dc))`.
    One,
    /// `HW/D = K X^M`. The form has no slope term.
    Two,
}

/// Admitted HDS-5 inlet: form, coefficients `K`, `M`, `c`, `Y` and the mitered flag.
///
/// The submerged equation is `HW/D = c X^2 + Y + Ks S`. `Ks` is `-0.5`, or `+0.7` for an inlet
/// mitered to the slope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertInlet {
    code: Option<u8>,
    shape: Option<CulvertBarrelShape>,
    form: CulvertInletForm,
    k: f64,
    m: f64,
    c: f64,
    y: f64,
    mitered: bool,
}

impl CulvertInlet {
    /// Admits the HDS-5 Table A.1 inlet `code`, numbered as in SWMM `culvert.c`.
    ///
    /// Codes 1 to 8 are circular and 9 to 26 concrete box inlets. Code 5 is mitered.
    /// Rejects every other code.
    pub fn hds5(code: u8) -> Result<Self, StructureError> {
        let entry = usize::from(code)
            .checked_sub(1)
            .and_then(|index| CODES.get(index))
            .ok_or_else(|| {
                StructureError::definition(format!(
                    "inlet code={code}: supply an HDS-5 code from 1 to 26 for a circular or box barrel, or explicit coefficients"
                ))
            })?;
        Ok(Self {
            code: Some(code),
            shape: Some(entry.shape),
            form: entry.form,
            k: entry.k,
            m: entry.m,
            c: entry.c,
            y: entry.y,
            mitered: entry.mitered,
        })
    }

    /// Admits explicit coefficients for any barrel shape.
    ///
    /// Rejects a nonfinite or nonpositive `K`, `M` or `c`, and a nonfinite `Y`. Each of these
    /// keeps the headwater rising within its band.
    pub fn explicit(
        form: CulvertInletForm,
        [k, m, c, y]: [f64; 4],
        mitered: bool,
    ) -> Result<Self, StructureError> {
        let fields = format!("form={form:?},k={k:?},m={m:?},c={c:?},y={y:?}");
        for (name, value) in [("K", k), ("M", m), ("c", c)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(StructureError::definition(format!(
                    "{fields}: supply a finite positive inlet coefficient {name}"
                )));
            }
        }
        if !y.is_finite() {
            return Err(StructureError::definition(format!(
                "{fields}: supply a finite inlet coefficient Y"
            )));
        }
        Ok(Self {
            code: None,
            shape: None,
            form,
            k,
            m,
            c,
            y,
            mitered,
        })
    }

    /// HDS-5 code, or `None` for explicit coefficients.
    pub fn code(&self) -> Option<u8> {
        self.code
    }

    /// Barrel shape a coded inlet requires, or `None` for explicit coefficients.
    pub fn barrel_shape(&self) -> Option<CulvertBarrelShape> {
        self.shape
    }

    /// Unsubmerged equation form.
    pub fn form(&self) -> CulvertInletForm {
        self.form
    }

    /// Coefficients `[K, M, c, Y]`.
    pub fn coefficients(&self) -> [f64; 4] {
        [self.k, self.m, self.c, self.y]
    }

    /// Whether the inlet is mitered to the slope, so `Ks = +0.7`.
    pub fn mitered(&self) -> bool {
        self.mitered
    }

    /// Names the inlet in rejection messages.
    pub(crate) fn describe(&self) -> String {
        let coefficients = format!(
            "form={:?},k={:?},m={:?},c={:?},y={:?},mitered={}",
            self.form, self.k, self.m, self.c, self.y, self.mitered
        );
        match self.code {
            Some(code) => format!("inlet code={code},{coefficients}"),
            None => format!("explicit inlet {coefficients}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;

    fn message(result: Result<CulvertInlet, StructureError>) -> Option<String> {
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
    fn codes_carry_their_shape_form_and_mitered_flag() -> Result<(), StructureError> {
        let mitered = CulvertInlet::hds5(5)?;
        assert!(mitered.mitered());
        assert_eq!(mitered.barrel_shape(), Some(CulvertBarrelShape::Circular));
        assert_eq!(mitered.coefficients(), [0.0210, 1.33, 0.0463, 0.75]);
        let boxed = CulvertInlet::hds5(26)?;
        assert_eq!(boxed.form(), CulvertInletForm::Two);
        assert_eq!(boxed.barrel_shape(), Some(CulvertBarrelShape::Box));
        assert_eq!(boxed.code(), Some(26));
        let unmitered = (1..=26).filter(|&code| code != 5);
        for code in unmitered {
            assert!(!CulvertInlet::hds5(code)?.mitered(), "code {code}");
        }
        Ok(())
    }

    #[test]
    fn admission_names_the_value_and_action() {
        let cases = [
            (
                CulvertInlet::hds5(0),
                "inlet code=0: supply an HDS-5 code from 1 to 26",
            ),
            (
                CulvertInlet::hds5(27),
                "inlet code=27: supply an HDS-5 code from 1 to 26",
            ),
            (
                CulvertInlet::explicit(CulvertInletForm::One, [0.0, 2.0, 0.04, 0.7], false),
                "k=0.0,m=2.0,c=0.04,y=0.7: supply a finite positive inlet coefficient K",
            ),
            (
                CulvertInlet::explicit(CulvertInletForm::Two, [0.5, f64::NAN, 0.04, 0.7], false),
                "m=NaN,c=0.04,y=0.7: supply a finite positive inlet coefficient M",
            ),
            (
                CulvertInlet::explicit(CulvertInletForm::Two, [0.5, 0.6, -0.04, 0.7], true),
                "supply a finite positive inlet coefficient c",
            ),
            (
                CulvertInlet::explicit(
                    CulvertInletForm::One,
                    [0.5, 0.6, 0.04, f64::INFINITY],
                    false,
                ),
                "y=inf: supply a finite inlet coefficient Y",
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
}
