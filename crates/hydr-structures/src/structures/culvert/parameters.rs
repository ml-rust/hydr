//! Culvert law definition: barrel, alignment, roughness, barrel count and one end per direction.

use crate::structures::{
    CulvertApplicability, CulvertBarrel, CulvertInlet, CulvertInletRow, CulvertOutletRow,
    StructureError,
};

/// Inlet of the culvert end that flow enters: an HDS-5 inlet and its entrance loss `ke`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertEnd {
    inlet: CulvertInlet,
    entrance_loss: f64,
}

impl CulvertEnd {
    /// Admits an inlet and a finite nonnegative entrance loss coefficient `ke`.
    pub fn new(inlet: CulvertInlet, entrance_loss: f64) -> Result<Self, StructureError> {
        if !(entrance_loss.is_finite() && entrance_loss >= 0.0) {
            return Err(StructureError::definition(format!(
                "{},entrance_loss={entrance_loss:?}: supply a finite nonnegative entrance loss coefficient",
                inlet.describe()
            )));
        }
        Ok(Self {
            inlet,
            entrance_loss,
        })
    }

    /// HDS-5 inlet of this end.
    pub fn inlet(&self) -> CulvertInlet {
        self.inlet
    }

    /// Entrance loss coefficient `ke`.
    pub fn entrance_loss(&self) -> f64 {
        self.entrance_loss
    }
}

/// Barrel length and the invert elevation at each endpoint, in metres on the structure datum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertAlignment {
    /// Barrel length `L` in metres.
    pub length_m: f64,
    /// Invert elevation at endpoint A.
    pub invert_a_m: f64,
    /// Invert elevation at endpoint B.
    pub invert_b_m: f64,
}

/// One culvert end per admitted flow direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertEnds {
    /// Inlet at endpoint A, entered by A-to-B flow.
    pub a: Option<CulvertEnd>,
    /// Inlet at endpoint B, entered by B-to-A flow.
    pub b: Option<CulvertEnd>,
}

/// Complete checked HDS-5 culvert: identical parallel barrels between two endpoints.
///
/// The law holds no water. A roadway over the culvert is a separate weir on the same endpoints.
#[derive(Clone, Debug, PartialEq)]
pub struct CulvertParameters {
    barrel: CulvertBarrel,
    alignment: CulvertAlignment,
    manning_n: f64,
    barrel_count: u32,
    ends: CulvertEnds,
    gravity_m_per_s2: f64,
    applicability: CulvertApplicability,
}

impl CulvertParameters {
    /// Checks the alignment, roughness, barrel count, ends, gravity and applicability evidence.
    ///
    /// Rejects a nonfinite or nonpositive length, Manning `n` or gravity, a nonfinite invert, a
    /// zero barrel count, and a definition without an end. Rejects an end whose coded inlet is
    /// for another barrel shape. Rejects an end whose barrel rises toward the far invert by more
    /// than half the rise. Monotone inlet admission runs when a batch uploads the culvert.
    pub fn new(
        barrel: CulvertBarrel,
        alignment: CulvertAlignment,
        manning_n: f64,
        barrel_count: u32,
        ends: CulvertEnds,
        gravity_m_per_s2: f64,
        applicability: CulvertApplicability,
    ) -> Result<Self, StructureError> {
        applicability.check()?;
        let CulvertAlignment {
            length_m,
            invert_a_m,
            invert_b_m,
        } = alignment;
        let fields = format!(
            "length_m={length_m:?},invert_a_m={invert_a_m:?},invert_b_m={invert_b_m:?},manning_n={manning_n:?},barrel_count={barrel_count},gravity_m_per_s2={gravity_m_per_s2:?}"
        );
        if !(invert_a_m.is_finite() && invert_b_m.is_finite()) {
            return Err(StructureError::definition(format!(
                "{fields}: supply finite invert elevations on the structure datum"
            )));
        }
        if barrel_count == 0 {
            return Err(StructureError::definition(format!(
                "{fields}: supply at least one barrel"
            )));
        }
        if ends.a.is_none() && ends.b.is_none() {
            return Err(StructureError::definition(format!(
                "{fields}: supply the culvert end of at least one flow direction"
            )));
        }
        let parameters = Self {
            barrel,
            alignment,
            manning_n,
            barrel_count,
            ends,
            gravity_m_per_s2,
            applicability,
        };
        for side in [CulvertSide::A, CulvertSide::B] {
            if let Some((inlet, outlet)) = parameters.rows(side) {
                outlet.check()?;
                inlet.check()?;
            }
        }
        Ok(parameters)
    }

    /// Barrel cross-section of every barrel.
    pub fn barrel(&self) -> CulvertBarrel {
        self.barrel
    }

    /// Barrel length and both inverts.
    pub fn alignment(&self) -> CulvertAlignment {
        self.alignment
    }

    /// Manning roughness `n` in `s/m^(1/3)`.
    pub fn manning_n(&self) -> f64 {
        self.manning_n
    }

    /// Number of identical parallel barrels.
    pub fn barrel_count(&self) -> u32 {
        self.barrel_count
    }

    /// End of each flow direction.
    pub fn ends(&self) -> CulvertEnds {
        self.ends
    }

    /// Gravity in metres per second squared.
    pub fn gravity_m_per_s2(&self) -> f64 {
        self.gravity_m_per_s2
    }

    /// Complete immutable applicability evidence.
    pub fn applicability(&self) -> &CulvertApplicability {
        &self.applicability
    }

    /// Inlet and outlet query rows of A-to-B flow, or `None` without an A end.
    ///
    /// The slope is `(z_a - z_b) / L`.
    pub fn forward_rows(&self) -> Option<(CulvertInletRow, CulvertOutletRow)> {
        self.rows(CulvertSide::A)
    }

    /// Inlet and outlet query rows of B-to-A flow, or `None` without a B end.
    ///
    /// The slope is `(z_b - z_a) / L`.
    pub fn reverse_rows(&self) -> Option<(CulvertInletRow, CulvertOutletRow)> {
        self.rows(CulvertSide::B)
    }

    /// Inlet and outlet rows of flow that enters at `side`, or `None` without that end.
    ///
    /// The slope is `(z_in - z_out) / L`, so the two directions carry opposite slopes.
    pub(crate) fn rows(&self, side: CulvertSide) -> Option<(CulvertInletRow, CulvertOutletRow)> {
        let CulvertAlignment {
            length_m,
            invert_a_m,
            invert_b_m,
        } = self.alignment;
        let (end, fall) = match side {
            CulvertSide::A => (self.ends.a?, invert_a_m - invert_b_m),
            CulvertSide::B => (self.ends.b?, invert_b_m - invert_a_m),
        };
        let slope = fall / length_m;
        Some((
            CulvertInletRow {
                barrel: self.barrel,
                inlet: end.inlet,
                slope,
                gravity_m_per_s2: self.gravity_m_per_s2,
            },
            CulvertOutletRow {
                barrel: self.barrel,
                length_m,
                manning_n: self.manning_n,
                entrance_loss: end.entrance_loss,
                slope,
                gravity_m_per_s2: self.gravity_m_per_s2,
            },
        ))
    }

    /// Invert at `side`.
    pub(crate) fn invert(&self, side: CulvertSide) -> f64 {
        match side {
            CulvertSide::A => self.alignment.invert_a_m,
            CulvertSide::B => self.alignment.invert_b_m,
        }
    }
}

/// Endpoint of a culvert, which is the inlet of the flow that enters there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CulvertSide {
    A,
    B,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;

    fn applicability() -> CulvertApplicability {
        CulvertApplicability {
            coefficient_provenance: "HDS-5 Table A.1".into(),
            barrel_condition_assertion: "Clean barrel".into(),
        }
    }

    fn end(code: u8) -> Result<CulvertEnd, StructureError> {
        CulvertEnd::new(CulvertInlet::hds5(code)?, 0.5)
    }

    fn culvert(
        alignment: CulvertAlignment,
        manning_n: f64,
        count: u32,
        ends: CulvertEnds,
    ) -> Result<CulvertParameters, StructureError> {
        CulvertParameters::new(
            CulvertBarrel::circular(1.2)?,
            alignment,
            manning_n,
            count,
            ends,
            9.81,
            applicability(),
        )
    }

    fn message(result: Result<CulvertParameters, StructureError>) -> Option<String> {
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
    fn rows_carry_opposite_slopes_and_their_own_end() -> Result<(), StructureError> {
        let alignment = CulvertAlignment {
            length_m: 40.0,
            invert_a_m: 10.3,
            invert_b_m: 10.1,
        };
        let parameters = culvert(
            alignment,
            0.012,
            2,
            CulvertEnds {
                a: Some(end(1)?),
                b: Some(CulvertEnd::new(CulvertInlet::hds5(3)?, 0.2)?),
            },
        )?;
        let forward = parameters.rows(CulvertSide::A);
        let reverse = parameters.rows(CulvertSide::B);
        let (Some((inlet_a, outlet_a)), Some((inlet_b, outlet_b))) = (forward, reverse) else {
            return Err(StructureError::definition("both ends exist"));
        };
        assert_eq!(inlet_a.slope.to_bits(), (-inlet_b.slope).to_bits());
        assert_eq!(outlet_a.slope.to_bits(), inlet_a.slope.to_bits());
        assert_eq!(inlet_a.inlet.code(), Some(1));
        assert_eq!(inlet_b.inlet.code(), Some(3));
        assert_eq!((outlet_a.entrance_loss, outlet_b.entrance_loss), (0.5, 0.2));
        assert_eq!(parameters.invert(CulvertSide::B), 10.1);
        Ok(())
    }

    #[test]
    fn admission_names_the_value_and_action() -> Result<(), StructureError> {
        let level = CulvertAlignment {
            length_m: 20.0,
            invert_a_m: 0.0,
            invert_b_m: 0.0,
        };
        let one = CulvertEnds {
            a: Some(end(1)?),
            b: None,
        };
        let cases = [
            (
                culvert(level, 0.013, 0, one),
                "barrel_count=0,gravity_m_per_s2=9.81: supply at least one barrel",
            ),
            (
                culvert(level, 0.013, 1, CulvertEnds { a: None, b: None }),
                "supply the culvert end of at least one flow direction",
            ),
            (
                culvert(
                    CulvertAlignment {
                        invert_b_m: f64::NAN,
                        ..level
                    },
                    0.013,
                    1,
                    one,
                ),
                "invert_b_m=NaN",
            ),
            (
                culvert(level, -0.013, 1, one),
                "manning_n=-0.013,entrance_loss=0.5,slope=0.0,gravity_m_per_s2=9.81: supply a finite positive Manning n",
            ),
            (
                culvert(
                    CulvertAlignment {
                        invert_b_m: 0.7,
                        ..level
                    },
                    0.013,
                    1,
                    one,
                ),
                "Supply an adverse fall of at most rise/2",
            ),
            (
                culvert(
                    level,
                    0.013,
                    1,
                    CulvertEnds {
                        a: Some(end(9)?),
                        b: None,
                    },
                ),
                "the code is for Box barrels",
            ),
        ];
        for (result, expected) in cases {
            let text = message(result);
            assert!(
                text.as_deref().is_some_and(|t| t.contains(expected)),
                "{text:?} lacks {expected:?}"
            );
        }
        let text = match CulvertEnd::new(CulvertInlet::hds5(1)?, -0.1) {
            Err(StructureError::InvalidInput { context, .. }) => context,
            Ok(_) => String::new(),
            Err(other) => other.to_string(),
        };
        assert!(text.contains("entrance_loss=-0.1: supply a finite nonnegative"));
        Ok(())
    }
}
