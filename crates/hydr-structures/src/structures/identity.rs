//! Exact original physical bits for complete structure-definition compatibility.

use super::{CulvertEnd, CulvertParameters, PumpSpeedPolicy, StructureElement, StructureLaw};

#[doc(hidden)]
#[inline]
pub fn same_elements(a: &[StructureElement], b: &[StructureElement]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.id == b.id
                && a.endpoint_a == b.endpoint_a
                && a.endpoint_b == b.endpoint_b
                && a.datum == b.datum
                && a.direction == b.direction
                && same_law(&a.law, &b.law)
        })
}
fn same_law(a: &StructureLaw, b: &StructureLaw) -> bool {
    match (a, b) {
        (StructureLaw::SubmergedOrifice(a), StructureLaw::SubmergedOrifice(b)) => {
            a.applicability() == b.applicability()
                && bits(
                    &[
                        a.area_m2(),
                        a.sill_m(),
                        a.crown_m(),
                        a.discharge_coefficient(),
                        a.gravity_m_per_s2(),
                    ],
                    &[
                        b.area_m2(),
                        b.sill_m(),
                        b.crown_m(),
                        b.discharge_coefficient(),
                        b.gravity_m_per_s2(),
                    ],
                )
        }
        (
            StructureLaw::SuppressedRectangularWeir(a),
            StructureLaw::SuppressedRectangularWeir(b),
        ) => {
            a.applicability() == b.applicability()
                && bits(
                    &[a.crest_length_m(), a.crest_m(), a.discharge_coefficient()],
                    &[b.crest_length_m(), b.crest_m(), b.discharge_coefficient()],
                )
        }
        (
            StructureLaw::HydrostaticRectangularGate(a),
            StructureLaw::HydrostaticRectangularGate(b),
        ) => {
            a.applicability() == b.applicability()
                && bits(
                    &[
                        a.width_m(),
                        a.sill_m(),
                        a.maximum_opening_m(),
                        a.discharge_coefficient(),
                        a.gravity_m_per_s2(),
                    ],
                    &[
                        b.width_m(),
                        b.sill_m(),
                        b.maximum_opening_m(),
                        b.discharge_coefficient(),
                        b.gravity_m_per_s2(),
                    ],
                )
        }
        (StructureLaw::Culvert(a), StructureLaw::Culvert(b)) => same_culvert(a, b),
        (StructureLaw::Pump(a), StructureLaw::Pump(b)) => {
            a.applicability() == b.applicability()
                && a.minimum_suction_stage_m().to_bits() == b.minimum_suction_stage_m().to_bits()
                && a.curve().key() == b.curve().key()
                && match (a.free_outlet(), b.free_outlet()) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        a.evidence == b.evidence && bits(&[a.elevation_m], &[b.elevation_m])
                    }
                    _ => false,
                }
                && match (a.speed_policy(), b.speed_policy()) {
                    (PumpSpeedPolicy::ReferenceOnly, PumpSpeedPolicy::ReferenceOnly) => true,
                    (
                        PumpSpeedPolicy::DeclaredAffinity {
                            minimum_ratio: a0,
                            maximum_ratio: a1,
                            applicability: ae,
                        },
                        PumpSpeedPolicy::DeclaredAffinity {
                            minimum_ratio: b0,
                            maximum_ratio: b1,
                            applicability: be,
                        },
                    ) => ae == be && bits(&[*a0, *a1], &[*b0, *b1]),
                    _ => false,
                }
        }
        _ => false,
    }
}
fn same_culvert(a: &CulvertParameters, b: &CulvertParameters) -> bool {
    let values = |p: &CulvertParameters| {
        let barrel = p.barrel();
        let alignment = p.alignment();
        [
            barrel.span_m(),
            barrel.rise_m(),
            alignment.length_m,
            alignment.invert_a_m,
            alignment.invert_b_m,
            p.manning_n(),
            p.gravity_m_per_s2(),
        ]
    };
    let same_end = |a: Option<CulvertEnd>, b: Option<CulvertEnd>| match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let (ia, ib) = (a.inlet(), b.inlet());
            ia.code() == ib.code()
                && ia.form() == ib.form()
                && ia.mitered() == ib.mitered()
                && bits(&ia.coefficients(), &ib.coefficients())
                && bits(&[a.entrance_loss()], &[b.entrance_loss()])
        }
        _ => false,
    };
    a.applicability() == b.applicability()
        && a.barrel().shape() == b.barrel().shape()
        && a.barrel_count() == b.barrel_count()
        && bits(&values(a), &values(b))
        && same_end(a.ends().a, b.ends().a)
        && same_end(a.ends().b, b.ends().b)
}
#[doc(hidden)]
#[inline]
pub fn bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[cfg(test)]
mod tests {
    use super::same_elements;
    use crate::structures::interval_fixture::{TestResult, pump};
    use crate::structures::{PumpFreeOutlet, StructureElement, StructureLaw};

    fn with_outlet(elevation_m: f64, evidence: &str) -> TestResult<StructureElement> {
        let mut element = pump("p")?;
        let StructureLaw::Pump(parameters) = element.law else {
            return Err("fixture law is not a pump".into());
        };
        element.law = StructureLaw::Pump(parameters.with_free_outlet(PumpFreeOutlet {
            elevation_m,
            evidence: evidence.into(),
        })?);
        Ok(element)
    }

    fn culvert(ke: f64, count: u32, code_b: Option<u8>) -> TestResult<StructureElement> {
        use crate::structures::{
            CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds,
            CulvertInlet, CulvertParameters, DirectionPolicy,
        };
        let end_b = match code_b {
            Some(code) => Some(CulvertEnd::new(CulvertInlet::hds5(code)?, 0.5)?),
            None => None,
        };
        Ok(StructureElement {
            id: "c".into(),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: "datum".into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::Culvert(CulvertParameters::new(
                CulvertBarrel::circular(1.0)?,
                CulvertAlignment {
                    length_m: 20.0,
                    invert_a_m: 0.1,
                    invert_b_m: 0.0,
                },
                0.013,
                count,
                CulvertEnds {
                    a: Some(CulvertEnd::new(CulvertInlet::hds5(1)?, ke)?),
                    b: end_b,
                },
                9.81,
                CulvertApplicability {
                    coefficient_provenance: "HDS-5".into(),
                    barrel_condition_assertion: "Clean".into(),
                },
            )?),
        })
    }

    #[test]
    fn entrance_loss_bits_count_and_ends_distinguish_culverts() -> TestResult {
        let base = culvert(0.5, 1, Some(3))?;
        assert!(same_elements(
            std::slice::from_ref(&base),
            &[culvert(0.5, 1, Some(3))?]
        ));
        for other in [
            culvert(0.5_f64.next_up(), 1, Some(3))?,
            culvert(0.5, 2, Some(3))?,
            culvert(0.5, 1, Some(2))?,
            culvert(0.5, 1, None)?,
        ] {
            assert!(!same_elements(
                std::slice::from_ref(&base),
                std::slice::from_ref(&other)
            ));
        }
        Ok(())
    }

    #[test]
    fn free_outlet_presence_elevation_bits_and_evidence_distinguish_pumps() -> TestResult {
        let plain = pump("p")?;
        let outlet = with_outlet(1.0, "Free outlet")?;
        assert!(same_elements(
            std::slice::from_ref(&plain),
            std::slice::from_ref(&plain)
        ));
        assert!(same_elements(
            std::slice::from_ref(&outlet),
            &[with_outlet(1.0, "Free outlet")?]
        ));
        for other in [
            plain,
            with_outlet(1.0_f64.next_up(), "Free outlet")?,
            with_outlet(1.0, "Other outlet")?,
        ] {
            assert!(!same_elements(
                std::slice::from_ref(&outlet),
                std::slice::from_ref(&other)
            ));
            assert!(!same_elements(
                std::slice::from_ref(&other),
                std::slice::from_ref(&outlet)
            ));
        }
        Ok(())
    }
}
