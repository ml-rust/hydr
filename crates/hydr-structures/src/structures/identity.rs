//! Exact original physical bits for complete structure-definition compatibility.

use super::{
    BlockageGeometry, CulvertEnd, CulvertParameters, PumpSpeedPolicy, StructureCondition,
    StructureElement, StructureLaw,
};

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
                && same_condition(&a.condition, &b.condition)
        })
}
fn same_condition(a: &StructureCondition, b: &StructureCondition) -> bool {
    match (a, b) {
        (StructureCondition::AsBuilt, StructureCondition::AsBuilt) => true,
        (
            StructureCondition::Closed { evidence: a },
            StructureCondition::Closed { evidence: b },
        ) => a == b,
        (StructureCondition::Blocked(a), StructureCondition::Blocked(b)) => {
            a.applicability_evidence() == b.applicability_evidence()
                && same_blockage_geometry(a.geometry(), b.geometry())
        }
        _ => false,
    }
}
fn same_blockage_geometry(a: &BlockageGeometry, b: &BlockageGeometry) -> bool {
    match (a, b) {
        (
            BlockageGeometry::Weir { crest_length: a },
            BlockageGeometry::Weir { crest_length: b },
        ) => bits(&[a.blocked_share()], &[b.blocked_share()]),
        (BlockageGeometry::Orifice { area: a }, BlockageGeometry::Orifice { area: b }) => {
            bits(&[a.blocked_share()], &[b.blocked_share()])
        }
        (
            BlockageGeometry::Gate {
                width: aw,
                opening_range: ao,
            },
            BlockageGeometry::Gate {
                width: bw,
                opening_range: bo,
            },
        ) => bits(
            &[aw.blocked_share(), ao.blocked_share()],
            &[bw.blocked_share(), bo.blocked_share()],
        ),
        (BlockageGeometry::Culvert(a), BlockageGeometry::Culvert(b)) => {
            a.closed_barrels == b.closed_barrels
                && bits(
                    &[a.face_a.blocked_share(), a.face_b.blocked_share()],
                    &[b.face_a.blocked_share(), b.face_b.blocked_share()],
                )
                && match (a.sediment, b.sediment) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        bits(&[a.depth_m, a.bed_manning_n], &[b.depth_m, b.bed_manning_n])
                    }
                    _ => false,
                }
        }
        _ => false,
    }
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
    let radii = |p: &CulvertParameters| p.barrel().given_radius_m().unwrap_or(&[]).to_vec();
    let same_floor = match (a.floor_manning_n(), b.floor_manning_n()) {
        (None, None) => true,
        (Some(a), Some(b)) => bits(&[a], &[b]),
        _ => false,
    };
    a.applicability() == b.applicability()
        && a.barrel().shape() == b.barrel().shape()
        && a.barrel_count() == b.barrel_count()
        && same_floor
        && bits(&values(a), &values(b))
        && bits(&radii(a), &radii(b))
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

    fn blocked_gate(share: f64, evidence: &str) -> TestResult<StructureElement> {
        use crate::structures::{BlockageFraction, StructureBlockage, StructureCondition};
        let mut element = crate::structures::interval_fixture::gate("g")?;
        element.condition = StructureCondition::Blocked(StructureBlockage::gate(
            BlockageFraction::new(share)?,
            BlockageFraction::NONE,
            evidence.into(),
        )?);
        Ok(element)
    }

    #[test]
    fn one_ulp_of_a_fraction_or_one_evidence_character_distinguishes_conditions() -> TestResult {
        use crate::structures::StructureCondition;
        let same = |a: &StructureElement, b: &StructureElement| {
            same_elements(std::slice::from_ref(a), std::slice::from_ref(b))
        };
        let base = blocked_gate(0.25, "Survey")?;
        assert!(same(&base, &blocked_gate(0.25, "Survey")?));
        assert!(!same(&base, &blocked_gate(0.25_f64.next_up(), "Survey")?));
        assert!(!same(&base, &blocked_gate(0.25, "Survey.")?));
        assert!(same(
            &blocked_gate(-0.0, "Survey")?,
            &blocked_gate(0.0, "Survey")?
        ));
        let mut built = base.clone();
        built.condition = StructureCondition::AsBuilt;
        assert!(!same(&base, &built));
        let closed = |evidence: &str| {
            let mut element = built.clone();
            element.condition = StructureCondition::Closed {
                evidence: evidence.into(),
            };
            element
        };
        assert!(same(&closed("Record"), &closed("Record")));
        assert!(!same(&closed("Record"), &closed("Record.")));
        assert!(!same(&closed("Record"), &built));
        Ok(())
    }

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
            condition: crate::structures::StructureCondition::AsBuilt,
        })
    }

    /// A one-end culvert on `barrel` with explicit form-1 inlet coefficients.
    fn arc_culvert(barrel: crate::structures::CulvertBarrel) -> TestResult<StructureElement> {
        use crate::structures::{
            CulvertAlignment, CulvertApplicability, CulvertEnd, CulvertEnds, CulvertInlet,
            CulvertInletForm, CulvertParameters, DirectionPolicy,
        };
        let inlet =
            CulvertInlet::explicit(CulvertInletForm::One, [0.0083, 2.0, 0.0379, 0.69], false)?;
        Ok(StructureElement {
            id: "c".into(),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: "datum".into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::Culvert(CulvertParameters::new(
                barrel,
                CulvertAlignment {
                    length_m: 20.0,
                    invert_a_m: 0.1,
                    invert_b_m: 0.0,
                },
                0.024,
                1,
                CulvertEnds {
                    a: Some(CulvertEnd::new(inlet, 0.5)?),
                    b: None,
                },
                9.81,
                CulvertApplicability {
                    coefficient_provenance: "HDS-5".into(),
                    barrel_condition_assertion: "Clean".into(),
                },
            )?),
            condition: crate::structures::StructureCondition::AsBuilt,
        })
    }

    #[test]
    fn one_ulp_of_a_given_radius_distinguishes_culverts() -> TestResult {
        use crate::structures::CulvertBarrel;
        let pipe =
            |corner: f64, bottom: f64| CulvertBarrel::pipe_arch(1.8542, 1.397, corner, bottom);
        let base = arc_culvert(pipe(0.4572, 1.938_528)?)?;
        assert!(same_elements(
            std::slice::from_ref(&base),
            &[arc_culvert(pipe(0.4572, 1.938_528)?)?]
        ));
        let ellipse = |side: f64| CulvertBarrel::horizontal_ellipse(1.524, 0.9652, side);
        let wide = arc_culvert(ellipse(0.419_475_92)?)?;
        for (reference, other) in [
            (&base, arc_culvert(pipe(0.4572_f64.next_up(), 1.938_528)?)?),
            (
                &base,
                arc_culvert(pipe(0.4572, 1.938_528_f64.next_down())?)?,
            ),
            (&wide, arc_culvert(ellipse(0.419_475_92_f64.next_up())?)?),
            (
                &wide,
                arc_culvert(CulvertBarrel::vertical_ellipse(1.524, 1.6, 0.419_475_92)?)?,
            ),
        ] {
            assert!(!same_elements(
                std::slice::from_ref(reference),
                std::slice::from_ref(&other)
            ));
        }
        Ok(())
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
    fn culvert_blockage_and_floor_manning_bits_distinguish_elements() -> TestResult {
        use crate::structures::{
            BlockageFraction, CulvertBarrel, CulvertBlockage, CulvertSediment, StructureBlockage,
            StructureCondition,
        };
        let boxed = CulvertBarrel::rectangular_box(2.0, 1.5)?;
        let make = |floor_n: Option<f64>,
                    blockage: CulvertBlockage,
                    ev: &str|
         -> TestResult<StructureElement> {
            let mut el = arc_culvert(boxed)?;
            if let StructureLaw::Culvert(p) = el.law {
                el.law = StructureLaw::Culvert(match floor_n {
                    Some(n) => p.with_floor_manning_n(n)?,
                    None => p,
                });
            }
            el.condition =
                StructureCondition::Blocked(StructureBlockage::culvert(blockage, ev.into())?);
            Ok(el)
        };
        let sed = CulvertSediment {
            depth_m: 0.3,
            bed_manning_n: 0.03,
        };
        let b0 = CulvertBlockage {
            face_a: BlockageFraction::new(0.25)?,
            face_b: BlockageFraction::NONE,
            sediment: Some(sed),
            closed_barrels: 1,
        };
        let base = make(Some(0.03), b0, "Survey")?;
        assert!(same_elements(
            std::slice::from_ref(&base),
            &[make(Some(0.03), b0, "Survey")?]
        ));
        let others = [
            make(None, b0, "Survey")?,
            make(Some(0.03_f64.next_up()), b0, "Survey")?,
            make(
                Some(0.03),
                CulvertBlockage {
                    face_a: BlockageFraction::new(0.25_f64.next_up())?,
                    ..b0
                },
                "Survey",
            )?,
            make(
                Some(0.03),
                CulvertBlockage {
                    sediment: Some(CulvertSediment {
                        depth_m: 0.3_f64.next_up(),
                        ..sed
                    }),
                    ..b0
                },
                "Survey",
            )?,
            make(
                Some(0.03),
                CulvertBlockage {
                    sediment: Some(CulvertSediment {
                        bed_manning_n: 0.03_f64.next_up(),
                        ..sed
                    }),
                    ..b0
                },
                "Survey",
            )?,
            make(
                Some(0.03),
                CulvertBlockage {
                    sediment: None,
                    ..b0
                },
                "Survey",
            )?,
            make(
                Some(0.03),
                CulvertBlockage {
                    closed_barrels: 2,
                    ..b0
                },
                "Survey",
            )?,
            make(Some(0.03), b0, "Survey.")?,
        ];
        for other in &others {
            assert!(!same_elements(
                std::slice::from_ref(&base),
                std::slice::from_ref(other)
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
