use super::fixture::*;
use hydr::structures::*;

pub struct Reference {
    pub id: &'static str,
    pub law: StructureLaw,
    pub high: f64,
    pub low: f64,
    pub discharge: Option<f64>,
    pub ordinary: bool,
}

pub fn catalog() -> Result<Vec<Reference>> {
    let tiny = f64::from_bits(1);
    let orifices = [
        (
            "orifice_ordinary",
            3.0,
            2.0,
            0.6,
            2.0,
            9.81,
            Some(5.315336301684024),
            true,
        ),
        (
            "orifice_velocity_overflow",
            f64::MAX,
            0.0,
            1e-308,
            1.0,
            f64::MAX,
            Some(2.542322012307292),
            false,
        ),
        (
            "orifice_coefficient_product_underflow",
            f64::MAX,
            0.0,
            1e-200,
            1e-200,
            f64::MAX,
            Some(2.542322012307292e-92),
            false,
        ),
        (
            "orifice_radicand_underflow",
            1e-200,
            0.0,
            1e200,
            1.0,
            1e-200,
            Some(f64::from_bits(0x3ff6_a09e_667f_3bcc)),
            false,
        ),
        (
            "orifice_discharge_underflow",
            1e-200,
            0.0,
            tiny,
            1.0,
            1.0,
            None,
            false,
        ),
    ];
    let mut rows = Vec::new();
    for (id, high, low, cd, area, gravity, discharge, ordinary) in orifices {
        rows.push(Reference {
            id,
            law: orifice(cd, area, gravity)?,
            high,
            low,
            discharge,
            ordinary,
        });
    }
    let weirs = [
        ("weir_free", 11.0, 9.0, 10.0, 1.7, 2.0, Some(3.4), true),
        (
            "weir_submerged",
            11.0,
            10.5,
            10.0,
            1.7,
            2.0,
            Some(2.8743125549917656),
            true,
        ),
        (
            "weir_adjacent_heads",
            1.0,
            1.0_f64.next_down(),
            0.0,
            1.0,
            1.0,
            Some(8.419340487230523e-7),
            false,
        ),
        (
            "weir_rounded_depth_plateau",
            1.0,
            1.0_f64.next_down(),
            -1.0,
            1.0,
            1.0,
            Some(1.8235871294913473e-6),
            false,
        ),
        (
            "weir_normalized_gap_underflow",
            tiny,
            0.0,
            -1e308,
            1.0,
            1.0,
            Some(1.0349091521962441e219),
            false,
        ),
        ("weir_near_free", 1.0, tiny, 0.0, 1.0, 1.0, Some(1.0), false),
        (
            "weir_free_discharge_overflow_intermediate",
            1e300,
            5e299,
            0.0,
            1e-200,
            1.0,
            Some(8.453860455858135e249),
            false,
        ),
        (
            "weir_coefficient_product_underflow",
            1e300,
            5e299,
            0.0,
            tiny,
            tiny,
            Some(2.063594627888692e-197),
            false,
        ),
        (
            "weir_discharge_overflow",
            4.0,
            0.0,
            0.0,
            f64::MAX,
            1.0,
            None,
            false,
        ),
    ];
    for (id, high, low, crest, coefficient, length, discharge, ordinary) in weirs {
        rows.push(Reference {
            id,
            law: weir(coefficient, length, crest)?,
            high,
            low,
            discharge,
            ordinary,
        });
    }
    Ok(rows)
}

#[test]
fn independent_discharge_references_and_reversal() -> Result {
    for row in catalog()? {
        let batch = StructureBatch::new(vec![element(
            row.id,
            row.law,
            DirectionPolicy::Bidirectional,
        )])?;
        for reverse in [false, true] {
            let (a, b) = if reverse {
                (row.low, row.high)
            } else {
                (row.high, row.low)
            };
            let proposal = batch.propose(&[heads(row.id, a, b)]);
            if let Some(expected) = row.discharge {
                let output = first(&proposal?)?;
                assert_eq!(output.status, StructureStatus::Active, "{}", row.id);
                assert_eq!(
                    output.proposed_direction,
                    Some(if reverse {
                        FlowDirection::BtoA
                    } else {
                        FlowDirection::AtoB
                    })
                );
                close(
                    if reverse {
                        -output.discharge_m3_per_s
                    } else {
                        output.discharge_m3_per_s
                    },
                    expected,
                    row.ordinary,
                );
            } else {
                let error = rejection(proposal)?;
                assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
                assert_eq!(error.element(), Some(0));
            }
        }
    }
    Ok(())
}
