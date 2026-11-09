use super::fixture::*;
use hydr::structures::*;

#[test]
fn gate_definitions_require_positive_geometry_and_complete_applicability() -> Result {
    for value in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        for field in 0..4 {
            let mut values = [1.0, 1.0, 1.0, 0.5];
            values[field] = value;
            assert!(
                GateParameters::new(
                    values[0],
                    0.0,
                    values[1],
                    values[2],
                    values[3],
                    applicability()
                )
                .is_err()
            );
        }
    }
    for sill in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(GateParameters::new(1.0, sill, 1.0, 1.0, 0.5, applicability()).is_err());
    }
    for field in 0..4 {
        for invalid in [String::new(), "Invalid\nassertion".into()] {
            let mut evidence = applicability();
            match field {
                0 => evidence.coefficient_provenance = invalid,
                1 => evidence.hydrostatic_quiescent_pool_assertion = invalid,
                2 => evidence.free_submerged_aperture_applicability = invalid,
                _ => evidence.unobstructed_aperture_assertion = invalid,
            }
            assert!(GateParameters::new(1.0, 0.0, 1.0, 1.0, 0.5, evidence).is_err());
        }
    }
    let mut evidence = applicability();
    evidence.symmetric_reversal = false;
    let law = StructureLaw::HydrostaticRectangularGate(GateParameters::new(
        1.0, 0.0, 1.0, 1.0, 0.5, evidence,
    )?);
    assert!(
        StructureBatch::new(vec![element(
            "gate",
            law.clone(),
            DirectionPolicy::Bidirectional
        )])
        .is_err()
    );
    assert!(StructureBatch::new(vec![element("gate", law, DirectionPolicy::AtoBOnly)]).is_ok());
    Ok(())
}
