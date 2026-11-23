use super::reference_inputs::{GateReferenceCase, ReferenceExpectation, ReferenceHead};
use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn applicability() -> GateApplicability {
    GateApplicability {
        coefficient_provenance: "Independent original-F64 analytical aperture fixture".into(),
        hydrostatic_quiescent_pool_assertion:
            "Hydrostatic quiescent pools share the analytical datum".into(),
        free_submerged_aperture_applicability:
            "One analytical coefficient covers partial wetting and all submergence limits".into(),
        unobstructed_aperture_assertion:
            "Both connections expose the complete sill-to-crown aperture".into(),
        symmetric_reversal: true,
    }
}

pub fn definition(case: &GateReferenceCase, id: &str) -> Result<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "reference-a".into(),
        endpoint_b: "reference-b".into(),
        datum: "Analytical fixture datum".into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            case.width_m,
            case.sill_m,
            case.maximum_opening_m,
            case.discharge_coefficient,
            case.gravity_m_per_s2,
            applicability(),
        )?),
        condition: StructureCondition::AsBuilt,
    })
}

pub fn support(head: ReferenceHead) -> HeadSupport {
    match head {
        ReferenceHead::Wet(stage_m) => HeadSupport::Wet { stage_m },
        ReferenceHead::Dry(bed_elevation_m) => HeadSupport::Dry { bed_elevation_m },
        ReferenceHead::Unknown => HeadSupport::Unknown,
    }
}

pub fn packet(case: &GateReferenceCase, id: &str, reverse: bool) -> StructureHeads {
    let (a, b) = if reverse {
        (case.endpoint_b, case.endpoint_a)
    } else {
        (case.endpoint_a, case.endpoint_b)
    };
    StructureHeads {
        structure_id: id.into(),
        endpoint_a: EndpointHead {
            id: "reference-a".into(),
            support: support(a),
        },
        endpoint_b: EndpointHead {
            id: "reference-b".into(),
            support: support(b),
        },
        datum: "Analytical fixture datum".into(),
        convention: HeadConvention::QuiescentStage,
    }
}

pub fn operation(case: &GateReferenceCase, id: &str) -> StructureOperation {
    StructureOperation {
        structure_id: id.into(),
        operating_point: StructureOperatingPoint::GateOpening {
            opening_m: case.opening_m,
        },
    }
}

pub fn expected_label(expected: ReferenceExpectation) -> &'static str {
    match expected {
        ReferenceExpectation::Success => "success",
        ReferenceExpectation::Numerical => "numerical",
        ReferenceExpectation::Unsupported => "unsupported",
    }
}

pub fn error_label(error: &StructureError) -> &'static str {
    match error {
        StructureError::Numerical { .. } => "numerical",
        StructureError::InvalidInput { .. } => "unsupported",
        StructureError::Runtime { .. } => "runtime",
    }
}

pub fn expected_input_kind(case: &GateReferenceCase) -> Option<StructureInputErrorKind> {
    match case.id {
        "positive_opening_unknown" => Some(StructureInputErrorKind::UnknownSupport),
        "dry_bed_above_sill" => Some(StructureInputErrorKind::UnsupportedRegime),
        _ => None,
    }
}
