use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn orifice_evidence() -> OrificeApplicability {
    OrificeApplicability {
        coefficient_provenance: "Independent analytical numerical fixture".into(),
        fully_submerged_applicability: "Both known wet stages strictly exceed crown".into(),
        symmetric_reversal: true,
    }
}

pub fn weir_evidence() -> WeirApplicability {
    WeirApplicability {
        coefficient_provenance: "Independent analytical numerical fixture".into(),
        free_nappe_and_approach_assertion:
            "Suppressed sharp-crested rectangular analytical fixture".into(),
        submergence_correction_assertion: "Adopted Villemonte analytical correction".into(),
        symmetric_reversal: true,
    }
}

pub fn orifice(cd: f64, area: f64, gravity: f64) -> Result<StructureLaw> {
    Ok(StructureLaw::SubmergedOrifice(
        SubmergedOrificeParameters::new(area, -2.0, -1.0, cd, gravity, orifice_evidence())?,
    ))
}

pub fn weir(coefficient: f64, length: f64, crest: f64) -> Result<StructureLaw> {
    Ok(StructureLaw::SuppressedRectangularWeir(
        SuppressedRectangularWeirParameters::new(length, crest, coefficient, weir_evidence())?,
    ))
}

pub fn element(id: &str, law: StructureLaw, direction: DirectionPolicy) -> StructureElement {
    StructureElement {
        id: id.into(),
        endpoint_a: "reference-a".into(),
        endpoint_b: "reference-b".into(),
        datum: "Analytical fixture datum".into(),
        direction,
        law,
    }
}

pub fn packet(id: &str, a: HeadSupport, b: HeadSupport) -> StructureHeads {
    StructureHeads {
        structure_id: id.into(),
        endpoint_a: EndpointHead {
            id: "reference-a".into(),
            support: a,
        },
        endpoint_b: EndpointHead {
            id: "reference-b".into(),
            support: b,
        },
        datum: "Analytical fixture datum".into(),
        convention: HeadConvention::QuiescentStage,
    }
}

pub fn heads(id: &str, a: f64, b: f64) -> StructureHeads {
    packet(
        id,
        HeadSupport::Wet { stage_m: a },
        HeadSupport::Wet { stage_m: b },
    )
}

pub fn first(proposal: &StructureProposal<crate::runtime::Runtime>) -> Result<StructureOutput> {
    proposal
        .outputs()?
        .into_iter()
        .next()
        .ok_or_else(|| "Missing structure output".into())
}

pub fn rejection<T>(value: std::result::Result<T, StructureError>) -> Result<StructureError> {
    match value {
        Ok(_) => Err("Expected structure rejection".into()),
        Err(error) => Ok(error),
    }
}

pub fn close(actual: f64, expected: f64, ordinary: bool) {
    let ulp = expected.next_up() - expected;
    let tolerance = if ordinary {
        1e-12 + 1e-12 * expected.abs()
    } else {
        (32.0 * ulp).max(1e-12 * expected.abs())
    };
    assert!(
        actual.is_finite() && actual > 0.0,
        "Positive finite discharge required: {actual}"
    );
    assert!(
        (actual - expected).abs() <= tolerance,
        "discharge {actual:e}, reference {expected:e}, tolerance {tolerance:e}"
    );
}
