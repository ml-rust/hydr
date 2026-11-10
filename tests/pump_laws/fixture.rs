use super::references::Case;
use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn applicability() -> PumpApplicability {
    PumpApplicability {
        endpoint_stage_head_assertion: "Quiescent stages on the common analytical datum".into(),
        connection_losses_and_velocity_head_account:
            "Effective analytical curve includes endpoint reference losses and velocity heads"
                .into(),
        stopped_isolation_assertion:
            "Declared isolation prevents forward siphoning and reverse flow".into(),
        operating_envelope: "Analytical numerical fixture, no measured pump applicability claim"
            .into(),
    }
}

pub fn affinity() -> PumpSpeedPolicy {
    PumpSpeedPolicy::DeclaredAffinity {
        minimum_ratio: f64::from_bits(1),
        maximum_ratio: f64::MAX,
        applicability: "Declared unchanged-geometry analytical affinity fixture".into(),
    }
}

pub fn curve(heads: &[f64], flows: &[f64]) -> Result<PumpCurve> {
    Ok(PumpCurve::new(
        heads
            .iter()
            .zip(flows)
            .map(|(&head_m, &discharge_m3_per_s)| PumpCurveKnot {
                head_m,
                discharge_m3_per_s,
            })
            .collect(),
        "Exact rational analytical SI curve, reference speed1, numerical fluid fixture".into(),
    )?)
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

pub fn definition(case: &Case, id: &str, reverse: bool) -> Result<StructureElement> {
    let policy = if case.reference_only {
        PumpSpeedPolicy::ReferenceOnly
    } else {
        affinity()
    };
    let parameters = PumpParameters::new(
        curve(&case.heads, &case.flows)?,
        -2.0,
        policy,
        applicability(),
    )?;
    Ok(element(
        id,
        StructureLaw::Pump(parameters),
        if reverse {
            DirectionPolicy::BtoAOnly
        } else {
            DirectionPolicy::AtoBOnly
        },
    ))
}

pub fn standard(id: &str) -> Result<StructureElement> {
    Ok(element(
        id,
        StructureLaw::Pump(PumpParameters::new(
            curve(&[0.0, 4.0, 8.0], &[4.0, 2.0, 0.0])?,
            -2.0,
            affinity(),
            applicability(),
        )?),
        DirectionPolicy::AtoBOnly,
    ))
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

pub fn case_heads(case: &Case, id: &str, reverse: bool) -> StructureHeads {
    if reverse {
        heads(id, case.receiver, case.donor)
    } else {
        heads(id, case.donor, case.receiver)
    }
}

pub fn operation(id: &str, operating_point: StructureOperatingPoint) -> StructureOperation {
    StructureOperation {
        structure_id: id.into(),
        operating_point,
    }
}

pub fn running(id: &str, speed_ratio: f64) -> StructureOperation {
    operation(id, StructureOperatingPoint::PumpRunning { speed_ratio })
}

pub fn first(proposal: &StructureProposal<crate::runtime::Runtime>) -> Result<StructureOutput> {
    proposal
        .outputs()?
        .into_iter()
        .next()
        .ok_or_else(|| "Missing pump output".into())
}

pub fn rejection<T>(result: std::result::Result<T, StructureError>) -> Result<StructureError> {
    match result {
        Ok(_) => Err("Expected pump rejection".into()),
        Err(error) => Ok(error),
    }
}

pub fn kind(
    batch: &StructureBatch<crate::runtime::Runtime>,
    heads: &[StructureHeads],
    operations: &[StructureOperation],
    expected: StructureInputErrorKind,
) -> Result {
    let error = rejection(batch.propose_with_operations(heads, operations))?;
    assert_eq!(error.input_kind(), Some(expected), "{error}");
    Ok(())
}
