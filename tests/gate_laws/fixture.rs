use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn applicability() -> GateApplicability {
    GateApplicability {
        coefficient_provenance: "Analytical numerical aperture fixture".into(),
        hydrostatic_quiescent_pool_assertion:
            "Hydrostatic pools with negligible approach velocity on one datum".into(),
        free_submerged_aperture_applicability:
            "One analytical coefficient covers free and submerged aperture limits".into(),
        unobstructed_aperture_assertion:
            "Each connection exposes the complete sill-to-crown aperture".into(),
        symmetric_reversal: true,
    }
}

pub fn parameters() -> Result<GateParameters> {
    Ok(GateParameters::new(
        1.0,
        0.0,
        1.0,
        1.0,
        0.5,
        applicability(),
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

pub fn standard(id: &str, direction: DirectionPolicy) -> Result<StructureElement> {
    Ok(element(
        id,
        StructureLaw::HydrostaticRectangularGate(parameters()?),
        direction,
    ))
}

pub fn passive_law() -> Result<StructureLaw> {
    Ok(StructureLaw::SuppressedRectangularWeir(
        SuppressedRectangularWeirParameters::new(
            1.0,
            0.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Analytical SI fixture".into(),
                free_nappe_and_approach_assertion: "Suppressed analytical crest".into(),
                submergence_correction_assertion: "Declared analytical correction".into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

pub fn pump_law() -> Result<StructureLaw> {
    Ok(StructureLaw::Pump(PumpParameters::new(
        PumpCurve::new(
            vec![
                PumpCurveKnot {
                    head_m: 0.0,
                    discharge_m3_per_s: 1.0,
                },
                PumpCurveKnot {
                    head_m: 1.0,
                    discharge_m3_per_s: 0.0,
                },
            ],
            "Analytical reference-speed curve".into(),
        )?,
        -1.0,
        PumpSpeedPolicy::ReferenceOnly,
        PumpApplicability {
            endpoint_stage_head_assertion: "Quiescent analytical endpoints".into(),
            connection_losses_and_velocity_head_account: "Analytical effective curve".into(),
            stopped_isolation_assertion: "Explicit closed isolation".into(),
            operating_envelope: "Analytical numerical fixture".into(),
        },
    )?))
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

pub fn operation(id: &str, operating_point: StructureOperatingPoint) -> StructureOperation {
    StructureOperation {
        structure_id: id.into(),
        operating_point,
    }
}

pub fn opening(id: &str, opening_m: f64) -> StructureOperation {
    operation(id, StructureOperatingPoint::GateOpening { opening_m })
}

pub fn first(proposal: &StructureProposal) -> Result<StructureOutput> {
    proposal
        .outputs()?
        .into_iter()
        .next()
        .ok_or_else(|| "Missing gate output".into())
}

pub fn rejection<T>(value: std::result::Result<T, StructureError>) -> Result<StructureError> {
    match value {
        Ok(_) => Err("Expected gate rejection".into()),
        Err(error) => Ok(error),
    }
}

pub fn kind(
    batch: &StructureBatch,
    packets: &[StructureHeads],
    operations: &[StructureOperation],
    expected: StructureInputErrorKind,
) -> Result {
    let error = rejection(batch.propose_with_operations(packets, operations))?;
    assert_eq!(error.input_kind(), Some(expected), "{error}");
    Ok(())
}
