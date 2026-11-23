//! Public-API observations for frozen high-precision structure-law fixtures.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

use std::io::{self, Write};

use hydr::structures::{
    DirectionPolicy, EndpointHead, HeadConvention, HeadSupport, HeldHeadInterval,
    OrificeApplicability, StructureBatch, StructureCondition, StructureElement, StructureError,
    StructureHeads, StructureLaw, StructureStatus, SubmergedOrificeParameters,
    SuppressedRectangularWeirParameters, WeirApplicability,
};

const DATUM: &str = "Analytical fixture datum";
const ENDPOINT_A: &str = "reference-a";
const ENDPOINT_B: &str = "reference-b";
const EVIDENCE: &str = "Numerical analytical fixtures. No measured physical applicability claim.";

enum RawLaw {
    Orifice {
        coefficient: f64,
        area: f64,
        gravity: f64,
    },
    Weir {
        crest: f64,
        coefficient: f64,
        length: f64,
    },
}

struct Case {
    id: &'static str,
    high: f64,
    low: f64,
    law: RawLaw,
    numerical_error_expected: bool,
}

impl Case {
    const fn orifice(
        id: &'static str,
        high: f64,
        low: f64,
        coefficient: f64,
        area: f64,
        gravity: f64,
        numerical_error_expected: bool,
    ) -> Self {
        Self {
            id,
            high,
            low,
            law: RawLaw::Orifice {
                coefficient,
                area,
                gravity,
            },
            numerical_error_expected,
        }
    }

    const fn weir(
        id: &'static str,
        high: f64,
        low: f64,
        crest: f64,
        coefficient: f64,
        length: f64,
        numerical_error_expected: bool,
    ) -> Self {
        Self {
            id,
            high,
            low,
            law: RawLaw::Weir {
                crest,
                coefficient,
                length,
            },
            numerical_error_expected,
        }
    }
}

const CASES: [Case; 14] = [
    Case::orifice("orifice_ordinary", 3.0, 2.0, 0.6, 2.0, 9.81, false),
    Case::orifice(
        "orifice_velocity_overflow",
        f64::MAX,
        0.0,
        1e-308,
        1.0,
        f64::MAX,
        false,
    ),
    Case::orifice(
        "orifice_coefficient_product_underflow",
        f64::MAX,
        0.0,
        1e-200,
        1e-200,
        f64::MAX,
        false,
    ),
    Case::orifice(
        "orifice_radicand_underflow",
        1e-200,
        0.0,
        1e200,
        1.0,
        1e-200,
        false,
    ),
    Case::orifice(
        "orifice_discharge_underflow",
        1e-200,
        0.0,
        f64::from_bits(1),
        1.0,
        1.0,
        true,
    ),
    Case::weir("weir_free", 11.0, 9.0, 10.0, 1.7, 2.0, false),
    Case::weir("weir_submerged", 11.0, 10.5, 10.0, 1.7, 2.0, false),
    Case::weir(
        "weir_adjacent_heads",
        1.0,
        0.9999999999999999,
        0.0,
        1.0,
        1.0,
        false,
    ),
    Case::weir(
        "weir_rounded_depth_plateau",
        1.0,
        0.9999999999999999,
        -1.0,
        1.0,
        1.0,
        false,
    ),
    Case::weir(
        "weir_normalized_gap_underflow",
        f64::from_bits(1),
        0.0,
        -1e308,
        1.0,
        1.0,
        false,
    ),
    Case::weir(
        "weir_near_free",
        1.0,
        f64::from_bits(1),
        0.0,
        1.0,
        1.0,
        false,
    ),
    Case::weir(
        "weir_free_discharge_overflow_intermediate",
        1e300,
        5e299,
        0.0,
        1e-200,
        1.0,
        false,
    ),
    Case::weir(
        "weir_coefficient_product_underflow",
        1e300,
        5e299,
        0.0,
        f64::from_bits(1),
        f64::from_bits(1),
        false,
    ),
    Case::weir(
        "weir_discharge_overflow",
        4.0,
        0.0,
        0.0,
        f64::MAX,
        1.0,
        true,
    ),
];

fn observation_error(case: &Case, detail: &str) -> StructureError {
    StructureError::Numerical {
        element: Some(0),
        context: format!(
            "reference case={}: {detail}; check fixture and production output",
            case.id
        ),
    }
}

fn observe(output: &mut impl Write, case: &Case) -> Result<(), Box<dyn std::error::Error>> {
    let law = match case.law {
        RawLaw::Orifice {
            coefficient,
            area,
            gravity,
        } => StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
            area,
            -2.0,
            -1.0,
            coefficient,
            gravity,
            OrificeApplicability {
                coefficient_provenance: EVIDENCE.into(),
                fully_submerged_applicability: EVIDENCE.into(),
                symmetric_reversal: true,
            },
        )?),
        RawLaw::Weir {
            crest,
            coefficient,
            length,
        } => StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            length,
            crest,
            coefficient,
            WeirApplicability {
                coefficient_provenance: EVIDENCE.into(),
                free_nappe_and_approach_assertion: EVIDENCE.into(),
                submergence_correction_assertion: EVIDENCE.into(),
                symmetric_reversal: true,
            },
        )?),
    };
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![StructureElement {
            id: case.id.into(),
            endpoint_a: ENDPOINT_A.into(),
            endpoint_b: ENDPOINT_B.into(),
            datum: DATUM.into(),
            direction: DirectionPolicy::Bidirectional,
            law,
            condition: StructureCondition::AsBuilt,
        }],
    )?;
    let proposal = match batch.propose(&[StructureHeads {
        structure_id: case.id.into(),
        endpoint_a: EndpointHead {
            id: ENDPOINT_A.into(),
            support: HeadSupport::Wet { stage_m: case.high },
        },
        endpoint_b: EndpointHead {
            id: ENDPOINT_B.into(),
            support: HeadSupport::Wet { stage_m: case.low },
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }]) {
        Ok(proposal) => proposal,
        Err(StructureError::Numerical { .. }) if case.numerical_error_expected => {
            writeln!(output, "{},numerical_error,,,numerical", case.id)?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    if case.numerical_error_expected {
        return Err(observation_error(
            case,
            "unrepresentable final discharge unexpectedly admitted",
        )
        .into());
    }
    let values = proposal.outputs()?;
    let value = values
        .first()
        .ok_or_else(|| observation_error(case, "missing discharge output"))?;
    let discharge = value.discharge_m3_per_s;
    if !matches!(value.status, StructureStatus::Active)
        || !discharge.is_finite()
        || discharge <= 0.0
    {
        return Err(observation_error(case, "expected active finite positive discharge").into());
    }
    let held = proposal.held_volumes(HeldHeadInterval {
        start: 0.0,
        end: 2.0,
    })?;
    let volumes = held.outputs()?;
    let volume = volumes
        .first()
        .ok_or_else(|| observation_error(case, "missing held-head volume"))?
        .volume_m3;
    if !volume.is_finite() || volume <= 0.0 {
        return Err(observation_error(case, "expected finite positive held-head volume").into());
    }
    writeln!(output, "{},active,{discharge:.17e},{volume:.17e},", case.id)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(
        output,
        "case,status,discharge_m3_per_s,held_volume_m3,error_kind"
    )?;
    for case in &CASES {
        observe(&mut output, case)?;
    }
    output.flush()?;
    Ok(())
}
