//! Public pump proposals for independently frozen exact-rational comparisons.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "pump_laws_reference/fixtures.rs"]
mod fixtures;

use std::io::{self, Write};

use fixtures::{Case, Expected};
use hydr::structures::{
    DirectionPolicy, EndpointHead, HeadConvention, HeadSupport, HeldHeadInterval,
    PumpApplicability, PumpCurve, PumpCurveKnot, PumpParameters, PumpSpeedPolicy, StructureBatch,
    StructureElement, StructureError, StructureHeads, StructureHydraulicOperands,
    StructureInputErrorKind, StructureLaw, StructureOperatingPoint, StructureOperation,
    StructureStatus,
};

const DATUM: &str = "Analytical fixture datum";
const EVIDENCE: &str = "Synthetic numerical fixture in SI units at reference speed with declared fluid applicability; no measured accuracy claim";

fn observation_error(case: &Case, detail: &str) -> StructureError {
    StructureError::Numerical {
        element: Some(0),
        context: format!(
            "pump reference case={}: {detail}; check original fixture and production output",
            case.id
        ),
    }
}

fn write_rejection(output: &mut impl Write, case: &Case, status: &str) -> io::Result<()> {
    write!(output, "{},{status}", case.id)?;
    for _ in 0..10 {
        write!(output, ",")?;
    }
    writeln!(output, ",{status}")
}

fn observe(output: &mut impl Write, case: &Case) -> Result<(), Box<dyn std::error::Error>> {
    let curve = PumpCurve::new(
        case.knots
            .iter()
            .map(|&(head_m, discharge_m3_per_s)| PumpCurveKnot {
                head_m,
                discharge_m3_per_s,
            })
            .collect(),
        EVIDENCE.into(),
    )?;
    let speed_policy = if case.reference_only {
        PumpSpeedPolicy::ReferenceOnly
    } else {
        PumpSpeedPolicy::DeclaredAffinity {
            minimum_ratio: f64::from_bits(1),
            maximum_ratio: f64::MAX,
            applicability:
                "Explicit synthetic affinity scaling over the declared numerical fixture envelope"
                    .into(),
        }
    };
    let parameters = PumpParameters::new(
        curve,
        -2.0,
        speed_policy,
        PumpApplicability {
            endpoint_stage_head_assertion: "Known wet quiescent endpoint stages use the common analytical fixture datum".into(),
            connection_losses_and_velocity_head_account: "The synthetic effective stage curve includes declared zero connection losses and velocity-head differences".into(),
            stopped_isolation_assertion: "A declared closed isolator prevents forward siphoning and reverse flow while stopped".into(),
            operating_envelope: EVIDENCE.into(),
        },
    )?;
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![StructureElement {
            id: case.id.into(),
            endpoint_a: "reference-a".into(),
            endpoint_b: "reference-b".into(),
            datum: DATUM.into(),
            direction: DirectionPolicy::AtoBOnly,
            law: StructureLaw::Pump(parameters),
        }],
    )?;
    let heads = [StructureHeads {
        structure_id: case.id.into(),
        endpoint_a: EndpointHead {
            id: "reference-a".into(),
            support: HeadSupport::Wet {
                stage_m: case.donor_stage,
            },
        },
        endpoint_b: EndpointHead {
            id: "reference-b".into(),
            support: HeadSupport::Wet {
                stage_m: case.receiving_stage,
            },
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }];
    let operations = [StructureOperation {
        structure_id: case.id.into(),
        operating_point: StructureOperatingPoint::PumpRunning {
            speed_ratio: case.speed,
        },
    }];
    let proposal = match batch.propose_with_operations(&heads, &operations) {
        Ok(proposal) => proposal,
        Err(StructureError::Numerical { .. }) if matches!(case.expected, Expected::Numerical) => {
            write_rejection(output, case, "Numerical")?;
            return Ok(());
        }
        Err(StructureError::InvalidInput {
            kind: StructureInputErrorKind::UnsupportedRegime,
            ..
        }) if matches!(case.expected, Expected::UnsupportedRegime) => {
            write_rejection(output, case, "UnsupportedRegime")?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let values = proposal.outputs()?;
    let value = values
        .first()
        .ok_or_else(|| observation_error(case, "missing pump output"))?;
    let status = match (&case.expected, &value.status) {
        (Expected::Running, StructureStatus::PumpRunning) => "PumpRunning",
        (Expected::Shutoff, StructureStatus::PumpShutoff) => "PumpShutoff",
        _ => return Err(observation_error(case, "unexpected admitted status").into()),
    };
    let Some(StructureHydraulicOperands::Pump(operands)) = &value.hydraulic_operands else {
        return Err(observation_error(case, "missing evaluated pump operands").into());
    };
    let held = proposal.held_volumes(HeldHeadInterval {
        start: 0.0,
        end: 3.0,
    })?;
    let volumes = held.outputs()?;
    let volume = volumes
        .first()
        .ok_or_else(|| observation_error(case, "missing held-head volume"))?
        .volume_m3;
    let numbers = [
        operands.receiving_minus_donor_head_m(),
        operands.receiving_minus_donor_head_low_m(),
        operands.reference_head_m(),
        operands.normalization_error_bound_m(),
        operands.reference_head_lower_m(),
        operands.reference_head_upper_m(),
        operands.reference_discharge_m3_per_s(),
        value.discharge_m3_per_s,
        volume,
    ];
    if numbers.iter().any(|number| !number.is_finite()) {
        return Err(observation_error(case, "nonfinite projected numerical operand").into());
    }
    write!(output, "{},{status}", case.id)?;
    for number in numbers {
        write!(output, ",{number:.17e}")?;
    }
    writeln!(output, ",{},", operands.division_exact())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(
        output,
        "case,status,head_high_m,head_low_m,reference_head_m,normalization_error_bound_m,reference_head_lower_m,reference_head_upper_m,reference_discharge_m3_per_s,discharge_m3_per_s,held_volume_m3,division_exact,error_kind"
    )?;
    for case in fixtures::OPERATING_CASES
        .iter()
        .chain(fixtures::ARITHMETIC_CASES.iter())
    {
        observe(&mut output, case)?;
    }
    output.flush()?;
    Ok(())
}
