use super::{
    reference_case,
    reference_inputs::{self, Case},
};
use hydr::structures::*;

pub use reference_case::Result;

pub fn case(id: &str) -> Result<&'static Case> {
    reference_inputs::cases()
        .find(|case| case.id == id)
        .ok_or_else(|| format!("Missing trajectory reference {id}").into())
}

pub fn bind(id: &str, width: usize) -> Result<GateTrajectoryBatch> {
    let case = case(id)?;
    let definition = reference_case::trajectory(case)?;
    Ok(reference_case::structure_batch(case, width)?
        .with_gate_trajectories(reference_case::assignments(&definition, width))?)
}

pub fn exact() -> GateTrajectoryAccuracy {
    GateTrajectoryAccuracy {
        maximum_absolute_opening_error_m: 0.0,
        maximum_relative_opening_error: 0.0,
    }
}

pub fn permissive() -> GateTrajectoryAccuracy {
    GateTrajectoryAccuracy {
        maximum_absolute_opening_error_m: f64::MAX,
        maximum_relative_opening_error: 0.0,
    }
}

pub fn prescribed(
    knots: &[(f64, f64)],
    accuracy: GateTrajectoryAccuracy,
) -> std::result::Result<GateOpeningTrajectory, StructureError> {
    GateOpeningTrajectory::new(
        knots
            .iter()
            .map(|&(time_s, opening_m)| GateOpeningKnot { time_s, opening_m })
            .collect(),
        GateMotionLimits {
            maximum_opening_speed_m_per_s: f64::MAX,
            maximum_closing_speed_m_per_s: f64::MAX,
        },
        accuracy,
        "Explicit analytical trajectory definition".into(),
    )
}

pub fn stationary(
    value: f64,
    start: f64,
    end: f64,
) -> std::result::Result<GateOpeningTrajectory, StructureError> {
    prescribed(&[(start, value), (end, value)], exact())
}

pub fn operation(id: &str, operating_point: StructureOperatingPoint) -> StructureOperation {
    StructureOperation {
        structure_id: id.into(),
        operating_point,
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

pub fn wet(id: &str) -> StructureHeads {
    packet(
        id,
        HeadSupport::Wet { stage_m: 2.0 },
        HeadSupport::Wet { stage_m: 1.0 },
    )
}

pub fn unknown(id: &str) -> StructureHeads {
    packet(id, HeadSupport::Unknown, HeadSupport::Unknown)
}

fn element(id: &str, law: StructureLaw, direction: DirectionPolicy) -> StructureElement {
    StructureElement {
        id: id.into(),
        endpoint_a: "reference-a".into(),
        endpoint_b: "reference-b".into(),
        datum: "Analytical fixture datum".into(),
        law,
        direction,
    }
}

pub fn passive(id: &str) -> Result<StructureElement> {
    Ok(element(
        id,
        StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            0.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Analytical sentinel".into(),
                free_nappe_and_approach_assertion: "Suppressed analytical crest".into(),
                submergence_correction_assertion: "Declared analytical correction".into(),
                symmetric_reversal: true,
            },
        )?),
        DirectionPolicy::Bidirectional,
    ))
}

pub fn pump(id: &str) -> Result<StructureElement> {
    Ok(element(
        id,
        StructureLaw::Pump(PumpParameters::new(
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
                "Analytical reference curve".into(),
            )?,
            -1.0,
            PumpSpeedPolicy::ReferenceOnly,
            PumpApplicability {
                endpoint_stage_head_assertion: "Quiescent analytical endpoints".into(),
                connection_losses_and_velocity_head_account:
                    "Analytical curve includes connection losses".into(),
                stopped_isolation_assertion: "Explicitly isolated stopped connection".into(),
                operating_envelope: "Analytical numerical fixture".into(),
            },
        )?),
        DirectionPolicy::AtoBOnly,
    ))
}

pub fn mixed() -> Result<StructureBatch> {
    Ok(StructureBatch::new(vec![
        passive("passive")?,
        reference_case::gate("gate-0", 1.0)?,
        pump("pump")?,
        reference_case::gate("gate-1", 1.0)?,
    ])?)
}

pub fn mixed_assignment(
    a: GateOpeningTrajectory,
    b: GateOpeningTrajectory,
) -> Vec<GateTrajectoryAssignment> {
    vec![
        GateTrajectoryAssignment {
            structure_id: "gate-0".into(),
            trajectory: a,
        },
        GateTrajectoryAssignment {
            structure_id: "gate-1".into(),
            trajectory: b,
        },
    ]
}

pub fn other_operations() -> Vec<StructureOperation> {
    vec![
        operation("passive", StructureOperatingPoint::Passive),
        operation("pump", StructureOperatingPoint::PumpStoppedIsolated),
    ]
}

pub fn mixed_heads() -> Vec<StructureHeads> {
    vec![
        packet(
            "passive",
            HeadSupport::Wet { stage_m: 0.0 },
            HeadSupport::Wet { stage_m: 0.0 },
        ),
        unknown("gate-0"),
        unknown("pump"),
        wet("gate-1"),
    ]
}

pub fn rejection<T>(result: std::result::Result<T, StructureError>) -> Result<StructureError> {
    result
        .err()
        .ok_or_else(|| "Expected trajectory rejection".into())
}

pub fn expect_error<T>(
    result: std::result::Result<T, StructureError>,
    expected: &str,
    element: Option<usize>,
) -> Result {
    reference_case::check_error(&rejection(result)?, expected, element)
}

pub fn expect_kind<T>(
    result: std::result::Result<T, StructureError>,
    expected: StructureInputErrorKind,
) -> Result {
    let error = rejection(result)?;
    assert_eq!(error.input_kind(), Some(expected), "{error}");
    Ok(())
}
