use super::reference_inputs::{Case, Query};
use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const LAYOUTS: [usize; 6] = [1, 31, 32, 33, 64, 65];

pub fn element_id(lane: usize) -> String {
    format!("gate-{lane}")
}

pub fn accuracy(case: &Case) -> GateTrajectoryAccuracy {
    GateTrajectoryAccuracy {
        maximum_absolute_opening_error_m: case.absolute,
        maximum_relative_opening_error: case.relative,
    }
}

pub fn trajectory(case: &Case) -> std::result::Result<GateOpeningTrajectory, StructureError> {
    GateOpeningTrajectory::new(
        case.knots
            .iter()
            .map(|&(time_s, opening_m)| GateOpeningKnot { time_s, opening_m })
            .collect(),
        GateMotionLimits {
            maximum_opening_speed_m_per_s: case.opening_speed,
            maximum_closing_speed_m_per_s: case.closing_speed,
        },
        accuracy(case),
        format!("Original-F64 numerical trajectory fixture {}", case.id),
    )
}

pub fn gate(
    id: &str,
    maximum_opening: f64,
) -> std::result::Result<StructureElement, StructureError> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "reference-a".into(),
        endpoint_b: "reference-b".into(),
        datum: "Analytical fixture datum".into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            1.0,
            0.0,
            maximum_opening,
            1.0,
            0.5,
            GateApplicability {
                coefficient_provenance: "Analytical trajectory fixture".into(),
                hydrostatic_quiescent_pool_assertion: "Quiescent hydrostatic endpoints".into(),
                free_submerged_aperture_applicability:
                    "One analytical coefficient covers the aperture envelope".into(),
                unobstructed_aperture_assertion: "Both endpoints expose the declared aperture"
                    .into(),
                symmetric_reversal: true,
            },
        )?),
    })
}

pub fn structure_batch(
    case: &Case,
    width: usize,
) -> std::result::Result<StructureBatch<crate::runtime::Runtime>, StructureError> {
    StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        (0..width)
            .map(|lane| gate(&element_id(lane), case.maximum_opening))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    )
}

pub fn assignments(
    trajectory: &GateOpeningTrajectory,
    width: usize,
) -> Vec<GateTrajectoryAssignment> {
    (0..width)
        .map(|lane| GateTrajectoryAssignment {
            structure_id: element_id(lane),
            trajectory: trajectory.clone(),
        })
        .collect()
}

pub fn category(error: &StructureError) -> String {
    match error {
        StructureError::InvalidInput { kind, .. } => format!("{kind:?}"),
        StructureError::Numerical { .. } => "Numerical".into(),
        StructureError::Runtime { .. } => "Runtime".into(),
    }
}

pub fn check_error(error: &StructureError, expected: &str, element: Option<usize>) -> Result {
    if category(error) != expected || error.element() != element {
        return Err(format!("Expected {expected},element={element:?}; received {error}").into());
    }
    Ok(())
}

pub fn check_output(
    case: &Case,
    query: &Query,
    output: &GateTrajectoryOutput,
    lane: usize,
) -> Result {
    if output.element != lane
        || output.structure_id != element_id(lane)
        || output.time_s.to_bits() != query.time.to_bits()
        || Some(output.segment_index) != query.segment
        || output.exact_knot_index != query.knot
        || output.closed_by_definition != query.closed
    {
        return Err(format!(
            "{} lane={lane}: original trajectory metadata differs: {output:?}",
            case.id
        )
        .into());
    }
    if !output.opening_m.is_finite()
        || output.opening_m < 0.0
        || output.opening_m > case.maximum_opening
        || !output.absolute_error_m.is_finite()
        || output.absolute_error_m < 0.0
        || (query.positive && output.opening_m <= 0.0)
        || (query.closed && (output.opening_m != 0.0 || output.absolute_error_m != 0.0))
    {
        return Err(format!(
            "{} lane={lane}: represented opening support or receipt differs: {output:?}",
            case.id
        )
        .into());
    }
    if query.exact_bits
        && (output.opening_m.to_bits()
            != query
                .reference
                .ok_or("Missing exact opening reference")?
                .to_bits()
            || output.absolute_error_m != 0.0)
    {
        return Err(format!(
            "{} lane={lane}: exact original knot or hold differs",
            case.id
        )
        .into());
    }
    Ok(())
}
