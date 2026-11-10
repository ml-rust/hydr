use super::inputs::Case;
use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const LAYOUTS: [usize; 6] = [1, 31, 32, 33, 64, 65];

pub fn actuator(lane: usize) -> String {
    format!("actuator-{lane}")
}
pub fn device(lane: usize) -> String {
    format!("device-{lane}")
}

pub fn build(case: &Case, width: usize) -> std::result::Result<GateActuatorBatch, StructureError> {
    let elements = (0..width)
        .map(|lane| {
            Ok(StructureElement {
                id: device(lane),
                endpoint_a: "reference-a".into(),
                endpoint_b: "reference-b".into(),
                datum: "Analytical actuator datum".into(),
                direction: DirectionPolicy::Bidirectional,
                law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                    1.0,
                    0.0,
                    case.maximum,
                    0.6,
                    9.81,
                    GateApplicability {
                        coefficient_provenance: "Synthetic actuator binding only".into(),
                        hydrostatic_quiescent_pool_assertion:
                            "Declared analytical endpoint support".into(),
                        free_submerged_aperture_applicability: "Complete declared aperture".into(),
                        unobstructed_aperture_assertion: "Complete aperture remains unobstructed"
                            .into(),
                        symmetric_reversal: true,
                    },
                )?),
            })
        })
        .collect::<std::result::Result<Vec<_>, StructureError>>()?;
    StructureBatch::new(elements)?.with_gate_actuators(
        (0..width)
            .map(|lane| GateActuatorDefinition {
                actuator_id: actuator(lane),
                structure_id: device(lane),
                opening_rate_m_per_s: case.opening_rate,
                closing_rate_m_per_s: case.closing_rate,
                accuracy: case.accuracy,
                rate_provenance: format!("Original-bit physical reference {}", case.id),
                ideal_response_applicability:
                    "Declared ideal capped constant-rate response. No motor or hydraulic claim"
                        .into(),
            })
            .collect(),
        (0..width)
            .map(|lane| GateActuatorInitialState {
                actuator_id: actuator(lane),
                opening_m: case.initial,
                target_opening_m: case.target,
            })
            .collect(),
        GateActuatorSchedule {
            initial_time_s: case.start,
            target_times_s: case.updates.iter().map(|u| u.time).collect(),
            end_s: case.end,
        },
    )
}

pub fn updates(case: &Case, width: usize, time: f64) -> Vec<GateTargetUpdate> {
    case.updates
        .iter()
        .find(|update| update.time.to_bits() == time.to_bits())
        .map_or_else(Vec::new, |update| {
            (0..width)
                .map(|lane| GateTargetUpdate {
                    actuator_id: actuator(lane),
                    structure_id: device(lane),
                    target_opening_m: update.target,
                    issued_at_s: update.time,
                    available_at_s: update.available,
                })
                .collect()
        })
}

pub fn category(error: &StructureError) -> String {
    match error {
        StructureError::InvalidInput { kind, .. } => format!("{kind:?}"),
        StructureError::Numerical { .. } => "Numerical".into(),
        StructureError::Runtime { .. } => "Runtime".into(),
    }
}
