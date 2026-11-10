use super::inputs::{Case, Decision};
use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const LAYOUTS: [usize; 6] = [1, 31, 32, 33, 64, 65];
pub const DATUM: &str = "Analytical controlled gate datum";
fn controller(lane: usize) -> String {
    format!("controller-{lane}")
}
fn actuator(lane: usize) -> String {
    format!("actuator-{lane}")
}
fn device(lane: usize) -> String {
    format!("device-{lane}")
}
fn sample(d: &Decision, lane: usize) -> ControlSample {
    ControlSample {
        controller_id: controller(lane),
        signal_id: format!("signal-{lane}"),
        support_id: format!("support-{lane}"),
        datum: DATUM.into(),
        observed_at_s: d.time,
        available_at_s: d.available,
        stage_m: d.stage,
    }
}
pub fn samples(case: &Case, width: usize, time: f64) -> Vec<ControlSample> {
    case.decisions
        .iter()
        .find(|d| d.time.to_bits() == time.to_bits())
        .map_or_else(Vec::new, |d| {
            (0..width).map(|lane| sample(d, lane)).collect()
        })
}
pub fn build(
    case: &Case,
    width: usize,
) -> std::result::Result<ControlledGateBatch<crate::runtime::Runtime>, StructureError> {
    let elements = (0..width)
        .map(|lane| {
            Ok(StructureElement {
                id: device(lane),
                endpoint_a: "reference-a".into(),
                endpoint_b: "reference-b".into(),
                datum: DATUM.into(),
                direction: DirectionPolicy::Bidirectional,
                law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                    1.0,
                    0.0,
                    1.0,
                    0.6,
                    9.81,
                    GateApplicability {
                        coefficient_provenance: "Analytical composition binding only".into(),
                        hydrostatic_quiescent_pool_assertion: "Declared quiescent supports".into(),
                        free_submerged_aperture_applicability: "Complete declared aperture".into(),
                        unobstructed_aperture_assertion: "Unobstructed aperture".into(),
                        symmetric_reversal: true,
                    },
                )?),
            })
        })
        .collect::<std::result::Result<Vec<_>, StructureError>>()?;
    let structures =
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), elements)?;
    let times = case.decisions.iter().map(|d| d.time).collect::<Vec<_>>();
    let controls = structures.with_operating_controls(
        (0..width)
            .map(|lane| OperatingControlDefinition {
                controller_id: controller(lane),
                structure_id: device(lane),
                signal: ControlStageSignal {
                    signal_id: format!("signal-{lane}"),
                    support_id: format!("support-{lane}"),
                    datum: DATUM.into(),
                },
                lower_stage_m: case.lower_stage,
                upper_stage_m: case.upper_stage,
                high_qualification_delay_s: case.high_delay,
                low_qualification_delay_s: case.low_delay,
                low_command: ControlCommand::GateTarget {
                    opening_m: case.low,
                },
                high_command: ControlCommand::GateTarget {
                    opening_m: case.high,
                },
                provenance: format!("Original-bit composition fixture {}", case.id),
            })
            .collect(),
        (0..width)
            .map(|lane| ControlInitialState {
                controller_id: controller(lane),
                latch: ControlLatch::Low,
                pending: None,
                antecedent_sample: None,
                last_transition_at_s: None,
            })
            .collect(),
        ControlDecisionSchedule {
            initial_time_s: case.start,
            decision_times_s: times.clone(),
            end_s: case.end,
        },
        SampledControlTiming {
            maximum_decision_gap_s: case.gap,
            maximum_timer_lateness_s: case.lateness,
            sampled_applicability:
                "Declared sampled qualification only. No continuous trigger claim".into(),
        },
    )?;
    let actuators = structures.with_gate_actuators(
        (0..width)
            .map(|lane| GateActuatorDefinition {
                actuator_id: actuator(lane),
                structure_id: device(lane),
                opening_rate_m_per_s: case.opening_rate,
                closing_rate_m_per_s: case.closing_rate,
                accuracy: case.accuracy,
                rate_provenance: "Original analytical directional rates".into(),
                ideal_response_applicability:
                    "Declared ideal finite-rate response without motor or hydraulic claims".into(),
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
            target_times_s: times,
            end_s: case.end,
        },
    )?;
    ControlledGateBatch::<crate::runtime::Runtime>::new(controls, actuators)
}
pub fn category(error: &StructureError) -> String {
    match error {
        StructureError::InvalidInput { kind, .. } => format!("{kind:?}"),
        StructureError::Numerical { .. } => "Numerical".into(),
        StructureError::Runtime { .. } => "Runtime".into(),
    }
}
