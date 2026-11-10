use super::reference_inputs::{Case, Sample};
use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const LAYOUTS: [usize; 6] = [1, 31, 32, 33, 64, 65];
pub const DATUM: &str = "Analytical control datum";

pub fn controller(lane: usize) -> String {
    format!("controller-{lane}")
}
pub fn device(lane: usize) -> String {
    format!("device-{lane}")
}

pub fn sample(input: Sample, lane: usize) -> ControlSample {
    ControlSample {
        controller_id: controller(lane),
        signal_id: format!("signal-{lane}"),
        support_id: format!("support-{lane}"),
        datum: DATUM.into(),
        observed_at_s: input.observed,
        available_at_s: input.available,
        stage_m: input.stage,
    }
}

fn pump_with_policy(
    id: &str,
    speed_policy: PumpSpeedPolicy,
) -> std::result::Result<StructureElement, StructureError> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "reference-a".into(),
        endpoint_b: "reference-b".into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::AtoBOnly,
        law: StructureLaw::Pump(PumpParameters::new(
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
                "Analytical command fixture curve".into(),
            )?,
            -1.0,
            speed_policy,
            PumpApplicability {
                endpoint_stage_head_assertion: "Quiescent analytical endpoints".into(),
                connection_losses_and_velocity_head_account:
                    "Analytical curve includes declared losses".into(),
                stopped_isolation_assertion:
                    "Stopped isolation excludes siphoning and reverse flow".into(),
                operating_envelope:
                    "Commands only. No physical response or hydraulic consumption claim".into(),
            },
        )?),
    })
}

pub fn pump(id: &str) -> std::result::Result<StructureElement, StructureError> {
    pump_with_policy(id, PumpSpeedPolicy::ReferenceOnly)
}

pub fn gate(id: &str) -> std::result::Result<StructureElement, StructureError> {
    Ok(StructureElement {
        id: id.into(),
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
                coefficient_provenance: "Synthetic command fixture".into(),
                hydrostatic_quiescent_pool_assertion: "Declared quiescent analytical endpoints"
                    .into(),
                free_submerged_aperture_applicability:
                    "Declared complete aperture command envelope".into(),
                unobstructed_aperture_assertion: "Both connections expose the complete aperture"
                    .into(),
                symmetric_reversal: true,
            },
        )?),
    })
}

pub fn definition(case: &Case, lane: usize, structure_id: &str) -> OperatingControlDefinition {
    OperatingControlDefinition {
        controller_id: controller(lane),
        structure_id: structure_id.into(),
        signal: ControlStageSignal {
            signal_id: format!("signal-{lane}"),
            support_id: format!("support-{lane}"),
            datum: DATUM.into(),
        },
        lower_stage_m: case.lower,
        upper_stage_m: case.upper,
        high_qualification_delay_s: case.high_delay,
        low_qualification_delay_s: case.low_delay,
        low_command: case.low_command,
        high_command: case.high_command,
        provenance: format!("Original-bit sampled control fixture {}", case.id),
    }
}

pub fn initial(case: &Case, lane: usize) -> ControlInitialState {
    ControlInitialState {
        controller_id: controller(lane),
        latch: case.initial_latch,
        pending: case.initial_pending,
        antecedent_sample: case.antecedent.map(|input| sample(input, lane)),
        last_transition_at_s: case.initial_transition,
    }
}

pub fn schedule(case: &Case) -> ControlDecisionSchedule {
    ControlDecisionSchedule {
        initial_time_s: case.start,
        decision_times_s: case.decision_times.to_vec(),
        end_s: case.end,
    }
}

pub fn timing(case: &Case) -> SampledControlTiming {
    SampledControlTiming {
        maximum_decision_gap_s: case.maximum_gap,
        maximum_timer_lateness_s: case.lateness,
        sampled_applicability:
            "Declared sampled-only detection and cadence. No between-sample trigger completeness"
                .into(),
    }
}

pub fn build(
    case: &Case,
    width: usize,
) -> std::result::Result<OperatingControlBatch, StructureError> {
    let elements=(0..width).map(|lane|if matches!(case.low_command,ControlCommand::GateTarget {..}) {
        gate(&device(lane))
    } else if case.id=="explicit_initial_high_reversed_commands" {
        pump_with_policy(&device(lane),PumpSpeedPolicy::DeclaredAffinity {
            minimum_ratio:0.5,maximum_ratio:1.0,
            applicability:"Synthetic unchanged-geometry half-speed and reference-speed command envelope".into(),
        })
    } else {pump(&device(lane))}).collect::<std::result::Result<Vec<_>,_>>()?;
    StructureBatch::new(elements)?.with_operating_controls(
        (0..width)
            .map(|lane| definition(case, lane, &device(lane)))
            .collect(),
        (0..width).map(|lane| initial(case, lane)).collect(),
        schedule(case),
        timing(case),
    )
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
        return Err(format!("Expected {expected},element={element:?}. Received {error}").into());
    }
    Ok(())
}

pub fn selected(case: &Case, latch: ControlLatch) -> ControlCommand {
    match latch {
        ControlLatch::Low => case.low_command,
        ControlLatch::High => case.high_command,
    }
}
