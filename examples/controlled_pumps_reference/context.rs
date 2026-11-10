use super::inputs::{Admission, Case, Decision, Support};
use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const LAYOUTS: [usize; 6] = [1, 31, 32, 33, 64, 65];
pub const DATUM: &str = "Analytical controlled pump datum";
pub const SOURCE: &str = "controlled-pump-source";
fn controller(lane: usize) -> String {
    format!("controller-{lane}")
}
fn device(lane: usize) -> String {
    format!("device-{lane}")
}
fn sampled(d: &Decision, lane: usize) -> ControlSample {
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
            (0..width).map(|lane| sampled(d, lane)).collect()
        })
}
pub fn build(
    case: &Case,
    width: usize,
) -> std::result::Result<
    (
        StructureBatch<crate::runtime::Runtime>,
        ControlledPumpBatch<crate::runtime::Runtime>,
    ),
    StructureError,
> {
    let elements = (0..width)
        .map(|lane| {
            Ok(StructureElement {
                id: device(lane),
                endpoint_a: "reference-a".into(),
                endpoint_b: "reference-b".into(),
                datum: DATUM.into(),
                direction: DirectionPolicy::AtoBOnly,
                law: StructureLaw::Pump(PumpParameters::new(
                    PumpCurve::new(
                        vec![
                            PumpCurveKnot {
                                head_m: 0.0,
                                discharge_m3_per_s: 4.0,
                            },
                            PumpCurveKnot {
                                head_m: 4.0,
                                discharge_m3_per_s: 0.0,
                            },
                        ],
                        "Analytical SI endpoint-stage curve Q0(H)=4-H on the closed original domain 0 through 4 m."
                            .into(),
                    )?,
                    0.0,
                    if case.affinity {
                        PumpSpeedPolicy::DeclaredAffinity {
                            minimum_ratio: 0.5,
                            maximum_ratio: 2.0,
                            applicability:
                                "Declared unchanged analytical pump geometry and reference fluid for speed ratios 0.5 through 2."
                                    .into(),
                        }
                    } else {
                        PumpSpeedPolicy::ReferenceOnly
                    },
                    PumpApplicability {
                        endpoint_stage_head_assertion:
                            "Declared quiescent endpoint-stage relation".into(),
                        connection_losses_and_velocity_head_account:
                            "Includes analytical connection and velocity-head accounting".into(),
                        stopped_isolation_assertion:
                            "Explicitly stopped pump excludes forward siphoning and reverse flow."
                                .into(),
                        operating_envelope:
                            "Analytical covered endpoint heads and original speed policy only".into(),
                    },
                )?),
            })
        })
        .collect::<std::result::Result<Vec<_>, StructureError>>()?;
    let structures =
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), elements)?;
    let mut controls = structures.with_operating_controls(
        (0..width)
            .map(|lane| OperatingControlDefinition {
                controller_id: controller(lane),
                structure_id: device(lane),
                signal: ControlStageSignal {
                    signal_id: format!("signal-{lane}"),
                    support_id: format!("support-{lane}"),
                    datum: DATUM.into(),
                },
                lower_stage_m: 1.0,
                upper_stage_m: 2.0,
                high_qualification_delay_s: case.high_delay,
                low_qualification_delay_s: case.low_delay,
                low_command: case.low,
                high_command: case.high,
                provenance: format!("Original-bit sampled pump fixture {}", case.id),
            })
            .collect(),
        (0..width)
            .map(|lane| ControlInitialState {
                controller_id: controller(lane),
                latch: case.initial_latch,
                pending: None,
                antecedent_sample: None,
                last_transition_at_s: None,
            })
            .collect(),
        ControlDecisionSchedule {
            initial_time_s: 0.0,
            decision_times_s: case.decisions.iter().map(|d| d.time).collect(),
            end_s: case.end,
        },
        SampledControlTiming {
            maximum_decision_gap_s: case.gap,
            maximum_timer_lateness_s: 0.0,
            sampled_applicability:
                "Prescribed independent stage samples only. No endogenous feedback or between-sample claim"
                    .into(),
        },
    )?;
    for decision in case.decisions.iter().filter(|d| d.time <= case.adoption) {
        controls = controls
            .trial_to(decision.time, &samples(case, width, decision.time))?
            .into_next();
    }
    let pump = ControlledPumpBatch::<crate::runtime::Runtime>::new(
        controls,
        (0..width)
            .map(|lane| PumpResponseDefinition {
                structure_id: device(lane),
                model: PumpResponseModel::IdealInstantaneous,
                response_provenance:
                    "Explicit analytical memoryless command-operation mapping".into(),
                ideal_response_applicability:
                    "Commanded stopped/running operation applies immediately on the sampled decision right side. No acceleration, startup, or coast-down model."
                        .into(),
            })
            .collect(),
    )?;
    Ok((structures, pump))
}
pub fn admission_case(point: &Admission) -> Case {
    const QUERIES: &[f64] = &[0.0];
    Case {
        id: point.id,
        end: 1.0,
        initial_latch: ControlLatch::Low,
        low: point.operation,
        high: point.operation,
        high_delay: 0.0,
        low_delay: 0.0,
        gap: 1.0,
        affinity: true,
        adoption: 0.0,
        decisions: &[],
        queries: QUERIES,
    }
}
pub fn heads(width: usize, donor: Support, receiver: Support) -> Vec<StructureHeads> {
    fn support(input: Support) -> HeadSupport {
        match input {
            Support::Wet(stage_m) => HeadSupport::Wet { stage_m },
            Support::Unknown => HeadSupport::Unknown,
            Support::MalformedKnownNaN => HeadSupport::Wet {
                stage_m: f64::from_bits(0x7ff8000000000000),
            },
        }
    }
    (0..width)
        .map(|lane| StructureHeads {
            structure_id: device(lane),
            endpoint_a: EndpointHead {
                id: "reference-a".into(),
                support: support(donor),
            },
            endpoint_b: EndpointHead {
                id: "reference-b".into(),
                support: support(receiver),
            },
            datum: DATUM.into(),
            convention: HeadConvention::QuiescentStage,
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
