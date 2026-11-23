//! Analytical gates, pumps, and their operating children for two-phase interval tests.

use crate::structures::*;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(crate) type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
pub(crate) type Cpu = CpuRuntime;

const DATUM: &str = "Analytical interval datum";
/// Decision and target times shared by every operating child.
pub(crate) const TIMES: [f64; 3] = [1.0, 2.0, 3.0];
/// Common terminal bound.
pub(crate) const END: f64 = 4.0;

/// Debug text: equal text means equal bits for every finite field.
pub(crate) fn bits<T: std::fmt::Debug>(value: &T) -> String {
    format!("{value:?}")
}

fn text(value: &str) -> String {
    value.into()
}

pub(crate) fn gate(id: &str) -> TestResult<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "pool".into(),
        endpoint_b: "river".into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            1.0,
            0.0,
            1.0,
            0.6,
            9.81,
            GateApplicability {
                coefficient_provenance: text("Analytical aperture"),
                hydrostatic_quiescent_pool_assertion: text("Quiescent analytical endpoints"),
                free_submerged_aperture_applicability: text("Declared aperture envelope"),
                unobstructed_aperture_assertion: text("Both connections expose the aperture"),
                symmetric_reversal: true,
            },
        )?),
        condition: StructureCondition::AsBuilt,
    })
}

pub(crate) fn pump(id: &str) -> TestResult<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "sump".into(),
        endpoint_b: "river".into(),
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
                        head_m: 8.0,
                        discharge_m3_per_s: 0.0,
                    },
                ],
                text("Analytical reference-speed curve"),
            )?,
            -2.0,
            PumpSpeedPolicy::DeclaredAffinity {
                minimum_ratio: 0.0625,
                maximum_ratio: 2.0,
                applicability: text("Declared analytical speed envelope"),
            },
            PumpApplicability {
                endpoint_stage_head_assertion: text("Quiescent analytical endpoint stages"),
                connection_losses_and_velocity_head_account: text("Curve includes losses"),
                stopped_isolation_assertion: text("Stopped device blocks reverse flow"),
                operating_envelope: text("Analytical fixture"),
            },
        )?),
        condition: StructureCondition::AsBuilt,
    })
}

/// Elements `g0` prescribed, `g1` actuated, `p2` controlled pump, `g3` manual.
pub(crate) fn context() -> TestResult<StructureBatch<Cpu>> {
    Ok(StructureBatch::<Cpu>::new(
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
        vec![gate("g0")?, gate("g1")?, pump("p2")?, gate("g3")?],
    )?)
}

pub(crate) fn controller(id: &str) -> String {
    format!("relay-{id}")
}

fn controls(
    context: &StructureBatch<Cpu>,
    id: &str,
    commands: [ControlCommand; 2],
    latch: ControlLatch,
) -> TestResult<OperatingControlBatch<Cpu>> {
    Ok(context.with_operating_controls(
        vec![OperatingControlDefinition {
            controller_id: controller(id),
            structure_id: id.into(),
            signal: ControlStageSignal {
                signal_id: format!("signal-{id}"),
                support_id: format!("support-{id}"),
                datum: DATUM.into(),
            },
            lower_stage_m: 1.0,
            upper_stage_m: 2.0,
            high_qualification_delay_s: 0.0,
            low_qualification_delay_s: 0.0,
            low_command: commands[0],
            high_command: commands[1],
            provenance: text("Analytical sampled relay"),
        }],
        vec![ControlInitialState {
            controller_id: controller(id),
            latch,
            pending: None,
            antecedent_sample: None,
            last_transition_at_s: None,
        }],
        ControlDecisionSchedule {
            initial_time_s: 0.0,
            decision_times_s: TIMES.to_vec(),
            end_s: END,
        },
        SampledControlTiming {
            maximum_decision_gap_s: END,
            maximum_timer_lateness_s: END,
            sampled_applicability: text("Analytical sampled policy"),
        },
    )?)
}

/// Actuator on `id`, opening at 0.25 m/s and closing at 0.5 m/s.
pub(crate) fn actuators(
    context: &StructureBatch<Cpu>,
    id: &str,
    opening_m: f64,
    target_opening_m: f64,
) -> TestResult<GateActuatorBatch<Cpu>> {
    actuators_at(context, id, (0.25, 0.5), (opening_m, target_opening_m))
}

/// Actuator on `id` opening and closing at `rates`, in m/s, from `(opening, target)`, in m.
pub(crate) fn actuators_at(
    context: &StructureBatch<Cpu>,
    id: &str,
    (opening_rate_m_per_s, closing_rate_m_per_s): (f64, f64),
    (opening_m, target_opening_m): (f64, f64),
) -> TestResult<GateActuatorBatch<Cpu>> {
    Ok(context.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: format!("actuator-{id}"),
            structure_id: id.into(),
            opening_rate_m_per_s,
            closing_rate_m_per_s,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: text("Analytical rates"),
            ideal_response_applicability: text("Ideal rate response"),
        }],
        vec![GateActuatorInitialState {
            actuator_id: format!("actuator-{id}"),
            opening_m,
            target_opening_m,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: TIMES.to_vec(),
            end_s: END,
        },
    )?)
}

/// Controlled gate on `g1`: low target 0 m, high target 1 m, starting closed and low.
pub(crate) fn controlled_gate(
    context: &StructureBatch<Cpu>,
) -> TestResult<ControlledGateBatch<Cpu>> {
    Ok(ControlledGateBatch::new(
        controls(
            context,
            "g1",
            [
                ControlCommand::GateTarget { opening_m: 0.0 },
                ControlCommand::GateTarget { opening_m: 1.0 },
            ],
            ControlLatch::Low,
        )?,
        actuators(context, "g1", 0.0, 0.0)?,
    )?)
}

/// Controlled pump on `p2`: stopped when low, running at full speed when high, starting low.
pub(crate) fn controlled_pump(
    context: &StructureBatch<Cpu>,
) -> TestResult<ControlledPumpBatch<Cpu>> {
    Ok(ControlledPumpBatch::new(
        controls(
            context,
            "p2",
            [
                ControlCommand::PumpStoppedIsolated,
                ControlCommand::PumpRunning { speed_ratio: 1.0 },
            ],
            ControlLatch::Low,
        )?,
        vec![PumpResponseDefinition {
            structure_id: "p2".into(),
            model: PumpResponseModel::IdealInstantaneous,
            response_provenance: text("Analytical instantaneous operation"),
            ideal_response_applicability: text("No startup delay"),
        }],
    )?)
}

/// Prescribed linear trajectory on `id` through `knots`.
pub(crate) fn trajectory(
    context: &StructureBatch<Cpu>,
    id: &str,
    knots: &[(f64, f64)],
) -> TestResult<GateTrajectoryBatch<Cpu>> {
    let trajectory = GateOpeningTrajectory::new(
        knots
            .iter()
            .map(|&(time_s, opening_m)| GateOpeningKnot { time_s, opening_m })
            .collect(),
        GateMotionLimits {
            maximum_opening_speed_m_per_s: f64::MAX,
            maximum_closing_speed_m_per_s: f64::MAX,
        },
        GateTrajectoryAccuracy {
            maximum_absolute_opening_error_m: 1e-12,
            maximum_relative_opening_error: 1e-12,
        },
        text("Analytical linear trajectory"),
    )?;
    Ok(
        context.with_gate_trajectory_subset(vec![GateTrajectoryAssignment {
            structure_id: id.into(),
            trajectory,
        }])?,
    )
}

/// One decision sample for the relay on `id`.
pub(crate) fn samples(id: &str, time_s: f64, stage_m: f64) -> Vec<ControlSample> {
    vec![ControlSample {
        controller_id: controller(id),
        signal_id: format!("signal-{id}"),
        support_id: format!("support-{id}"),
        datum: DATUM.into(),
        observed_at_s: time_s,
        available_at_s: time_s,
        stage_m,
    }]
}

/// Heads per element: gates drain pool to river, the pump lifts from sump to river.
pub(crate) fn heads(context: &StructureBatch<Cpu>, support: HeadSupport) -> Vec<StructureHeads> {
    context
        .elements()
        .iter()
        .map(|element| {
            let pumped = matches!(element.law, StructureLaw::Pump(_));
            let (a, b) = if pumped { (1.0, 2.0) } else { (2.0, 1.0) };
            let wet = |stage_m| match &support {
                HeadSupport::Wet { .. } => HeadSupport::Wet { stage_m },
                other => other.clone(),
            };
            StructureHeads {
                structure_id: element.id.clone(),
                endpoint_a: EndpointHead {
                    id: element.endpoint_a.clone(),
                    support: if pumped {
                        HeadSupport::Wet { stage_m: a }
                    } else {
                        wet(a)
                    },
                },
                endpoint_b: EndpointHead {
                    id: element.endpoint_b.clone(),
                    support: if pumped {
                        HeadSupport::Wet { stage_m: b }
                    } else {
                        wet(b)
                    },
                },
                datum: element.datum.clone(),
                convention: HeadConvention::QuiescentStage,
            }
        })
        .collect()
}
