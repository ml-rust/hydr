//! Sampled targets and actual finite-rate response accepted as one immutable pair.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

use hydr::structures::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, ControlLatch, ControlSample,
    ControlStageSignal, ControlledGateBatch, DirectionPolicy, GateActuatorAccuracy,
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule, GateApplicability,
    GateParameters, OperatingControlDefinition, SampledControlTiming, StructureBatch,
    StructureElement, StructureLaw,
};

const DATUM: &str = "synthetic common datum";

fn model() -> Result<ControlledGateBatch<crate::runtime::Runtime>, Box<dyn std::error::Error>> {
    let structures = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![StructureElement {
            id: "gate".into(),
            endpoint_a: "upstream".into(),
            endpoint_b: "downstream".into(),
            datum: DATUM.into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                1.0,
                0.0,
                1.0,
                0.6,
                9.81,
                GateApplicability {
                    coefficient_provenance: "explicit synthetic Cd=0.6 without field calibration"
                        .into(),
                    hydrostatic_quiescent_pool_assertion: "synthetic quiescent endpoint pools"
                        .into(),
                    free_submerged_aperture_applicability:
                        "one synthetic coefficient covers the declared aperture envelope".into(),
                    unobstructed_aperture_assertion:
                        "both synthetic connections expose the complete aperture".into(),
                    symmetric_reversal: true,
                },
            )?),
        }],
    )?;
    let decision_times = vec![1.0, 2.0, 3.0, 4.0];
    let controls = structures.with_operating_controls(
        vec![OperatingControlDefinition {
            controller_id: "relay".into(),
            structure_id: "gate".into(),
            signal: ControlStageSignal {
                signal_id: "level".into(),
                support_id: "external-level-support".into(),
                datum: DATUM.into(),
            },
            lower_stage_m: 1.0,
            upper_stage_m: 2.0,
            high_qualification_delay_s: 0.0,
            low_qualification_delay_s: 0.0,
            low_command: ControlCommand::GateTarget { opening_m: 0.0 },
            high_command: ControlCommand::GateTarget { opening_m: 1.0 },
            provenance: "synthetic sampled relay with explicit zero qualification delays".into(),
        }],
        vec![ControlInitialState {
            controller_id: "relay".into(),
            latch: ControlLatch::Low,
            pending: None,
            antecedent_sample: None,
            last_transition_at_s: None,
        }],
        ControlDecisionSchedule {
            initial_time_s: 0.0,
            decision_times_s: decision_times.clone(),
            end_s: 5.0,
        },
        SampledControlTiming {
            maximum_decision_gap_s: 1.0,
            maximum_timer_lateness_s: 0.0,
            sampled_applicability:
                "synthetic one-second decisions without between-sample trigger claims".into(),
        },
    )?;
    let actuators = structures.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: "drive".into(),
            structure_id: "gate".into(),
            opening_rate_m_per_s: 0.25,
            closing_rate_m_per_s: 0.5,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "explicit synthetic asymmetric constant rates".into(),
            ideal_response_applicability:
                "ideal response without inertia, backlash, or load-dependent rates".into(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: "drive".into(),
            opening_m: 0.0,
            target_opening_m: 0.0,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: decision_times,
            end_s: 5.0,
        },
    )?;
    Ok(ControlledGateBatch::<crate::runtime::Runtime>::new(
        controls, actuators,
    )?)
}

fn print_state(
    batch: &ControlledGateBatch<crate::runtime::Runtime>,
) -> Result<(), Box<dyn std::error::Error>> {
    for (command, actual) in batch
        .controls()
        .commands()?
        .into_iter()
        .zip(batch.actuators().outputs()?)
    {
        let ControlCommand::GateTarget { opening_m } = command.command else {
            return Err("controlled gate command differs from its admitted gate target".into());
        };
        println!(
            "time_s={}, requested_target_m={}, actual_opening_m={}, support={:?}",
            batch.time(),
            opening_m,
            actual.position.opening_m,
            actual.support,
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = model()?;
    let mut accepted = initial.clone();
    let stages = [2.0, 2.0, 0.0, 2.0];
    print_state(&accepted)?;
    while let Some(event) = accepted.next_event()? {
        let samples = if let Some(index) = event.decision_index {
            vec![ControlSample {
                controller_id: "relay".into(),
                signal_id: "level".into(),
                support_id: "external-level-support".into(),
                datum: DATUM.into(),
                observed_at_s: event.time_s,
                available_at_s: event.time_s - 0.25,
                stage_m: *stages
                    .get(index)
                    .ok_or("decision index exceeds the original synthetic sample calendar")?,
            }]
        } else {
            Vec::new()
        };
        accepted = accepted.trial_to(event.time_s, &samples)?.into_next();
        print_state(&accepted)?;
    }
    let mut restored = initial;
    restored.restore_data(&accepted.snapshot_data()?)?;
    println!(
        "restored_time_s={}, accepted_steps={}",
        restored.time(),
        restored.accepted_steps(),
    );
    Ok(())
}
