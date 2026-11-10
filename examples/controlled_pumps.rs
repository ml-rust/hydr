//! Explicit instantaneous pump response with sampled qualification and point proposals.

use hydr::structures::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, ControlLatch, ControlSample,
    ControlStageSignal, ControlledPumpBatch, DirectionPolicy, EndpointHead, HeadConvention,
    HeadSupport, OperatingControlDefinition, OperatingSourceBinding, OperatingSourceFrame,
    PumpApplicability, PumpCurve, PumpCurveKnot, PumpParameters, PumpResponseDefinition,
    PumpResponseModel, PumpSpeedPolicy, SampledControlTiming, StructureBatch, StructureElement,
    StructureHeads, StructureLaw,
};

const DATUM: &str = "synthetic common datum";

fn model() -> Result<(StructureBatch, ControlledPumpBatch), Box<dyn std::error::Error>> {
    let structures = StructureBatch::new(vec![StructureElement {
        id: "pump".into(),
        endpoint_a: "donor-pool".into(),
        endpoint_b: "receiving-pool".into(),
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
                "explicit synthetic reference-speed curve without field calibration".into(),
            )?,
            0.0,
            PumpSpeedPolicy::DeclaredAffinity {
                minimum_ratio: 0.5,
                maximum_ratio: 2.0,
                applicability: "synthetic unchanged-geometry speed envelope".into(),
            },
            PumpApplicability {
                endpoint_stage_head_assertion: "synthetic quiescent endpoint stages".into(),
                connection_losses_and_velocity_head_account:
                    "synthetic curve includes connection losses".into(),
                stopped_isolation_assertion:
                    "stopped synthetic device blocks siphoning and reverse flow".into(),
                operating_envelope: "synthetic wet suction and receiving support".into(),
            },
        )?),
    }])?;
    let controls = structures.with_operating_controls(
        vec![OperatingControlDefinition {
            controller_id: "relay".into(),
            structure_id: "pump".into(),
            signal: ControlStageSignal {
                signal_id: "level".into(),
                support_id: "external-level-support".into(),
                datum: DATUM.into(),
            },
            lower_stage_m: 1.0,
            upper_stage_m: 2.0,
            high_qualification_delay_s: 1.0,
            low_qualification_delay_s: 0.0,
            low_command: ControlCommand::PumpStoppedIsolated,
            high_command: ControlCommand::PumpRunning { speed_ratio: 1.0 },
            provenance: "synthetic sampled relay with one-second high qualification".into(),
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
            decision_times_s: vec![1.0, 2.0, 3.0],
            end_s: 4.0,
        },
        SampledControlTiming {
            maximum_decision_gap_s: 1.0,
            maximum_timer_lateness_s: 0.0,
            sampled_applicability:
                "synthetic one-second decisions without continuous trigger claims".into(),
        },
    )?;
    let pump = ControlledPumpBatch::new(
        controls,
        vec![PumpResponseDefinition {
            structure_id: "pump".into(),
            model: PumpResponseModel::IdealInstantaneous,
            response_provenance: "explicit synthetic instantaneous operation assumption".into(),
            ideal_response_applicability:
                "no acceleration, startup delay, or coast-down within the synthetic envelope".into(),
        }],
    )?;
    Ok((structures, pump))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (structures, initial) = model()?;
    let registry =
        structures.with_operating_sources(vec![OperatingSourceBinding::ControlledPump {
            source_id: "sampled-pump".into(),
            pump: &initial,
        }])?;
    let heads = [StructureHeads {
        structure_id: "pump".into(),
        endpoint_a: EndpointHead {
            id: "donor-pool".into(),
            support: HeadSupport::Wet { stage_m: 2.0 },
        },
        endpoint_b: EndpointHead {
            id: "receiving-pool".into(),
            support: HeadSupport::Wet { stage_m: 3.0 },
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }];
    let mut accepted = initial.clone();
    loop {
        let proposal = registry.propose_at(
            accepted.time(),
            &heads,
            &[],
            &[OperatingSourceFrame::ControlledPump {
                source_id: "sampled-pump",
                pump: &accepted,
            }],
        )?;
        for output in proposal.outputs()? {
            println!(
                "time_s={}, actual_operation={:?}, proposed_discharge_m3_per_s={}",
                accepted.time(),
                output.hydraulic.operation,
                output.hydraulic.discharge_m3_per_s
            );
        }
        let Some(event) = accepted.next_event() else {
            break;
        };
        let observations = if let Some(index) = event.decision_index {
            let stages = [2.0, 2.0, 1.0];
            vec![ControlSample {
                controller_id: "relay".into(),
                signal_id: "level".into(),
                support_id: "external-level-support".into(),
                datum: DATUM.into(),
                observed_at_s: event.time_s,
                available_at_s: event.time_s - 0.25,
                stage_m: *stages
                    .get(index)
                    .ok_or("decision exceeds the supplied synthetic sample calendar")?,
            }]
        } else {
            Vec::new()
        };
        accepted = accepted.trial_to(event.time_s, &observations)?.into_next();
    }
    let mut restored = initial;
    restored.restore_data(&accepted.snapshot_data()?)?;
    println!(
        "restored_time_s={}, accepted_control_steps={}",
        restored.time(),
        restored.accepted_steps()
    );
    Ok(())
}
