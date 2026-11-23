//! Condition admission, the blocked gate range at every opening site, and packages that carry
//! a blocked gate.

use super::fixture::{
    EVIDENCE, Result, Runtime, batch, blocked, blocked_gate, closed, element, gate, operation,
    packet, pump, rejected, weir, wet,
};
use hydr::structures::*;

/// Blocked limit `m_e = 1 m · (1 - 0.5)` of the blocked test gate.
const LIMIT: f64 = 0.5;

fn blocked_gates() -> Result<Vec<StructureElement>> {
    Ok(vec![element("g", gate()?, blocked_gate(0.25, 0.5)?)])
}

#[test]
fn fractions_outside_the_half_open_unit_reject() {
    for share in [1.0, 1.5, -0.1, f64::NAN] {
        let message = rejected(BlockageFraction::new(share)).unwrap_or_default();
        assert!(
            message.contains("blocked_share=") && message.contains("StructureCondition::Closed"),
            "share={share}: {message}"
        );
    }
}

#[test]
fn conditions_that_do_not_fit_the_law_reject() -> Result {
    let tiny = StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
        f64::MIN_POSITIVE,
        1.0,
        1.7,
        WeirApplicability {
            coefficient_provenance: EVIDENCE.into(),
            free_nappe_and_approach_assertion: EVIDENCE.into(),
            submergence_correction_assertion: EVIDENCE.into(),
            symmetric_reversal: true,
        },
    )?);
    let cases = [
        (
            element("w", tiny.clone(), blocked(&tiny, 0.5)?),
            "crest_length_m=",
        ),
        (
            element("w", weir()?, blocked_gate(0.5, 0.0)?),
            "a Gate blockage cannot apply to a SuppressedRectangularWeir law. Supply StructureBlockage::weir",
        ),
        (
            element("p", pump()?, blocked(&weir()?, 0.5)?),
            "a pump curve has no geometry to block",
        ),
        (
            element("p", pump()?, closed()),
            "a pump curve has no geometry to block",
        ),
        (
            element(
                "w",
                weir()?,
                StructureCondition::Closed {
                    evidence: String::new(),
                },
            ),
            "condition.evidence",
        ),
        (element("c", culvert()?, closed()), "not yet available"),
    ];
    for (element, fragment) in cases {
        let message = rejected(batch(vec![element.clone()]))?;
        assert!(message.contains(fragment), "{element:?}: {message}");
        assert!(message.contains("element=Some(0)"), "{message}");
    }
    let empty = rejected(StructureBlockage::weir(BlockageFraction::NONE, " ".into()))?;
    assert!(empty.contains("blockage.applicability_evidence"), "{empty}");
    Ok(())
}

fn culvert() -> Result<StructureLaw> {
    Ok(StructureLaw::Culvert(CulvertParameters::new(
        CulvertBarrel::circular(1.0)?,
        CulvertAlignment {
            length_m: 20.0,
            invert_a_m: 0.1,
            invert_b_m: 0.0,
        },
        0.013,
        1,
        CulvertEnds {
            a: Some(CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?),
            b: Some(CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?),
        },
        9.81,
        CulvertApplicability {
            coefficient_provenance: EVIDENCE.into(),
            barrel_condition_assertion: EVIDENCE.into(),
        },
    )?))
}

#[test]
fn manual_openings_above_the_blocked_limit_reject() -> Result {
    let structures = batch(blocked_gates()?)?;
    let heads = [packet("g", wet(2.0), wet(1.0))];
    let element = &structures.elements()[0];
    structures.propose_with_operations(&heads, &[operation(element, LIMIT)])?;
    let message = rejected(
        structures.propose_with_operations(&heads, &[operation(element, LIMIT.next_up())]),
    )?;
    assert!(message.contains("maximum_opening_m=0.5"), "{message}");
    let shut = batch(vec![super::fixture::element("g", gate()?, closed())])?;
    shut.propose_with_operations(&heads, &[operation(&shut.elements()[0], 1.0)])?;
    Ok(())
}

fn trajectory(opening_m: f64) -> Result<Vec<GateTrajectoryAssignment>> {
    Ok(vec![GateTrajectoryAssignment {
        structure_id: "g".into(),
        trajectory: GateOpeningTrajectory::new(
            vec![
                GateOpeningKnot {
                    time_s: 0.0,
                    opening_m: 0.0,
                },
                GateOpeningKnot {
                    time_s: 1.0,
                    opening_m,
                },
            ],
            GateMotionLimits {
                maximum_opening_speed_m_per_s: f64::MAX,
                maximum_closing_speed_m_per_s: f64::MAX,
            },
            GateTrajectoryAccuracy {
                maximum_absolute_opening_error_m: 0.0,
                maximum_relative_opening_error: 0.0,
            },
            EVIDENCE.into(),
        )?,
    }])
}

#[test]
fn trajectory_knots_above_the_blocked_limit_reject() -> Result {
    let structures = batch(blocked_gates()?)?;
    structures.with_gate_trajectories(trajectory(LIMIT)?)?;
    let message = rejected(structures.with_gate_trajectories(trajectory(LIMIT.next_up())?))?;
    assert!(message.contains("maximum=0.5"), "{message}");
    Ok(())
}

fn actuators(
    structures: &StructureBatch<Runtime>,
    target: f64,
) -> std::result::Result<GateActuatorBatch<Runtime>, StructureError> {
    structures.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: "act".into(),
            structure_id: "g".into(),
            opening_rate_m_per_s: 1.0,
            closing_rate_m_per_s: 1.0,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-12,
            },
            rate_provenance: EVIDENCE.into(),
            ideal_response_applicability: EVIDENCE.into(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: "act".into(),
            opening_m: 0.0,
            target_opening_m: target,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: vec![1.0],
            end_s: 4.0,
        },
    )
}

fn update(target_opening_m: f64) -> GateTargetUpdate {
    GateTargetUpdate {
        actuator_id: "act".into(),
        structure_id: "g".into(),
        target_opening_m,
        issued_at_s: 1.0,
        available_at_s: 1.0,
    }
}

#[test]
fn actuator_targets_above_the_blocked_limit_reject() -> Result {
    let structures = batch(blocked_gates()?)?;
    let message = rejected(actuators(&structures, LIMIT.next_up()))?;
    assert!(message.contains("inside actual gate range"), "{message}");
    actuators(&structures, LIMIT)?;
    let origin = actuators(&structures, 0.0)?;
    let next = origin.trial_to(1.0, &[update(LIMIT)])?.into_next();
    assert_eq!(next.time(), 1.0);
    let message = rejected(origin.trial_to(1.0, &[update(LIMIT.next_up())]))?;
    assert!(message.contains("outside actual gate range"), "{message}");
    Ok(())
}

#[test]
fn actuator_packages_restore_a_blocked_gate_and_reject_a_forged_condition() -> Result {
    let structures = batch(blocked_gates()?)?;
    let source = actuators(&structures, 0.0)?
        .trial_to(1.0, &[update(LIMIT)])?
        .into_next();
    let data = source.snapshot_data()?;
    let mut destination = actuators(&structures, 0.0)?;
    destination.restore_data(&data)?;
    assert_eq!(destination.snapshot_data()?, data);
    let mut forged = data.clone();
    forged.elements[0].condition = StructureCondition::AsBuilt;
    let before = destination.snapshot_data()?;
    rejected(destination.restore_data(&forged))?;
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

fn controls(
    structures: &StructureBatch<Runtime>,
    target: f64,
) -> std::result::Result<OperatingControlBatch<Runtime>, StructureError> {
    structures.with_operating_controls(
        vec![OperatingControlDefinition {
            controller_id: "relay".into(),
            structure_id: "g".into(),
            signal: ControlStageSignal {
                signal_id: "level".into(),
                support_id: "level-support".into(),
                datum: "Analytical blockage datum".into(),
            },
            lower_stage_m: 1.0,
            upper_stage_m: 2.0,
            high_qualification_delay_s: 0.0,
            low_qualification_delay_s: 0.0,
            low_command: ControlCommand::GateTarget { opening_m: 0.0 },
            high_command: ControlCommand::GateTarget { opening_m: target },
            provenance: EVIDENCE.into(),
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
            decision_times_s: vec![1.0],
            end_s: 2.0,
        },
        SampledControlTiming {
            maximum_decision_gap_s: 2.0,
            maximum_timer_lateness_s: 0.0,
            sampled_applicability: EVIDENCE.into(),
        },
    )
}

#[test]
fn control_targets_above_the_blocked_limit_reject() -> Result {
    let structures = batch(blocked_gates()?)?;
    let message = rejected(controls(&structures, LIMIT.next_up()))?;
    assert!(message.contains("control command="), "{message}");
    let admitted = controls(&structures, LIMIT)?;
    let data = admitted.snapshot_data()?;
    let mut destination = controls(&structures, LIMIT)?;
    destination.restore_data(&data)?;
    let mut forged = data.clone();
    forged.elements[0].condition = StructureCondition::AsBuilt;
    rejected(destination.restore_data(&forged))?;
    Ok(())
}
