//! Exclusive manual, prescribed, and accepted actuator inputs for one hydraulic point.

use hydr::structures::{
    ActuatorSourceFrame, DirectionPolicy, EndpointHead, GateActuatorAccuracy,
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule, GateApplicability,
    GateMotionLimits, GateOpeningKnot, GateOpeningTrajectory, GateParameters, GateTargetUpdate,
    GateTrajectoryAccuracy, GateTrajectoryAssignment, HeadConvention, HeadSupport,
    OperatingSourceBinding, OrificeApplicability, PumpApplicability, PumpCurve, PumpCurveKnot,
    PumpParameters, PumpSpeedPolicy, StructureBatch, StructureElement, StructureError,
    StructureHeads, StructureLaw, StructureOperatingPoint, StructureOperation,
    SubmergedOrificeParameters,
};

const DATUM: &str = "synthetic common datum";

fn element(id: &str, direction: DirectionPolicy, law: StructureLaw) -> StructureElement {
    StructureElement {
        id: id.into(),
        endpoint_a: "upstream".into(),
        endpoint_b: "downstream".into(),
        datum: DATUM.into(),
        direction,
        law,
    }
}

fn gate(id: &str) -> Result<StructureElement, StructureError> {
    Ok(element(
        id,
        DirectionPolicy::Bidirectional,
        StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            1.0,
            0.0,
            1.0,
            0.6,
            9.81,
            GateApplicability {
                coefficient_provenance: "explicit synthetic Cd=0.6, without field calibration"
                    .into(),
                hydrostatic_quiescent_pool_assertion:
                    "synthetic hydrostatic pools with negligible approach velocity".into(),
                free_submerged_aperture_applicability:
                    "one synthetic coefficient covers the admitted aperture envelope".into(),
                unobstructed_aperture_assertion:
                    "both synthetic connections expose the complete aperture".into(),
                symmetric_reversal: true,
            },
        )?),
    ))
}

fn heads(id: &str, a: f64, b: f64) -> StructureHeads {
    StructureHeads {
        structure_id: id.into(),
        endpoint_a: EndpointHead {
            id: "upstream".into(),
            support: HeadSupport::Wet { stage_m: a },
        },
        endpoint_b: EndpointHead {
            id: "downstream".into(),
            support: HeadSupport::Wet { stage_m: b },
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let passive = element(
        "passive-orifice",
        DirectionPolicy::Bidirectional,
        StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
            0.5,
            0.0,
            0.5,
            0.6,
            9.81,
            OrificeApplicability {
                coefficient_provenance: "explicit synthetic Cd=0.6".into(),
                fully_submerged_applicability:
                    "known wet synthetic pools strictly cover the opening crown".into(),
                symmetric_reversal: true,
            },
        )?),
    );
    let pump = element(
        "manual-pump",
        DirectionPolicy::AtoBOnly,
        StructureLaw::Pump(PumpParameters::new(
            PumpCurve::new(
                vec![
                    PumpCurveKnot { head_m: 0.0, discharge_m3_per_s: 1.0 },
                    PumpCurveKnot { head_m: 4.0, discharge_m3_per_s: 0.0 },
                ],
                "synthetic SI water curve at explicit reference speed 1000 rpm".into(),
            )?,
            0.0,
            PumpSpeedPolicy::ReferenceOnly,
            PumpApplicability {
                endpoint_stage_head_assertion: "synthetic quiescent common-datum endpoints".into(),
                connection_losses_and_velocity_head_account: "effective fixture curve includes zero connection losses and velocity-head differences".into(),
                stopped_isolation_assertion: "explicit isolator prevents forward siphoning and reverse flow while stopped".into(),
                operating_envelope: "known wet donor above stage zero and known wet receiving pool".into(),
            },
        )?),
    );
    let structures = StructureBatch::new(vec![
        gate("manual-gate")?,
        passive,
        gate("prescribed-gate")?,
        pump,
        gate("actuated-gate")?,
    ])?;
    let prescribed = structures.with_gate_trajectory_subset(vec![GateTrajectoryAssignment {
        structure_id: "prescribed-gate".into(),
        trajectory: GateOpeningTrajectory::new(
            vec![
                GateOpeningKnot {
                    time_s: 0.0,
                    opening_m: 0.25,
                },
                GateOpeningKnot {
                    time_s: 2.0,
                    opening_m: 0.75,
                },
            ],
            GateMotionLimits {
                maximum_opening_speed_m_per_s: 0.5,
                maximum_closing_speed_m_per_s: 0.5,
            },
            GateTrajectoryAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
            },
            "explicit synthetic opening schedule and declared motion envelope".into(),
        )?,
    }])?;
    let actuator = structures.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: "gate-drive".into(),
            structure_id: "actuated-gate".into(),
            opening_rate_m_per_s: 0.25,
            closing_rate_m_per_s: 0.5,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "explicit synthetic asymmetric directional rates".into(),
            ideal_response_applicability:
                "ideal constant rates without inertia, backlash, or load dependence".into(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: "gate-drive".into(),
            opening_m: 0.0,
            target_opening_m: 0.0,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: vec![0.5],
            end_s: 2.0,
        },
    )?;
    let registry = structures.with_operating_sources(vec![
        OperatingSourceBinding::Manual {
            source_id: "manual-position".into(),
            structure_ids: vec!["manual-gate".into()],
        },
        OperatingSourceBinding::Prescribed {
            source_id: "opening-schedule".into(),
            trajectory: &prescribed,
        },
        OperatingSourceBinding::Manual {
            source_id: "manual-equipment".into(),
            structure_ids: vec!["passive-orifice".into(), "manual-pump".into()],
        },
        OperatingSourceBinding::Actuated {
            source_id: "actual-drive".into(),
            actuator: &actuator,
        },
    ])?;
    let actuator = actuator
        .trial_to(
            0.5,
            &[GateTargetUpdate {
                actuator_id: "gate-drive".into(),
                structure_id: "actuated-gate".into(),
                target_opening_m: 1.0,
                issued_at_s: 0.5,
                available_at_s: 0.5,
            }],
        )?
        .into_next();
    let actuator = actuator.trial_to(1.0, &[])?.into_next();
    let proposal = registry.propose_at(
        1.0,
        &[
            heads("manual-gate", 3.0, 1.0),
            heads("passive-orifice", 3.0, 1.0),
            heads("prescribed-gate", 3.0, 1.0),
            heads("manual-pump", 3.0, 4.0),
            heads("actuated-gate", 3.0, 1.0),
        ],
        &[
            StructureOperation {
                structure_id: "manual-gate".into(),
                operating_point: StructureOperatingPoint::GateOpening { opening_m: 0.5 },
            },
            StructureOperation {
                structure_id: "passive-orifice".into(),
                operating_point: StructureOperatingPoint::Passive,
            },
            StructureOperation {
                structure_id: "manual-pump".into(),
                operating_point: StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            },
        ],
        &[ActuatorSourceFrame {
            source_id: "actual-drive",
            actuator: &actuator,
        }],
    )?;
    println!("query_time_s={}", proposal.time_s());
    for output in proposal.outputs()? {
        println!("point_output={output:?}");
    }
    Ok(())
}
