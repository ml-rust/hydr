use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn gate(id: &str) -> Result<StructureElement> {
    gate_max(id, 1.0)
}

pub fn gate_max(id: &str, maximum: f64) -> Result<StructureElement> {
    Ok(element(
        id,
        StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            1.0,
            0.0,
            maximum,
            0.6,
            9.81,
            GateApplicability {
                coefficient_provenance: "Analytical point-source fixture".into(),
                hydrostatic_quiescent_pool_assertion: "Quiescent analytical endpoints".into(),
                free_submerged_aperture_applicability: "Declared analytical aperture envelope"
                    .into(),
                unobstructed_aperture_assertion: "Both endpoints expose the aperture".into(),
                symmetric_reversal: true,
            },
        )?),
    ))
}

fn element(id: &str, law: StructureLaw) -> StructureElement {
    StructureElement {
        id: id.into(),
        endpoint_a: "pool".into(),
        endpoint_b: "river".into(),
        datum: "Analytical point datum".into(),
        law,
        direction: DirectionPolicy::Bidirectional,
    }
}

pub fn passive(id: &str) -> Result<StructureElement> {
    Ok(element(
        id,
        StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            0.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Analytical crest".into(),
                free_nappe_and_approach_assertion: "Suppressed crest".into(),
                submergence_correction_assertion: "Declared correction".into(),
                symmetric_reversal: true,
            },
        )?),
    ))
}

pub fn pump(id: &str) -> Result<StructureElement> {
    let mut element = element(
        id,
        StructureLaw::Pump(PumpParameters::new(
            PumpCurve::new(
                vec![
                    PumpCurveKnot {
                        head_m: -2.0,
                        discharge_m3_per_s: 3.0,
                    },
                    PumpCurveKnot {
                        head_m: 2.0,
                        discharge_m3_per_s: 0.0,
                    },
                ],
                "Analytical bounded curve".into(),
            )?,
            -1.0,
            PumpSpeedPolicy::ReferenceOnly,
            PumpApplicability {
                endpoint_stage_head_assertion: "Quiescent analytical endpoints".into(),
                connection_losses_and_velocity_head_account: "Curve includes connection losses"
                    .into(),
                stopped_isolation_assertion: "Explicit stopped isolation".into(),
                operating_envelope: "Analytical numerical fixture".into(),
            },
        )?),
    );
    element.direction = DirectionPolicy::AtoBOnly;
    Ok(element)
}

pub fn heads(context: &StructureBatch) -> Vec<StructureHeads> {
    context
        .elements()
        .iter()
        .map(|element| StructureHeads {
            structure_id: element.id.clone(),
            endpoint_a: EndpointHead {
                id: element.endpoint_a.clone(),
                support: HeadSupport::Wet { stage_m: 2.0 },
            },
            endpoint_b: EndpointHead {
                id: element.endpoint_b.clone(),
                support: HeadSupport::Wet { stage_m: 1.0 },
            },
            datum: element.datum.clone(),
            convention: HeadConvention::QuiescentStage,
        })
        .collect()
}

pub fn operation(id: &str, point: StructureOperatingPoint) -> StructureOperation {
    StructureOperation {
        structure_id: id.into(),
        operating_point: point,
    }
}

pub fn opening(id: &str, value: f64) -> StructureOperation {
    operation(
        id,
        StructureOperatingPoint::GateOpening { opening_m: value },
    )
}

pub fn trajectory(
    context: &StructureBatch,
    id: &str,
    knots: &[(f64, f64)],
) -> Result<GateTrajectoryBatch> {
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
        "Original analytical point trajectory".into(),
    )?;
    Ok(
        context.with_gate_trajectory_subset(vec![GateTrajectoryAssignment {
            structure_id: id.into(),
            trajectory,
        }])?,
    )
}

pub fn actuator(
    context: &StructureBatch,
    id: &str,
    opening: f64,
    target: f64,
    rates: (f64, f64),
    start: f64,
    end: f64,
) -> Result<GateActuatorBatch> {
    Ok(context.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: format!("actuator-{id}"),
            structure_id: id.into(),
            opening_rate_m_per_s: rates.0,
            closing_rate_m_per_s: rates.1,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "Explicit analytical rates".into(),
            ideal_response_applicability: "Ideal rate-limited response without inertia or backlash"
                .into(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: format!("actuator-{id}"),
            opening_m: opening,
            target_opening_m: target,
        }],
        GateActuatorSchedule {
            initial_time_s: start,
            target_times_s: vec![],
            end_s: end,
        },
    )?)
}

pub fn advance(mut batch: GateActuatorBatch, end: f64) -> Result<GateActuatorBatch> {
    while let Some(event) = batch.next_event()? {
        assert!(event.time_s > batch.time());
        if event.time_s >= end {
            break;
        }
        assert!(event.target_index.is_none());
        batch = batch.trial_to(event.time_s, &[])?.into_next();
    }
    Ok(batch.trial_to(end, &[])?.into_next())
}

pub fn rejected<T>(result: std::result::Result<T, StructureError>) -> Result<StructureError> {
    result
        .err()
        .ok_or_else(|| "Expected operating-source rejection".into())
}

pub fn kind<T>(
    result: std::result::Result<T, StructureError>,
    expected: StructureInputErrorKind,
) -> Result {
    let error = rejected(result)?;
    assert_eq!(error.input_kind(), Some(expected), "{error}");
    Ok(())
}

pub struct Mixed {
    pub context: StructureBatch,
    pub prescribed: [GateTrajectoryBatch; 2],
    pub actuators: [GateActuatorBatch; 2],
    pub registry: OperatingSourceBatch,
}

impl Mixed {
    pub fn new() -> Result<Self> {
        let context = StructureBatch::new(vec![
            gate("manual-a")?,
            gate("prescribed-a")?,
            passive("weir")?,
            gate("actuated-a")?,
            pump("pump")?,
            gate("prescribed-b")?,
            gate("actuated-b")?,
            gate("manual-b")?,
        ])?;
        let prescribed = [
            trajectory(&context, "prescribed-a", &[(0.0, 0.125), (2.0, 0.375)])?,
            trajectory(&context, "prescribed-b", &[(0.0, 0.75), (2.0, 0.5)])?,
        ];
        let actuators = [
            actuator(&context, "actuated-a", 0.25, 1.0, (0.25, 0.5), 0.0, 4.0)?,
            actuator(&context, "actuated-b", 0.75, 0.0, (0.25, 0.125), 0.0, 4.0)?,
        ];
        let registry = context.with_operating_sources(vec![
            OperatingSourceBinding::Actuated {
                source_id: "actual-b".into(),
                actuator: &actuators[1],
            },
            OperatingSourceBinding::Prescribed {
                source_id: "path-a".into(),
                trajectory: &prescribed[0],
            },
            OperatingSourceBinding::Manual {
                source_id: "manual".into(),
                structure_ids: ["manual-a", "weir", "pump", "manual-b"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
            OperatingSourceBinding::Actuated {
                source_id: "actual-a".into(),
                actuator: &actuators[0],
            },
            OperatingSourceBinding::Prescribed {
                source_id: "path-b".into(),
                trajectory: &prescribed[1],
            },
        ])?;
        Ok(Self {
            context,
            prescribed,
            actuators,
            registry,
        })
    }
    pub fn frames(&self) -> Vec<OperatingSourceFrame<'_>> {
        vec![
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-b",
                actuator: &self.actuators[1],
            },
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-a",
                actuator: &self.actuators[0],
            },
        ]
    }
    pub fn manual(&self) -> Vec<StructureOperation> {
        vec![
            opening("manual-a", 0.125),
            operation("weir", StructureOperatingPoint::Passive),
            operation(
                "pump",
                StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            ),
            opening("manual-b", 0.875),
        ]
    }
}

pub fn explicit_match(
    context: &StructureBatch,
    heads: &[StructureHeads],
    proposal: &OperatingSourceProposal,
) -> Result {
    let operations = proposal.operations()?;
    let expected = context
        .propose_with_operations(heads, &operations)?
        .outputs()?;
    let actual = proposal.outputs()?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.hydraulic, expected);
    }
    Ok(())
}
