use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const DATUM: &str = "Analytical controlled pump datum";

#[derive(Clone)]
pub struct Config {
    pub start: f64,
    pub end: f64,
    pub times: Vec<f64>,
    pub commands: [ControlCommand; 2],
    pub delays: (f64, f64),
}
impl Config {
    pub fn switching() -> Self {
        Self {
            start: 0.0,
            end: 4.0,
            times: vec![1.0, 2.0, 3.0],
            commands: [
                ControlCommand::PumpStoppedIsolated,
                ControlCommand::PumpRunning { speed_ratio: 1.0 },
            ],
            delays: (0.0, 0.0),
        }
    }
    pub fn controls(
        &self,
        context: &StructureBatch<crate::runtime::Runtime>,
        ids: &[&str],
    ) -> Result<OperatingControlBatch<crate::runtime::Runtime>> {
        Ok(context.with_operating_controls(
            ids.iter()
                .map(|id| OperatingControlDefinition {
                    controller_id: controller(id),
                    structure_id: (*id).into(),
                    signal: ControlStageSignal {
                        signal_id: format!("signal-{id}"),
                        support_id: format!("support-{id}"),
                        datum: DATUM.into(),
                    },
                    lower_stage_m: 1.0,
                    upper_stage_m: 2.0,
                    high_qualification_delay_s: self.delays.0,
                    low_qualification_delay_s: self.delays.1,
                    low_command: self.commands[0],
                    high_command: self.commands[1],
                    provenance: "Explicit analytical sampled pump relay".into(),
                })
                .collect(),
            ids.iter()
                .map(|id| ControlInitialState {
                    controller_id: controller(id),
                    latch: ControlLatch::Low,
                    pending: None,
                    antecedent_sample: None,
                    last_transition_at_s: None,
                })
                .collect(),
            ControlDecisionSchedule {
                initial_time_s: self.start,
                decision_times_s: self.times.clone(),
                end_s: self.end,
            },
            SampledControlTiming {
                maximum_decision_gap_s: self.end - self.start,
                maximum_timer_lateness_s: self.end - self.start,
                sampled_applicability:
                    "Explicit sampled analytical policy without crossing completeness".into(),
            },
        )?)
    }
    pub fn build(
        &self,
        context: &StructureBatch<crate::runtime::Runtime>,
        ids: &[&str],
    ) -> Result<ControlledPumpBatch<crate::runtime::Runtime>> {
        Ok(ControlledPumpBatch::<crate::runtime::Runtime>::new(
            self.controls(context, ids)?,
            responses(ids),
        )?)
    }
}

pub fn controller(id: &str) -> String {
    format!("relay-{id}")
}

pub fn responses(ids: &[&str]) -> Vec<PumpResponseDefinition> {
    ids.iter()
        .map(|id| PumpResponseDefinition {
            structure_id: (*id).into(),
            model: PumpResponseModel::IdealInstantaneous,
            response_provenance: "Explicit analytical instantaneous operation".into(),
            ideal_response_applicability: "No startup delay, acceleration, or coast-down".into(),
        })
        .collect()
}

pub fn pump(id: &str) -> Result<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "pool".into(),
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
                        head_m: 4.0,
                        discharge_m3_per_s: 2.0,
                    },
                    PumpCurveKnot {
                        head_m: 8.0,
                        discharge_m3_per_s: 0.0,
                    },
                ],
                "Exact analytical bounded reference-speed curve".into(),
            )?,
            -2.0,
            PumpSpeedPolicy::DeclaredAffinity {
                minimum_ratio: 0.0625,
                maximum_ratio: 2.0,
                applicability: "Declared unchanged-geometry analytical speed envelope".into(),
            },
            PumpApplicability {
                endpoint_stage_head_assertion: "Quiescent analytical endpoint stages".into(),
                connection_losses_and_velocity_head_account:
                    "Declared curve includes connection losses".into(),
                stopped_isolation_assertion: "Stopped device blocks siphoning and reverse flow"
                    .into(),
                operating_envelope: "Analytical numerical fixture without measured calibration"
                    .into(),
            },
        )?),
        condition: StructureCondition::AsBuilt,
    })
}

pub fn gate(id: &str) -> Result<StructureElement> {
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
                coefficient_provenance: "Analytical mixed-source aperture".into(),
                hydrostatic_quiescent_pool_assertion: "Quiescent analytical endpoints".into(),
                free_submerged_aperture_applicability: "Declared complete aperture envelope".into(),
                unobstructed_aperture_assertion: "Both connections expose the aperture".into(),
                symmetric_reversal: true,
            },
        )?),
        condition: StructureCondition::AsBuilt,
    })
}

pub fn samples(ids: &[&str], time: f64, stage: f64) -> Vec<ControlSample> {
    ids.iter()
        .map(|id| ControlSample {
            controller_id: controller(id),
            signal_id: format!("signal-{id}"),
            support_id: format!("support-{id}"),
            datum: DATUM.into(),
            observed_at_s: time,
            available_at_s: time,
            stage_m: stage,
        })
        .collect()
}

pub fn accept(
    batch: ControlledPumpBatch<crate::runtime::Runtime>,
    end: f64,
    stage: Option<f64>,
) -> Result<ControlledPumpBatch<crate::runtime::Runtime>> {
    let ids = batch
        .controls()
        .definitions()
        .iter()
        .map(|value| value.structure_id.as_str())
        .collect::<Vec<_>>();
    let samples = stage.map_or_else(Vec::new, |value| samples(&ids, end, value));
    Ok(batch.trial_to(end, &samples)?.into_next())
}

pub fn heads(id: &str, a: HeadSupport, b: HeadSupport) -> StructureHeads {
    StructureHeads {
        structure_id: id.into(),
        endpoint_a: EndpointHead {
            id: "pool".into(),
            support: a,
        },
        endpoint_b: EndpointHead {
            id: "river".into(),
            support: b,
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }
}

pub fn wet(id: &str, receiver: f64) -> StructureHeads {
    heads(
        id,
        HeadSupport::Wet { stage_m: 0.0 },
        HeadSupport::Wet { stage_m: receiver },
    )
}

pub fn registry(
    context: &StructureBatch<crate::runtime::Runtime>,
    pump: &ControlledPumpBatch<crate::runtime::Runtime>,
) -> Result<OperatingSourceBatch<crate::runtime::Runtime>> {
    Ok(
        context.with_operating_sources(vec![OperatingSourceBinding::ControlledPump {
            source_id: "controlled".into(),
            pump,
        }])?,
    )
}

pub fn frame(
    pump: &ControlledPumpBatch<crate::runtime::Runtime>,
) -> OperatingSourceFrame<'_, crate::runtime::Runtime> {
    OperatingSourceFrame::ControlledPump {
        source_id: "controlled",
        pump,
    }
}

pub fn point(
    registry: &OperatingSourceBatch<crate::runtime::Runtime>,
    pump: &ControlledPumpBatch<crate::runtime::Runtime>,
    packets: &[StructureHeads],
) -> Result<Vec<OperatingSourceOutput>> {
    Ok(registry
        .propose_at(pump.time(), packets, &[], &[frame(pump)])?
        .outputs()?)
}

pub fn rejected<T>(result: std::result::Result<T, StructureError>) -> Result<StructureError> {
    result
        .err()
        .ok_or_else(|| "Expected controlled pump rejection".into())
}

pub fn kind<T>(
    result: std::result::Result<T, StructureError>,
    expected: StructureInputErrorKind,
) -> Result {
    let error = rejected(result)?;
    assert_eq!(error.input_kind(), Some(expected), "{error}");
    Ok(())
}

pub fn actual_gate(
    context: &StructureBatch<crate::runtime::Runtime>,
    id: &str,
    end: f64,
) -> Result<GateActuatorBatch<crate::runtime::Runtime>> {
    Ok(context.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: format!("actuator-{id}"),
            structure_id: id.into(),
            opening_rate_m_per_s: 0.0,
            closing_rate_m_per_s: 0.0,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "Explicit zero analytical rates".into(),
            ideal_response_applicability: "Ideal stationary aperture".into(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: format!("actuator-{id}"),
            opening_m: 0.25,
            target_opening_m: 0.25,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: Vec::new(),
            end_s: end,
        },
    )?)
}
