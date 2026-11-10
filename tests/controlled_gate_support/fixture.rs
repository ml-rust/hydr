use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const DATUM: &str = "Analytical controlled gate datum";

#[derive(Clone)]
pub struct Config {
    pub width: usize,
    pub start: f64,
    pub end: f64,
    pub times: Vec<f64>,
    pub opening: f64,
    pub target: f64,
    pub low: f64,
    pub high: f64,
    pub initial_latch: ControlLatch,
    pub rates: (f64, f64),
    pub delays: (f64, f64),
    pub maximum_gap: f64,
    pub lateness: f64,
    pub accuracy: GateActuatorAccuracy,
}

impl Config {
    pub fn switching() -> Self {
        Self {
            width: 1,
            start: 0.0,
            end: 4.0,
            times: vec![1.0, 2.0, 3.0],
            opening: 0.0,
            target: 0.0,
            low: 0.0,
            high: 1.0,
            initial_latch: ControlLatch::Low,
            rates: (0.25, 0.5),
            delays: (0.0, 0.0),
            maximum_gap: 4.0,
            lateness: 4.0,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
        }
    }

    pub fn context(&self) -> Result<StructureBatch<crate::runtime::Runtime>> {
        Ok(StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            (0..self.width).map(gate).collect::<Result<Vec<_>>>()?,
        )?)
    }

    pub fn controls(
        &self,
        context: &StructureBatch<crate::runtime::Runtime>,
    ) -> Result<OperatingControlBatch<crate::runtime::Runtime>> {
        Ok(context.with_operating_controls(
            (0..self.width)
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
                    high_qualification_delay_s: self.delays.0,
                    low_qualification_delay_s: self.delays.1,
                    low_command: ControlCommand::GateTarget {
                        opening_m: self.low,
                    },
                    high_command: ControlCommand::GateTarget {
                        opening_m: self.high,
                    },
                    provenance: "Explicit analytical sampled relay".into(),
                })
                .collect(),
            (0..self.width)
                .map(|lane| ControlInitialState {
                    controller_id: controller(lane),
                    latch: self.initial_latch,
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
                maximum_decision_gap_s: self.maximum_gap,
                maximum_timer_lateness_s: self.lateness,
                sampled_applicability: "Explicit sampled-only analytical policy".into(),
            },
        )?)
    }

    pub fn actuators(
        &self,
        context: &StructureBatch<crate::runtime::Runtime>,
    ) -> Result<GateActuatorBatch<crate::runtime::Runtime>> {
        Ok(context.with_gate_actuators(
            (0..self.width)
                .map(|lane| GateActuatorDefinition {
                    actuator_id: actuator(lane),
                    structure_id: device(lane),
                    opening_rate_m_per_s: self.rates.0,
                    closing_rate_m_per_s: self.rates.1,
                    accuracy: self.accuracy,
                    rate_provenance: "Explicit analytical directional rates".into(),
                    ideal_response_applicability: "Ideal rate response without inertia or backlash"
                        .into(),
                })
                .collect(),
            (0..self.width)
                .map(|lane| GateActuatorInitialState {
                    actuator_id: actuator(lane),
                    opening_m: self.opening,
                    target_opening_m: self.target,
                })
                .collect(),
            GateActuatorSchedule {
                initial_time_s: self.start,
                target_times_s: self.times.clone(),
                end_s: self.end,
            },
        )?)
    }

    pub fn build(&self) -> Result<ControlledGateBatch<crate::runtime::Runtime>> {
        let context = self.context()?;
        Ok(ControlledGateBatch::<crate::runtime::Runtime>::new(
            self.controls(&context)?,
            self.actuators(&context)?,
        )?)
    }
}

pub fn device(lane: usize) -> String {
    format!("gate-{lane}")
}
pub fn controller(lane: usize) -> String {
    format!("controller-{lane}")
}
pub fn actuator(lane: usize) -> String {
    format!("actuator-{lane}")
}

pub fn gate(lane: usize) -> Result<StructureElement> {
    Ok(StructureElement {
        id: device(lane),
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
                coefficient_provenance: "Analytical controlled aperture".into(),
                hydrostatic_quiescent_pool_assertion: "Quiescent analytical endpoints".into(),
                free_submerged_aperture_applicability: "Declared complete aperture envelope".into(),
                unobstructed_aperture_assertion: "Both connections expose the aperture".into(),
                symmetric_reversal: true,
            },
        )?),
    })
}

pub fn samples(width: usize, time: f64, stage: f64) -> Vec<ControlSample> {
    (0..width)
        .map(|lane| ControlSample {
            controller_id: controller(lane),
            signal_id: format!("signal-{lane}"),
            support_id: format!("support-{lane}"),
            datum: DATUM.into(),
            observed_at_s: time,
            available_at_s: time,
            stage_m: stage,
        })
        .collect()
}

pub fn updates(width: usize, time: f64, target: f64) -> Vec<GateTargetUpdate> {
    (0..width)
        .map(|lane| GateTargetUpdate {
            actuator_id: actuator(lane),
            structure_id: device(lane),
            target_opening_m: target,
            issued_at_s: time,
            available_at_s: time,
        })
        .collect()
}

pub fn accept(
    mut batch: ControlledGateBatch<crate::runtime::Runtime>,
    end: f64,
    stage: Option<f64>,
) -> Result<ControlledGateBatch<crate::runtime::Runtime>> {
    while let Some(event) = batch.next_event()? {
        if event.time_s >= end {
            break;
        }
        assert!(event.time_s > batch.time());
        assert!(
            event.decision_index.is_none(),
            "Unsupplied decision: {event:?}"
        );
        batch = batch.trial_to(event.time_s, &[])?.into_next();
    }
    let packets = stage.map_or_else(Vec::new, |value| {
        samples(batch.controls().definitions().len(), end, value)
    });
    Ok(batch.trial_to(end, &packets)?.into_next())
}

pub fn position(batch: &ControlledGateBatch<crate::runtime::Runtime>, exact: f64) -> Result {
    for output in batch.actuators().outputs()? {
        assert!(output.position.lower_opening_m <= exact, "{output:?}");
        assert!(exact <= output.position.upper_opening_m, "{output:?}");
        assert!(output.position.lower_opening_m <= output.position.opening_m);
        assert!(output.position.opening_m <= output.position.upper_opening_m);
    }
    Ok(())
}

pub fn rejected<T>(result: std::result::Result<T, StructureError>) -> Result<StructureError> {
    result
        .err()
        .ok_or_else(|| "Expected controlled gate rejection".into())
}

pub fn malformed_restore(
    destination: &mut ControlledGateBatch<crate::runtime::Runtime>,
    data: &ControlledGateSnapshotData,
) -> Result {
    let before = destination.snapshot_data()?;
    let error = rejected(destination.restore_data(data))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition),
        "{error}"
    );
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}
