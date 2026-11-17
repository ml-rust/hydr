use super::Result;
use hydr::structures::*;

pub const DATUM: &str = "Analytical runtime control datum";

/// Lanes cover one controlled device and a partial tail beyond 32 lanes.
pub const WIDTHS: [usize; 2] = [1, 33];

/// Stage relay with thresholds `1 m` and `2 m`, without qualification delays.
pub struct Relay {
    pub low: ControlCommand,
    pub high: ControlCommand,
    pub times: Vec<f64>,
    pub end: f64,
    pub maximum_gap: f64,
    pub lateness: f64,
}

/// Arguments of `StructureBatch::with_operating_controls`, in order.
pub type RelayParts = (
    Vec<OperatingControlDefinition>,
    Vec<ControlInitialState>,
    ControlDecisionSchedule,
    SampledControlTiming,
);

pub fn device(lane: usize) -> String {
    format!("device-{lane}")
}

pub fn controller(lane: usize) -> String {
    format!("controller-{lane}")
}

impl Relay {
    pub fn parts(&self, width: usize) -> RelayParts {
        (
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
                    high_qualification_delay_s: 0.0,
                    low_qualification_delay_s: 0.0,
                    low_command: self.low,
                    high_command: self.high,
                    provenance: "Explicit analytical runtime relay".into(),
                })
                .collect(),
            (0..width)
                .map(|lane| ControlInitialState {
                    controller_id: controller(lane),
                    latch: ControlLatch::Low,
                    pending: None,
                    antecedent_sample: None,
                    last_transition_at_s: None,
                })
                .collect(),
            ControlDecisionSchedule {
                initial_time_s: 0.0,
                decision_times_s: self.times.clone(),
                end_s: self.end,
            },
            SampledControlTiming {
                maximum_decision_gap_s: self.maximum_gap,
                maximum_timer_lateness_s: self.lateness,
                sampled_applicability: "Explicit sampled-only analytical policy".into(),
            },
        )
    }
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

pub fn pumps(width: usize) -> Result<Vec<StructureElement>> {
    (0..width)
        .map(|lane| {
            Ok(StructureElement {
                id: device(lane),
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
                                head_m: 8.0,
                                discharge_m3_per_s: 0.0,
                            },
                        ],
                        "Exact analytical reference-speed curve".into(),
                    )?,
                    -2.0,
                    PumpSpeedPolicy::DeclaredAffinity {
                        minimum_ratio: 0.0625,
                        maximum_ratio: 2.0,
                        applicability: "Declared unchanged-geometry analytical speed envelope"
                            .into(),
                    },
                    PumpApplicability {
                        endpoint_stage_head_assertion: "Quiescent analytical endpoint stages"
                            .into(),
                        connection_losses_and_velocity_head_account:
                            "Declared curve includes connection losses".into(),
                        stopped_isolation_assertion:
                            "Stopped device blocks siphoning and reverse flow".into(),
                        operating_envelope: "Analytical numerical fixture without calibration"
                            .into(),
                    },
                )?),
            })
        })
        .collect()
}

pub fn gates(width: usize) -> Result<Vec<StructureElement>> {
    (0..width)
        .map(|lane| {
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
                        hydrostatic_quiescent_pool_assertion: "Quiescent analytical endpoints"
                            .into(),
                        free_submerged_aperture_applicability:
                            "Declared complete aperture envelope".into(),
                        unobstructed_aperture_assertion: "Both connections expose the aperture"
                            .into(),
                        symmetric_reversal: true,
                    },
                )?),
            })
        })
        .collect()
}
