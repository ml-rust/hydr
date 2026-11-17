//! Minimal control, actuator, controlled-gate and controlled-pump cases for capability checks.

use crate::gate_fixture::Config;
use hydr::capability::ComponentCapabilities;
use hydr::structures::{
    ControlSample, ControlledGateBatch, ControlledPumpBatch, DirectionPolicy, GateActuatorBatch,
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule, GateTargetUpdate,
    OperatingControlBatch, OrificeApplicability, StructureBatch, StructureElement, StructureError,
    StructureInputErrorKind, StructureLaw, SubmergedOrificeParameters,
    SuppressedRectangularWeirParameters, WeirApplicability, controlled_gate_capabilities,
    controlled_pump_capabilities, gate_actuator_capabilities, operating_control_capabilities,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;
pub type Runtime = crate::runtime::Runtime;

/// Structure component, one per declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    OperatingControl,
    GateActuator,
    ControlledGate,
    ControlledPump,
}

pub const KINDS: [Kind; 4] = [
    Kind::OperatingControl,
    Kind::GateActuator,
    Kind::ControlledGate,
    Kind::ControlledPump,
];

impl Kind {
    pub fn capabilities(self) -> &'static ComponentCapabilities {
        match self {
            Self::OperatingControl => operating_control_capabilities(),
            Self::GateActuator => gate_actuator_capabilities(),
            Self::ControlledGate => controlled_gate_capabilities(),
            Self::ControlledPump => controlled_pump_capabilities(),
        }
    }
}

/// Error class a rejection reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Input(StructureInputErrorKind),
    Numerical,
    Runtime,
}

/// Rejection variant and message, or `None` when the case is accepted.
pub type Outcome = Result<Option<(Variant, String)>, Box<dyn std::error::Error>>;

pub fn variant(error: &StructureError) -> Variant {
    match error {
        StructureError::InvalidInput { kind, .. } => Variant::Input(*kind),
        StructureError::Numerical { .. } => Variant::Numerical,
        StructureError::Runtime { .. } => Variant::Runtime,
    }
}

/// Classifies a constructor result.
pub fn built<T>(result: Result<T, StructureError>) -> Outcome {
    Ok(result
        .err()
        .map(|error| (variant(&error), error.to_string())))
}

/// Asserts that `outcome` admits its case.
pub fn admitted(outcome: Outcome, case: &str) -> TestResult {
    match outcome? {
        None => Ok(()),
        Some((variant, reason)) => Err(format!("{case}: {variant:?} {reason}").into()),
    }
}

/// Runs one control trial to `end`; a rejection must leave accepted state unchanged.
pub fn control_outcome(
    batch: &OperatingControlBatch<Runtime>,
    end: f64,
    samples: &[ControlSample],
) -> Outcome {
    let before = batch.snapshot_data()?;
    let outcome = match batch.trial_to(end, samples) {
        Ok(trial) => {
            trial.into_next().commands()?;
            None
        }
        Err(error) => Some((variant(&error), error.to_string())),
    };
    assert_eq!(batch.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Runs one actuator trial to `end`; a rejection must leave accepted state unchanged.
pub fn actuator_outcome(
    batch: &GateActuatorBatch<Runtime>,
    end: f64,
    updates: &[GateTargetUpdate],
) -> Outcome {
    let before = batch.snapshot_data()?;
    let outcome = match batch.trial_to(end, updates) {
        Ok(trial) => {
            trial.into_next().outputs()?;
            None
        }
        Err(error) => Some((variant(&error), error.to_string())),
    };
    assert_eq!(batch.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Runs one controlled-gate trial to `end`; a rejection must leave both children unchanged.
pub fn controlled_gate_outcome(
    batch: &ControlledGateBatch<Runtime>,
    end: f64,
    samples: &[ControlSample],
) -> Outcome {
    let before = batch.snapshot_data()?;
    let outcome = match batch.trial_to(end, samples) {
        Ok(trial) => {
            trial.into_next().actuators().outputs()?;
            None
        }
        Err(error) => Some((variant(&error), error.to_string())),
    };
    assert_eq!(batch.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Runs one controlled-pump trial to `end`; a rejection must leave the controller unchanged.
pub fn controlled_pump_outcome(
    batch: &ControlledPumpBatch<Runtime>,
    end: f64,
    samples: &[ControlSample],
) -> Outcome {
    let before = batch.snapshot_data()?;
    let outcome = match batch.trial_to(end, samples) {
        Ok(trial) => {
            trial.into_next().outputs()?;
            None
        }
        Err(error) => Some((variant(&error), error.to_string())),
    };
    assert_eq!(batch.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Actuator `id` on `structure` with the switching fixture's rates and accuracy.
pub fn actuator(id: &str, structure: &str) -> GateActuatorDefinition {
    let config = Config::switching();
    GateActuatorDefinition {
        actuator_id: id.into(),
        structure_id: structure.into(),
        opening_rate_m_per_s: config.rates.0,
        closing_rate_m_per_s: config.rates.1,
        accuracy: config.accuracy,
        rate_provenance: "Explicit analytical directional rates".into(),
        ideal_response_applicability: "Ideal rate response without inertia or backlash".into(),
    }
}

/// Actuators closed at zero with one target event at `1 s` and terminal support at `2 s`.
pub fn actuators(
    context: &StructureBatch<Runtime>,
    definitions: Vec<GateActuatorDefinition>,
) -> Result<GateActuatorBatch<Runtime>, StructureError> {
    let initial = definitions
        .iter()
        .map(|definition| GateActuatorInitialState {
            actuator_id: definition.actuator_id.clone(),
            opening_m: 0.0,
            target_opening_m: 0.0,
        })
        .collect();
    context.with_gate_actuators(
        definitions,
        initial,
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: vec![1.0],
            end_s: 2.0,
        },
    )
}

/// Suppressed weir `id` on the controlled-gate fixture datum.
pub fn weir(id: &str) -> Result<StructureElement, StructureError> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "pool".into(),
        endpoint_b: "river".into(),
        datum: crate::gate_fixture::DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            0.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Analytical weir fixture".into(),
                free_nappe_and_approach_assertion: "Suppressed analytical crest".into(),
                submergence_correction_assertion: "Declared analytical correction".into(),
                symmetric_reversal: true,
            },
        )?),
    })
}

/// Submerged orifice `id` on the controlled-gate fixture datum.
pub fn orifice(id: &str) -> Result<StructureElement, StructureError> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "pool".into(),
        endpoint_b: "river".into(),
        datum: crate::gate_fixture::DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
            1.0,
            0.0,
            1.0,
            0.6,
            9.81,
            OrificeApplicability {
                coefficient_provenance: "Analytical orifice fixture".into(),
                fully_submerged_applicability: "Declared submerged analytical envelope".into(),
                symmetric_reversal: true,
            },
        )?),
    })
}
