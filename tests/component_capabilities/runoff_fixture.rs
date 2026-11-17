//! Minimal soil, reservoir, subcatchment and runoff/pool cases for capability checks.

use hydr::capability::ComponentCapabilities;
use hydr::coupling::{
    RunoffPoolBatch, RunoffPoolError, RunoffPoolForcingInterval, RunoffPoolInitialState,
    RunoffPoolSetup, runoff_pool_capabilities,
};
use hydr::runoff::{
    ReservoirBatch, ReservoirError, ReservoirInput, ReservoirParameters, SubcatchmentBatch,
    SubcatchmentError, SubcatchmentInput, reservoir_capabilities, subcatchment_capabilities,
};
use hydr::soil::{
    ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, NativeSoilInput, SoilError,
    soil_capabilities,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;
pub type Soil = ModifiedGreenAmptBatch<crate::runtime::Runtime>;
pub type Reservoir = ReservoirBatch<crate::runtime::Runtime>;
pub type Subcatchment = SubcatchmentBatch<crate::runtime::Runtime>;
pub type RunoffPools = RunoffPoolBatch<crate::runtime::Runtime>;

/// Runoff component, one per declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Soil,
    Reservoir,
    Subcatchment,
    RunoffPools,
}

pub const KINDS: [Kind; 4] = [
    Kind::Soil,
    Kind::Reservoir,
    Kind::Subcatchment,
    Kind::RunoffPools,
];

impl Kind {
    pub fn capabilities(self) -> &'static ComponentCapabilities {
        match self {
            Self::Soil => soil_capabilities(),
            Self::Reservoir => reservoir_capabilities(),
            Self::Subcatchment => subcatchment_capabilities(),
            Self::RunoffPools => runoff_pool_capabilities(),
        }
    }
}

/// Error class a rejection reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Input,
    Numerical,
    Runtime,
    /// A nested subcatchment rejects inside the runoff/pool coordinator.
    Runoff,
    /// Nested coupled pools reject inside the runoff/pool coordinator.
    Pools,
    Unresolved,
}

/// Rejection variant and message, or `None` when the case is accepted.
pub type Outcome = Result<Option<(Variant, String)>, Box<dyn std::error::Error>>;

pub fn soil_variant(error: &SoilError) -> Variant {
    match error {
        SoilError::InvalidInput { .. } => Variant::Input,
        SoilError::Numerical { .. } => Variant::Numerical,
        SoilError::Runtime { .. } => Variant::Runtime,
    }
}

pub fn reservoir_variant(error: &ReservoirError) -> Variant {
    match error {
        ReservoirError::InvalidInput { .. } => Variant::Input,
        ReservoirError::Numerical { .. } => Variant::Numerical,
        ReservoirError::Runtime { .. } => Variant::Runtime,
    }
}

pub fn subcatchment_variant(error: &SubcatchmentError) -> Variant {
    match error {
        SubcatchmentError::InvalidInput { .. } => Variant::Input,
        SubcatchmentError::Numerical { .. } => Variant::Numerical,
        SubcatchmentError::Runtime { .. } => Variant::Runtime,
    }
}

pub fn runoff_pool_variant(error: &RunoffPoolError) -> Variant {
    match error {
        RunoffPoolError::Input(_) => Variant::Input,
        RunoffPoolError::Runoff(_) => Variant::Runoff,
        RunoffPoolError::Pools(_) => Variant::Pools,
        RunoffPoolError::Runtime(_) => Variant::Runtime,
        RunoffPoolError::Numerical(_) => Variant::Numerical,
        RunoffPoolError::Unresolved { .. } => Variant::Unresolved,
    }
}

/// Classifies a constructor result with `variant`.
pub fn built<T, E: std::fmt::Display>(result: Result<T, E>, variant: fn(&E) -> Variant) -> Outcome {
    Ok(result
        .err()
        .map(|error| (variant(&error), error.to_string())))
}

/// Pinned soil parameters shared with the soil component tests.
pub fn soil_parameters() -> Result<ModifiedGreenAmptParameters, SoilError> {
    ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)
}

/// Native soil input of `rate` ft/s without ponding.
pub fn soil_input(rate: f64) -> NativeSoilInput {
    NativeSoilInput {
        incoming_rate_ft_per_s: rate,
        ponded_depth_ft: 0.0,
    }
}

/// Runs one native soil trial; a rejection must leave the accepted memory unchanged.
pub fn soil_outcome(batch: &Soil, dt: f64, input: NativeSoilInput) -> Outcome {
    let before = batch.native_states()?;
    let outcome = match batch.trial_native(dt, &[input]) {
        Ok(trial) => {
            let output = trial.outputs()?;
            assert!(output[0].native_rate_ft_per_s > 0.0, "no infiltration");
            None
        }
        Err(error) => Some((soil_variant(&error), error.to_string())),
    };
    assert_eq!(batch.native_states()?, before, "rejection changed state");
    Ok(outcome)
}

/// Runs one reservoir trial to `1 s`; a rejection must leave accepted state unchanged.
pub fn reservoir_outcome(batch: &Reservoir, rate: f64) -> Outcome {
    let before = batch.snapshot_data()?;
    let input = [ReservoirInput {
        incoming_rate_m_per_s: rate,
    }];
    let outcome = match batch.trial_to(1.0, &input) {
        Ok(trial) => {
            trial.outputs()?;
            None
        }
        Err(error) => Some((reservoir_variant(&error), error.to_string())),
    };
    assert_eq!(batch.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Drainage branch that the reservoir parameters select, or the reason no branch applies.
pub fn drainage_branch(parameters: ReservoirParameters) -> Result<&'static str, String> {
    match (parameters.is_instantaneous(), parameters.is_closed()) {
        (false, false) => Ok("nonlinear"),
        (false, true) => Ok("closed"),
        (true, false) => Ok("instantaneous-overflow"),
        (true, true) => Err(format!("{parameters:?} selects two branches")),
    }
}

/// One parameter set per drainage branch, on the shared native reservoir geometry.
pub fn drainage_samples() -> Result<[ReservoirParameters; 3], ReservoirError> {
    let native =
        |width, n| ReservoirParameters::from_native(1024.0, 1024.0, width, 1.0 / 64.0, n, 0.125);
    Ok([
        native(32.0, 1.0 / 32.0)?,
        native(0.0, 1.0 / 32.0)?,
        native(32.0, 0.0)?,
    ])
}

/// Rain, evaporation and runon for one subcatchment.
pub fn subcatchment_input(rain: f64, evaporation: f64, runon: [f64; 3]) -> SubcatchmentInput {
    SubcatchmentInput {
        rain_rate_m_per_s: rain,
        potential_evaporation_m_per_s: evaporation,
        external_runon_volumes_m3: runon,
    }
}

/// Runs one subcatchment trial to `end`; a rejection must leave accepted state unchanged.
pub fn subcatchment_outcome(batch: &Subcatchment, end: f64, input: SubcatchmentInput) -> Outcome {
    let before = batch.snapshot_data()?;
    let outcome = match batch.trial_to(end, &[input]) {
        Ok(trial) => {
            trial.diagnostics()?;
            None
        }
        Err(error) => Some((subcatchment_variant(&error), error.to_string())),
    };
    assert_eq!(batch.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Constructs a runoff/pool batch and runs one trial to `end`, keeping accepted state on rejection.
pub fn runoff_pool_outcome(
    setup: RunoffPoolSetup,
    initial: RunoffPoolInitialState,
    end: f64,
    forcing: &RunoffPoolForcingInterval,
) -> Outcome {
    let batch = match RunoffPools::new(crate::runtime::client(), setup, initial) {
        Ok(batch) => batch,
        Err(error) => return Ok(Some((runoff_pool_variant(&error), error.to_string()))),
    };
    let before = batch.snapshot_data()?;
    let outcome = match batch.trial_with_forcing_to(end, forcing) {
        Ok(trial) => {
            trial.diagnostics()?;
            None
        }
        Err(error) => Some((runoff_pool_variant(&error), error.to_string())),
    };
    assert_eq!(batch.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Asserts that `outcome` admits its case.
pub fn admitted(outcome: Outcome, case: &str) -> TestResult {
    match outcome? {
        None => Ok(()),
        Some((variant, reason)) => Err(format!("{case}: {variant:?} {reason}").into()),
    }
}
