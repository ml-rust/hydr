//! Explicit immutable physical setup and accepted initial state.

use crate::contract::{BoundaryConditions, ForcingSchedule, SolverConfig};
use crate::model::{Domain, FlowState};
use crate::soil::{LandModel, SoilState};

/// Complete physical and numerical definition of an engine.
#[derive(Clone, Debug, PartialEq)]
pub struct SolverSetup {
    /// Hydraulic geometry, reference, and roughness.
    pub domain: Domain,
    /// Explicit per-cell physical land mechanism.
    pub land: LandModel,
    /// Declared exterior boundary support.
    pub boundaries: BoundaryConditions,
    /// Explicit bounded rain, evaporation, withdrawal, and boundary forcing.
    pub forcing: ForcingSchedule,
    /// Numerical controls and adjustment budget.
    pub config: SolverConfig,
}

/// Complete accepted initial state without an inferred antecedent soil condition.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialState {
    /// Elapsed time in seconds.
    pub time: f64,
    /// Hydraulic depth and discharge.
    pub flow: FlowState,
    /// Complete native antecedent soil state.
    pub soil: SoilState,
}
