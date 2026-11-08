use super::super::engine::{Accepted, Definition};
use super::super::{
    BoundaryConditions, ForcingSchedule, SolverConfig, StepDiagnostics, WaterLedger,
};
use crate::model::{Domain, FlowState};
use std::sync::Arc;

/// Typed checkpoint schema identity.
pub const SNAPSHOT_SCHEMA: &str = "hydr-checkpoint-1";

/// Caller-owned checkpoint values for same-build restart.
///
/// Adapters must preserve every f64 bit, including positive infinity in timestep bounds.
/// Callers own encoding, persistence, and checksums. Import checks consistency, not historical authenticity.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Immutable domain and spatial metadata.
    pub domain: Domain,
    /// Declared walls and ports.
    pub boundaries: BoundaryConditions,
    /// Complete bounded forcing schedule.
    pub forcing: ForcingSchedule,
    /// Numerical controls.
    pub config: SolverConfig,
    /// Accepted elapsed time in seconds.
    pub time: f64,
    /// Accepted cell fields.
    pub state: FlowState,
    /// Cumulative water accounting.
    pub ledger: WaterLedger,
    /// Number of accepted steps.
    pub accepted_steps: usize,
    /// Last accepted interval, absent before the first step.
    pub last_step: Option<StepDiagnostics>,
}

/// Complete immutable in-memory continuation state for one numerical contract.
///
/// Includes accepted fields, time, cumulative ledgers, last-step diagnostics, step history,
/// immutable geometry, parameters, boundaries, and the complete forcing schedule.
/// Caller adapters own file encoding and persistence.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub(in crate::solver) contract: &'static str,
    pub(in crate::solver) definition: Arc<Definition>,
    pub(in crate::solver) accepted: Accepted,
}

impl Snapshot {
    /// Copies complete checkpoint values for caller-owned encoding and persistence.
    pub fn data(&self) -> SnapshotData {
        SnapshotData {
            schema: SNAPSHOT_SCHEMA.into(),
            contract: self.contract.into(),
            domain: self.definition.domain.clone(),
            boundaries: self.definition.boundaries.clone(),
            forcing: self.definition.forcing.clone(),
            config: self.definition.config.clone(),
            time: self.accepted.time,
            state: self.accepted.state.clone(),
            ledger: self.accepted.ledger.clone(),
            accepted_steps: self.accepted.steps,
            last_step: self.accepted.last.clone(),
        }
    }
    /// Returns the snapshot's accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.accepted.time
    }

    /// Returns the snapshot's numerical contract identity.
    pub fn contract(&self) -> &'static str {
        self.contract
    }
}
