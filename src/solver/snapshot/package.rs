use super::super::engine::{Accepted, Definition};
use super::super::{InitialState, SolverSetup, StepDiagnostics, WaterLedger};
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
    /// Complete immutable numerical definition, including land and forcing.
    pub setup: SolverSetup,
    /// Accepted elapsed time, hydraulic fields, and complete native soil memory.
    pub accepted: InitialState,
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
            setup: SolverSetup {
                domain: self.definition.domain.clone(),
                land: self.definition.land.clone(),
                boundaries: self.definition.boundaries.clone(),
                forcing: self.definition.forcing.clone(),
                config: self.definition.config.clone(),
            },
            accepted: InitialState {
                time: self.accepted.time,
                flow: self.accepted.state.clone(),
                soil: self.accepted.soil.clone(),
            },
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
