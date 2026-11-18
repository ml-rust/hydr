use super::super::definition::Definition;
use super::super::engine::Accepted;
use super::super::receipts::StepReceipt;
use super::super::{ForcingPrefix, InitialState, SolverSetup, StepDiagnostics, WaterLedger};
use crate::contract::EngineError;
use crate::solver::tensor::TensorDomain;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Typed checkpoint schema identity.
pub const SNAPSHOT_SCHEMA: &str = "hydr-checkpoint-3";

/// Caller-owned checkpoint values for same-build restart.
///
/// Adapters must preserve every f64 bit, including positive infinity in timestep bounds.
/// Callers own encoding, persistence, and checksums. Import checks consistency, not historical authenticity.
///
/// `setup.forcing` holds a bounded window, never the complete forcing schedule.
/// The window starts at the interval containing the last step start.
/// Without a last step it starts at the interval containing the accepted time, or the final interval.
/// `forcing_prefix` folds every earlier interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Immutable numerical definition, including land and the retained forcing window.
    pub setup: SolverSetup,
    /// Folded identity of every forcing interval before `setup.forcing`.
    pub forcing_prefix: ForcingPrefix,
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
/// Includes accepted tensors, time, cumulative ledgers, and bounded last-step receipts.
/// Retains immutable geometry, parameters, boundaries, and the engine's forcing window with its prefix.
/// [`Snapshot::data`] projects a bounded forcing window. Caller adapters own file encoding and persistence.
#[derive(Clone)]
pub struct Snapshot<R: Runtime<DType = DType>> {
    pub(in crate::solver) context: Arc<TensorDomain<R>>,
    pub(in crate::solver) contract: &'static str,
    pub(in crate::solver) definition: Arc<Definition>,
    pub(in crate::solver) accepted: Accepted<R>,
}

impl<R: Runtime<DType = DType>> Snapshot<R>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::UnaryOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::HypotOps<R>
        + numr::ops::ShapeOps<R>
        + solvr::optimize::roots::BatchedBisectionAlgorithms<R>
        + solvr::optimize::roots::BatchedNewtonAlgorithms<R>,
{
    /// Projects complete checkpoint values for caller-owned encoding and persistence.
    ///
    /// Forcing is bounded to the retained window plus a folded prefix. See [`SnapshotData`].
    ///
    /// # Errors
    /// Returns runtime transfer, physical state, or receipt shape errors.
    pub fn data(&self) -> Result<SnapshotData, EngineError> {
        let definition = &self.definition;
        let retained = definition.retained_from(
            self.accepted.time,
            self.accepted.last.as_ref().map(StepReceipt::start),
        );
        let window = || {
            EngineError::rejected(
                "snapshot forcing window",
                format!(
                    "first retained interval {retained} of {}: capture the snapshot from a checked engine",
                    definition.forcing.intervals().len()
                ),
            )
        };
        Ok(SnapshotData {
            schema: SNAPSHOT_SCHEMA.into(),
            contract: self.contract.into(),
            setup: SolverSetup {
                domain: definition.domain.clone(),
                land: definition.land.clone(),
                boundaries: definition.boundaries.clone(),
                forcing: definition.forcing.window(retained).ok_or_else(window)?,
                config: definition.config.clone(),
            },
            forcing_prefix: definition.folded(retained).ok_or_else(window)?,
            accepted: InitialState {
                time: self.accepted.time,
                flow: self.accepted.flow.to_flow(&self.definition.domain)?,
                soil: self.accepted.land.soil_state()?,
            },
            ledger: self.accepted.ledger.clone(),
            accepted_steps: self.accepted.steps,
            last_step: self
                .accepted
                .last
                .as_ref()
                .map(|receipt| receipt.project(&self.context, &self.accepted.land))
                .transpose()?,
        })
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

impl<R: Runtime<DType = DType>> std::fmt::Debug for Snapshot<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("runtime", &R::name())
            .field("time", &self.accepted.time)
            .field("steps", &self.accepted.steps)
            .finish_non_exhaustive()
    }
}
