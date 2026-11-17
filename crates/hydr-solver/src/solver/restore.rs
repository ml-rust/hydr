//! Trusted in-memory restoration from a same-device snapshot.

use super::compatibility::Candidate;
use super::receipts::{StepReceipt, rebase_forcing_index};
use super::snapshot::reject_exchange;
use super::{CONTRACT_VERSION, Snapshot, Solver};
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + numr::ops::CumulativeOps<R>
        + numr::ops::ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Restores complete accepted state after checking the immutable numerical definition.
    ///
    /// The snapshot and this engine can hold different forcing windows of one schedule.
    /// Both histories fold to the later prefix, then the folded prefixes and remaining intervals compare bitwise.
    /// This engine's window must hold the interval of the snapshot's last step.
    ///
    /// A snapshot with coordinator exchange water restores only through its coordinator.
    ///
    /// # Errors
    /// Rejects different devices, geometry, parameters, boundaries, forcing, contracts, an uncovered
    /// window, or coordinator exchange before mutation.
    pub fn restore(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
        // Cumulative exchange covers the last step, so the ledger alone decides.
        reject_exchange(&snapshot.accepted.ledger, None)?;
        self.restore_snapshot(snapshot)
    }

    /// Restores a snapshot that can carry coordinator exchange water.
    ///
    /// Runs every [`Solver::restore`] check except the standalone exchange rejection.
    ///
    /// # Errors
    /// Rejects different devices, geometry, parameters, boundaries, forcing, contracts, or an uncovered window before mutation.
    #[doc(hidden)]
    pub fn coupling_restore(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
        self.restore_snapshot(snapshot)
    }

    fn restore_snapshot(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
        let source = &snapshot.definition;
        let retained = source.retained_from(
            snapshot.accepted.time,
            snapshot.accepted.last.as_ref().map(StepReceipt::start),
        );
        if !self
            .context
            .client
            .device()
            .is_same(snapshot.context.client.device())
            || snapshot.contract != CONTRACT_VERSION
            || !self
                .definition
                .equivalent(&Candidate::of(source, retained))?
        {
            return Err(EngineError::rejected(
                "snapshot",
                "definition or contract mismatch: restore into an engine built from the same definition and forcing schedule",
            ));
        }
        let mut land = self.accepted.land.clone();
        land.restore_trusted(&snapshot.accepted.land)?;
        let mut accepted = snapshot.accepted.clone();
        accepted.land = land;
        if let Some(StepReceipt::Device(receipt)) = &mut accepted.last {
            receipt.forcing_index = rebase_forcing_index(
                receipt.forcing_index,
                source.prefix.intervals,
                self.definition.prefix.intervals,
                self.context.sources.len(),
            )?;
            receipt.sources.soil.next = accepted.land.clone();
        }
        self.accepted = accepted;
        Ok(())
    }
}
