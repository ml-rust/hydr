//! Complete owning child packages without an extra decision-history bridge.

use super::super::ControlledGateBatch;
use crate::structures::{GateActuatorSnapshotData, OperatingControlSnapshotData, StructureError};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Schema identity of the controlled gate snapshot package.
pub const CONTROLLED_GATE_SNAPSHOT_SCHEMA: &str = "hydr-controlled-gate-checkpoint-1";
/// Numerical contract identity of the controlled gate snapshot package.
pub const CONTROLLED_GATE_CONTRACT_VERSION: &str = "hydr-controlled-gate-numr-cpu-1";

/// Complete caller-owned sampled controls and actual gate response continuation.
///
/// Encoding must preserve every child field and original F64 bit, including signed zeros.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledGateSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Complete original model, current control memory, and bounded accepted decision receipts.
    pub controls: OperatingControlSnapshotData,
    /// Complete original model, actual position proofs, and bounded accepted response receipts.
    pub actuators: GateActuatorSnapshotData,
}
/// Trusted complete composite clone without numerical projection or recursive prior composites.
#[derive(Clone, Debug)]
pub struct ControlledGateSnapshot<R: Runtime<DType = DType>> {
    pub(super) batch: ControlledGateBatch<R>,
}
impl<R: Runtime<DType = DType>> ControlledGateSnapshot<R> {
    /// Common accepted snapshot time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
    /// Explicitly project both complete caller-owned packages.
    pub fn data(&self) -> Result<ControlledGateSnapshotData, StructureError> {
        self.batch.snapshot_data()
    }
}
impl<R: Runtime<DType = DType>> ControlledGateBatch<R> {
    /// Clone both complete accepted children without derived-state array readback.
    pub fn snapshot(&self) -> ControlledGateSnapshot<R> {
        ControlledGateSnapshot {
            batch: self.clone(),
        }
    }
    /// Explicitly project complete child continuation without inventing additional historical evidence.
    pub fn snapshot_data(&self) -> Result<ControlledGateSnapshotData, StructureError> {
        Ok(ControlledGateSnapshotData {
            schema: CONTROLLED_GATE_SNAPSHOT_SCHEMA.into(),
            contract: CONTROLLED_GATE_CONTRACT_VERSION.into(),
            controls: self.controls.snapshot_data()?,
            actuators: self.actuators.snapshot_data()?,
        })
    }
}
