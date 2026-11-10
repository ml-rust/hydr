//! Complete response assumptions and control continuation with atomic restoration.

use super::{ControlledPumpBatch, PumpResponseDefinition};
use crate::structures::{OperatingControlSnapshotData, StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Complete caller-owned response assumptions and the existing controller continuation.
///
/// Preserve every original control F64 bit during encoding. No independent response state exists.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledPumpSnapshotData {
    /// Complete ordered original response models and explicit applicability evidence.
    pub definitions: Vec<PumpResponseDefinition>,
    /// Complete original controller model, accepted memory, calendar, and bounded receipts.
    pub controls: OperatingControlSnapshotData,
}
/// Trusted complete owner clone without numerical projection or prior-owner links.
#[derive(Clone, Debug)]
pub struct ControlledPumpSnapshot<R: Runtime<DType = DType>> {
    batch: ControlledPumpBatch<R>,
}
impl<R: Runtime<DType = DType>> ControlledPumpSnapshot<R> {
    /// Accepted original snapshot time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
    /// Explicitly project complete response assumptions and controller continuation.
    pub fn data(&self) -> Result<ControlledPumpSnapshotData, StructureError> {
        self.batch.snapshot_data()
    }
}
fn require(condition: bool, message: &str) -> Result<(), StructureError> {
    if condition {
        Ok(())
    } else {
        Err(StructureError::input(
            StructureInputErrorKind::Definition,
            message,
        ))
    }
}
impl<R: Runtime<DType = DType>> ControlledPumpBatch<R> {
    /// Clone complete accepted continuation without derived-state array readback.
    pub fn snapshot(&self) -> ControlledPumpSnapshot<R> {
        ControlledPumpSnapshot {
            batch: self.clone(),
        }
    }
    /// Explicitly project response assumptions and complete original control continuation.
    pub fn snapshot_data(&self) -> Result<ControlledPumpSnapshotData, StructureError> {
        Ok(ControlledPumpSnapshotData {
            definitions: self.definitions().to_vec(),
            controls: self.controls.snapshot_data()?,
        })
    }
    /// Restore trusted control continuation while retaining the complete destination response context.
    pub fn restore(&mut self, snapshot: &ControlledPumpSnapshot<R>) -> Result<(), StructureError> {
        require(
            super::compatibility::same_context(&self.context, &snapshot.batch.context),
            "controlled pump snapshot response assumptions or immutable controller model differ",
        )?;
        let mut controls = self.controls.clone();
        controls.restore(&snapshot.batch.controls.snapshot())?;
        self.controls = controls;
        Ok(())
    }
    /// Check response assumptions and complete caller-owned child continuation before atomic replacement.
    ///
    /// Child bounded consistency checks establish no complete historical authenticity.
    pub fn restore_data(&mut self, data: &ControlledPumpSnapshotData) -> Result<(), StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        require(
            self.definitions() == data.definitions.as_slice(),
            "controlled pump snapshot original response assumptions differ",
        )?;
        let mut controls = self.controls.clone();
        controls.restore_data(&data.controls)?;
        self.controls = controls;
        Ok(())
    }
}
