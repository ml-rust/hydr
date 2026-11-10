//! Atomic restoration preserves both destination child contexts.

use super::super::ControlledGateBatch;
use super::{ControlledGateSnapshot, ControlledGateSnapshotData};
use crate::structures::StructureError;

impl ControlledGateBatch {
    /// Restore trusted child tensors atomically while retaining this composite's immutable contexts.
    pub fn restore(&mut self, snapshot: &ControlledGateSnapshot) -> Result<(), StructureError> {
        let mut controls = self.controls.clone();
        let mut actuators = self.actuators.clone();
        controls.restore(&snapshot.batch.controls.snapshot())?;
        actuators.restore(&snapshot.batch.actuators.snapshot())?;
        super::check::metadata(&controls, &actuators)?;
        self.controls = controls;
        self.actuators = actuators;
        Ok(())
    }
    /// Check both complete child packages and their correspondence before atomically replacing either child.
    ///
    /// Bounded checks establish continuation consistency. They establish no full historical authenticity.
    pub fn restore_data(
        &mut self,
        data: &ControlledGateSnapshotData,
    ) -> Result<(), StructureError> {
        let mut controls = self.controls.clone();
        let mut actuators = self.actuators.clone();
        controls.restore_data(&data.controls)?;
        actuators.restore_data(&data.actuators)?;
        super::check::data(self, data)?;
        self.controls = controls;
        self.actuators = actuators;
        Ok(())
    }
}
