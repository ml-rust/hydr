//! Atomic restoration preserves both destination child contexts.

use super::super::ControlledGateBatch;
use super::{
    CONTROLLED_GATE_CONTRACT_VERSION, CONTROLLED_GATE_SNAPSHOT_SCHEMA, ControlledGateSnapshot,
    ControlledGateSnapshotData,
};
use crate::structures::{StructureError, StructureInputErrorKind};
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> ControlledGateBatch<R> {
    /// Restore trusted child tensors atomically while retaining this composite's immutable contexts.
    pub fn restore(&mut self, snapshot: &ControlledGateSnapshot<R>) -> Result<(), StructureError> {
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
    pub fn restore_data(&mut self, data: &ControlledGateSnapshotData) -> Result<(), StructureError>
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
        package_identity(
            "controlled gate",
            &data.schema,
            CONTROLLED_GATE_SNAPSHOT_SCHEMA,
            &data.contract,
            CONTROLLED_GATE_CONTRACT_VERSION,
        )
        .map_err(|message| StructureError::input(StructureInputErrorKind::Definition, message))?;
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
