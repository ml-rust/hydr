//! Supplied opening upload or retained trajectory tensors for immutable proposals.

use crate::cpu::{
    CpuTensor,
    structures::{CpuStructureContext, operations::CheckedOperations},
};
use crate::structures::{StructureError, StructureOperatingPoint};

#[derive(Debug)]
pub(crate) struct CpuGateInput {
    pub opening: CpuTensor,
}
impl CpuGateInput {
    pub(crate) fn new(
        context: &CpuStructureContext,
        operations: &CheckedOperations<'_>,
    ) -> Result<Option<Self>, StructureError> {
        if context.gate.is_none() {
            return Ok(None);
        }
        if let Some(opening) = operations.opening() {
            return Ok(Some(Self {
                opening: opening.clone(),
            }));
        }
        let values = (0..operations.len())
            .map(|index| match operations.supplied(index) {
                Some(StructureOperatingPoint::GateOpening { opening_m }) => opening_m,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        Ok(Some(Self {
            opening: context.tensor(&values)?,
        }))
    }
}
