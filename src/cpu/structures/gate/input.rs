//! Original admitted gate openings uploaded once per immutable proposal.

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
        let values = operations
            .packets()
            .iter()
            .map(|operation| match operation.operating_point {
                StructureOperatingPoint::GateOpening { opening_m } => opening_m,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        Ok(Some(Self {
            opening: context.tensor(&values)?,
        }))
    }
}
