//! Supplied opening upload or retained trajectory tensors for immutable proposals.

use crate::cpu::{
    CpuTensor,
    structures::{CpuStructureContext, operations::CheckedOperations},
};
use crate::structures::{StructureError, StructureOperatingPoint};
use numr::ops::ConditionalOps;

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
        let values = (0..operations.len())
            .map(|index| match operations.supplied(index) {
                Some(StructureOperatingPoint::GateOpening { opening_m }) => opening_m,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        let manual = context.tensor(&values)?;
        let opening = match operations.derived_opening() {
            Some((derived, mask)) => context.client.where_cond(mask, derived, &manual)?,
            None => manual,
        };
        context.finite(&opening, "assembled finite actual gate opening")?;
        Ok(Some(Self { opening }))
    }
}
