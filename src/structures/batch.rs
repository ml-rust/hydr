//! Immutable typed law batches and original supported-head proposals.

use super::{StructureElement, StructureError, StructureHeads, StructureProposal};
use crate::cpu::structures::{CpuStructureContext, CpuStructureInput};
use std::sync::Arc;

/// Persistent checked law definitions without evolving component storage.
#[derive(Clone, Debug)]
pub struct StructureBatch {
    pub(super) context: Arc<CpuStructureContext>,
}
impl StructureBatch {
    /// Check explicit law geometry, applicability, identities, and ideal direction policies.
    pub fn new(elements: Vec<StructureElement>) -> Result<Self, StructureError> {
        Ok(Self {
            context: Arc::new(CpuStructureContext::new(elements)?),
        })
    }
    /// Immutable concrete definitions in original numerical order.
    pub fn elements(&self) -> &[StructureElement] {
        &self.context.elements
    }
    /// Evaluate original supported heads without mutating definitions or earlier proposals.
    pub fn propose(&self, heads: &[StructureHeads]) -> Result<StructureProposal, StructureError> {
        let input = CpuStructureInput::new(&self.context, heads)?;
        let result = crate::numerics::structures::evaluate(&self.context, &input)?;
        Ok(StructureProposal::new(
            Arc::clone(&self.context),
            heads.to_vec(),
            result,
        ))
    }
}
