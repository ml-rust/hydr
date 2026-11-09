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
        if self.context.elements.iter().any(|element| {
            matches!(
                element.law,
                super::StructureLaw::Pump(_) | super::StructureLaw::HydrostaticRectangularGate(_)
            )
        }) {
            return Err(StructureError::input(
                super::StructureInputErrorKind::OperatingPoint,
                "device operations are missing. Use propose_with_operations with explicit identity-bound operations",
            ));
        }
        let operations = self
            .elements()
            .iter()
            .map(|e| super::StructureOperation {
                structure_id: e.id.clone(),
                operating_point: super::StructureOperatingPoint::Passive,
            })
            .collect::<Vec<_>>();
        self.propose_with_operations(heads, &operations)
    }
    /// Evaluate complete identity-bound device operations without mutating earlier proposals.
    pub fn propose_with_operations(
        &self,
        heads: &[StructureHeads],
        operations: &[super::StructureOperation],
    ) -> Result<StructureProposal, StructureError> {
        let input = CpuStructureInput::with_operations(&self.context, heads, operations)?;
        let passive = crate::numerics::structures::evaluate(&self.context, &input)?;
        let result = crate::numerics::structures::pump::evaluate(&self.context, &input, passive)?;
        let result = crate::numerics::structures::gate::evaluate(&self.context, &input, result)?;
        Ok(StructureProposal::new(
            Arc::clone(&self.context),
            heads.to_vec(),
            operations.to_vec(),
            result,
        ))
    }
}
