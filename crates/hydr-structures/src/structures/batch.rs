//! Immutable typed law batches and original supported-head proposals.
use numr::{dtype::DType, runtime::Runtime};

use super::{StructureElement, StructureError, StructureHeads, StructureProposal};
use crate::structures::tensor::{TensorStructureContext, TensorStructureInput};
use std::sync::Arc;

/// Persistent checked law definitions without evolving component storage.
#[derive(Clone, Debug)]
pub struct StructureBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<TensorStructureContext<R>>,
}
impl<R: Runtime<DType = DType>> StructureBatch<R> {
    /// Check explicit law geometry, applicability, identities, and ideal direction policies.
    pub fn new(client: R::Client, elements: Vec<StructureElement>) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        Ok(Self {
            context: Arc::new(TensorStructureContext::new(client, elements)?),
        })
    }
    /// Immutable concrete definitions in original numerical order.
    pub fn elements(&self) -> &[StructureElement] {
        &self.context.elements
    }
    /// Evaluate original supported heads without mutating definitions or earlier proposals.
    pub fn propose(&self, heads: &[StructureHeads]) -> Result<StructureProposal<R>, StructureError>
    where
        R::Client: numr::ops::CumulativeOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::ShapeOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>
            + solvr::interpolate::Interp1dAlgorithms<R>,
    {
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
    ) -> Result<StructureProposal<R>, StructureError>
    where
        R::Client: numr::ops::CumulativeOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::ShapeOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>
            + solvr::interpolate::Interp1dAlgorithms<R>,
    {
        let input = TensorStructureInput::with_operations(&self.context, heads, operations)?;
        let result = crate::structures::numerics::proposal(&self.context, &input)?;
        Ok(StructureProposal::new(
            Arc::clone(&self.context),
            heads.to_vec(),
            operations.to_vec(),
            result,
        ))
    }
}
