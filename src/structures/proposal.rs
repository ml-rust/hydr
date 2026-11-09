//! Opaque immutable proposal tensors with explicit output projection.

use super::{StructureError, StructureHeads, StructureOutput};
use crate::cpu::structures::{CpuStructureContext, CpuStructureResult};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct ProposalData {
    pub context: Arc<CpuStructureContext>,
    pub heads: Vec<StructureHeads>,
    pub result: CpuStructureResult,
    pub operations: Vec<super::StructureOperation>,
}

/// Immutable constitutive proposals. No donor availability or paired exchange acceptance follows.
#[derive(Clone, Debug)]
pub struct StructureProposal {
    pub(super) inner: Arc<ProposalData>,
}
impl StructureProposal {
    pub(super) fn new(
        context: Arc<CpuStructureContext>,
        heads: Vec<StructureHeads>,
        operations: Vec<super::StructureOperation>,
        result: CpuStructureResult,
    ) -> Self {
        Self {
            inner: Arc::new(ProposalData {
                context,
                heads,
                result,
                operations,
            }),
        }
    }
    /// Project original support, physical regime, ideal-flap status, and active law operands.
    pub fn outputs(&self) -> Result<Vec<StructureOutput>, StructureError> {
        super::projection::outputs(&self.inner)
    }
}
