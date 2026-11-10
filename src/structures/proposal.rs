//! Opaque immutable proposal tensors with explicit output projection.

use super::{StructureError, StructureHeads, StructureOutput};
use crate::cpu::structures::{CpuStructureContext, CpuStructureResult};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct ProposalData {
    pub context: Arc<CpuStructureContext>,
    pub heads: Vec<StructureHeads>,
    pub result: CpuStructureResult,
    pub operations: super::operation_projection::ProposalOperations,
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
                operations: super::operation_projection::ProposalOperations::Supplied(operations),
            }),
        }
    }
    pub(crate) fn from_trajectory(
        context: Arc<CpuStructureContext>,
        heads: Vec<StructureHeads>,
        evaluation: Arc<super::GateTrajectoryEvaluationData>,
        other: Vec<super::StructureOperation>,
        result: CpuStructureResult,
    ) -> Self {
        Self {
            inner: Arc::new(ProposalData {
                context,
                heads,
                result,
                operations: super::operation_projection::ProposalOperations::Trajectory {
                    evaluation,
                    other,
                },
            }),
        }
    }
    pub(crate) fn from_derived(
        context: Arc<CpuStructureContext>,
        heads: Vec<StructureHeads>,
        manual: Vec<super::StructureOperation>,
        derived_slots: Vec<usize>,
        opening: crate::cpu::CpuTensor,
        result: CpuStructureResult,
    ) -> Self {
        Self {
            inner: Arc::new(ProposalData {
                context: Arc::clone(&context),
                heads,
                result,
                operations: super::operation_projection::ProposalOperations::Derived {
                    context,
                    slots: derived_slots,
                    opening,
                    manual,
                },
            }),
        }
    }
    /// Project complete ordered operations without repeating trajectory evaluation.
    pub fn operations(&self) -> Result<Vec<super::StructureOperation>, StructureError> {
        super::operation_projection::project(&self.inner.operations)
    }
    /// Project original support, physical regime, ideal-flap status, and active law operands.
    pub fn outputs(&self) -> Result<Vec<StructureOutput>, StructureError> {
        super::projection::outputs(&self.inner)
    }
}
