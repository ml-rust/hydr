//! Opaque immutable proposal tensors with explicit output projection.
use numr::tensor::Tensor;
use numr::{dtype::DType, runtime::Runtime};

use super::{StructureError, StructureHeads, StructureOutput};
use crate::tensor::structures::{TensorStructureContext, TensorStructureResult};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct ProposalData<R: Runtime<DType = DType>> {
    pub context: Arc<TensorStructureContext<R>>,
    pub heads: Vec<StructureHeads>,
    pub result: TensorStructureResult<R>,
    pub operations: super::operation_projection::ProposalOperations<R>,
}

/// Immutable constitutive proposals. No donor availability or paired exchange acceptance follows.
#[derive(Clone, Debug)]
pub struct StructureProposal<R: Runtime<DType = DType>> {
    pub(super) inner: Arc<ProposalData<R>>,
}
impl<R: Runtime<DType = DType>> StructureProposal<R> {
    pub(super) fn new(
        context: Arc<TensorStructureContext<R>>,
        heads: Vec<StructureHeads>,
        operations: Vec<super::StructureOperation>,
        result: TensorStructureResult<R>,
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
        context: Arc<TensorStructureContext<R>>,
        heads: Vec<StructureHeads>,
        evaluation: Arc<super::GateTrajectoryEvaluationData<R>>,
        other: Vec<super::StructureOperation>,
        result: TensorStructureResult<R>,
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
        context: Arc<TensorStructureContext<R>>,
        heads: Vec<StructureHeads>,
        manual: Vec<super::StructureOperation>,
        derived_slots: Vec<usize>,
        opening: Tensor<R>,
        pumps: Option<crate::tensor::structures::DerivedPumpOperationData<R>>,
        result: TensorStructureResult<R>,
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
                    pumps: pumps.map(Box::new),
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
