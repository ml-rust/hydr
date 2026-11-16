#[cfg(test)]
use super::wet_root_types::WetRootEvidence;
use super::{
    root_invocation::{RootInvocation, RootInvocationWitness},
    wet_root_types::WetRootDomain,
    wet_star_levels::WetStarLevels,
};
use crate::channel::section::SupportedAreaQuery;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::{
    common::arithmetic::Interval,
    optimize::{CertifiedRootContinuation, CertifiedRootTolerance},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum WetDepthStatus {
    DepthConverged = 0,
    AccuracyFloor = 1,
    WorkLimit = 2,
    Unresolved = 3,
}

pub(super) enum DepthRefinementState<R: Runtime<DType = DType>> {
    Deferred {
        original: Box<WetRootDomain<R>>,
    },
    Initialized {
        state: Box<InitializedDepthState<R>>,
    },
}

pub(super) struct InitializedDepthState<R: Runtime<DType = DType>> {
    pub(super) invocation_witness: RootInvocationWitness,
    pub(super) continuation: CertifiedRootContinuation<R>,
    pub(super) chart: SupportedAreaQuery<R>,
}

/// Physical certificates retain their area expression independently of normalized-depth projection.
pub(super) struct WetDepthDomain<R: Runtime<DType = DType>> {
    pub(super) invocation_witness: RootInvocationWitness,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) depth: Interval<R>,
    pub(super) normalized_depth: Interval<R>,
    pub(super) status: Tensor<R>,
    /// Simpson panel level of each row's star residual and star endpoints.
    pub(super) levels: WetStarLevels<R>,
    pub(super) iterations: usize,
    pub(super) callback_points: usize,
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(super) mapped_points: usize,
    pub(super) refinement: DepthRefinementState<R>,
}

impl<R: Runtime<DType = DType>> WetDepthDomain<R> {
    pub(super) fn check_invocation(
        &self,
        invocation: &RootInvocation,
    ) -> crate::channel::flow::surveyed::error::HydraulicResult<()> {
        invocation.check(&self.invocation_witness)?;
        match &self.refinement {
            DepthRefinementState::Deferred { original } => {
                invocation.check(&original.invocation_witness)
            }
            DepthRefinementState::Initialized { state } => {
                invocation.check(&state.invocation_witness)
            }
        }
    }

    pub(super) fn is_deferred(&self) -> bool {
        matches!(self.refinement, DepthRefinementState::Deferred { .. })
    }
}

pub(super) struct WetDepthState<R: Runtime<DType = DType>> {
    pub(super) locator_status: Tensor<R>,
    #[cfg(test)]
    pub(super) evidence: Vec<WetRootEvidence<R>>,
    pub(super) domains: Vec<WetDepthDomain<R>>,
    pub(super) tolerance: CertifiedRootTolerance,
    pub(super) terminal: bool,
}

pub(super) type WetDepthSession<'s, 'a, R> = super::point_root_session::PointRootSession<'s, 'a, R>;
