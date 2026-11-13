use super::{
    paired_sources::PairedSources,
    source_binding::RootSourceWitness,
    wet_root_types::{WetRootDomain, WetRootEvidence},
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
    pub(super) continuation: CertifiedRootContinuation<R>,
    pub(super) chart: SupportedAreaQuery<R>,
}

/// Physical certificates retain their area expression independently of normalized-depth projection.
pub(super) struct WetDepthDomain<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) depth: Interval<R>,
    pub(super) normalized_depth: Interval<R>,
    pub(super) status: Tensor<R>,
    pub(super) iterations: usize,
    pub(super) callback_points: usize,
    pub(super) mapped_points: usize,
    pub(super) refinement: DepthRefinementState<R>,
}

impl<R: Runtime<DType = DType>> WetDepthDomain<R> {
    pub(super) fn is_deferred(&self) -> bool {
        matches!(self.refinement, DepthRefinementState::Deferred { .. })
    }
}

/// Own depth evidence and numerical continuations while exclusively borrowing the fixed source graph.
pub(super) struct WetDepthSession<'s, 'a, R: Runtime<DType = DType>> {
    pub(super) sources: &'s mut PairedSources<'a, R>,
    pub(super) witness: RootSourceWitness,
    pub(super) invocation_rows: Tensor<R>,
    pub(super) locator_status: Tensor<R>,
    pub(super) evidence: Vec<WetRootEvidence<R>>,
    pub(super) domains: Vec<WetDepthDomain<R>>,
    pub(super) point_limit: usize,
    pub(super) callback_points: usize,
    pub(super) tolerance: CertifiedRootTolerance,
}

impl<'s, 'a, R: Runtime<DType = DType>> WetDepthSession<'s, 'a, R> {
    pub(super) fn sources(&self) -> &PairedSources<'a, R> {
        self.sources
    }
    pub(super) fn invocation_rows(&self) -> &Tensor<R> {
        &self.invocation_rows
    }
    pub(super) fn locator_status(&self) -> &Tensor<R> {
        &self.locator_status
    }
    pub(super) fn evidence(&self) -> &[WetRootEvidence<R>] {
        &self.evidence
    }
    pub(super) fn domains(&self) -> &[WetDepthDomain<R>] {
        &self.domains
    }
    pub(super) fn point_limit(&self) -> usize {
        self.point_limit
    }
    pub(super) fn callback_points(&self) -> usize {
        self.callback_points
    }
    pub(super) fn tolerance(&self) -> &CertifiedRootTolerance {
        &self.tolerance
    }
}
