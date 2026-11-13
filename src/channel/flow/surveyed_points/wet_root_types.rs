use super::{
    root_invocation::{RootInvocation, RootInvocationWitness},
    wave_curve_types::CurveValue,
};
use crate::channel::section::SupportedAreaQuery;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum WetRootStatus {
    Located = 0,
    Vacuum = 1,
    TouchingDry = 2,
    CapExceeded = 3,
    Unresolved = 4,
    WorkLimit = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WetRootEvidenceKind {
    Uniform,
    DryLimit,
    Cap,
}

pub(super) struct WetRootEvidence<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) kind: WetRootEvidenceKind,
    pub(super) value: CurveValue<R>,
}

/// A certified existence domain, without a selected center or hydraulic convergence claim.
pub(super) struct WetRootDomain<R: Runtime<DType = DType>> {
    pub(super) invocation_witness: RootInvocationWitness,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    /// Bounds of the increasing chart residual. Subtract charts describe negative physical F.
    pub(super) lower_residual: Interval<R>,
    pub(super) upper_residual: Interval<R>,
    /// Bracket-derived values lift the complete intervals, without restoring original exact residual identity.
    pub(super) lower_retained: CertifiedExpansion<R>,
    pub(super) upper_retained: CertifiedExpansion<R>,
}

pub(super) struct WetRootDomains<R: Runtime<DType = DType>> {
    pub(super) invocation: RootInvocation,
    pub(super) invocation_rows: Tensor<R>,
    pub(super) status: Tensor<R>,
    pub(super) domains: Vec<WetRootDomain<R>>,
    pub(super) evidence: Vec<WetRootEvidence<R>>,
}

impl<R: Runtime<DType = DType>> WetRootDomains<R> {
    pub(super) fn callback_points(&self) -> usize {
        self.invocation.callback_points()
    }
    pub(super) fn point_limit(&self) -> usize {
        self.invocation.point_limit()
    }
}
