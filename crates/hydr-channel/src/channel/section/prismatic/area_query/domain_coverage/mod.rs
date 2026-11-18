mod admit;
mod align;
mod cover;
#[cfg(test)]
mod fixture;
mod proof;

pub(crate) use cover::cover_domain;

use super::chart::SupportedAreaQuery;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::CertifiedExpansion;

/// Coverage decisions in invocation order. These decisions do not establish root existence.
pub(crate) struct DomainCoverage<R: Runtime<DType = DType>> {
    /// Mask in input order: one for proved coverage, zero otherwise.
    #[cfg(test)]
    pub(crate) proved: Tensor<R>,
    /// Zero-based `I64` invocation indices, not section owners or diagnostic row IDs.
    pub(crate) owner: Tensor<R>,
    /// Certificates in `owner` order. `None` when no row proves coverage.
    pub(crate) pair: Option<CoveredDomain<R>>,
}

/// Two supported charts and retained inequalities proving their complete physical-domain coverage.
pub(crate) struct CoveredDomain<R: Runtime<DType = DType>> {
    pub(crate) lower: SupportedAreaQuery<R>,
    pub(crate) cap: SupportedAreaQuery<R>,
    /// Capacity minus the lower chart's maximum area, proved nonnegative.
    #[cfg(test)]
    pub(crate) cap_minus_lower: CertifiedExpansion<R>,
    /// Capacity minus the cap chart's maximum deficit, proved nonnegative.
    #[cfg(test)]
    pub(crate) cap_minus_deficit: CertifiedExpansion<R>,
    /// Cap chart's minimum area minus lower chart's maximum area, proved nonpositive.
    #[cfg(test)]
    pub(crate) overlap_gap: CertifiedExpansion<R>,
}

pub(super) struct ProofCandidates<R: Runtime<DType = DType>> {
    /// Zero-based indices of candidates with positive spans in the supplied capacity batch.
    pub(super) owner: Tensor<R>,
    pub(super) span: CertifiedExpansion<R>,
    /// Mask of proved inequalities in `owner` order, before chart support checks.
    pub(super) proved: Tensor<R>,
    #[cfg(test)]
    pub(super) cap_minus_lower: CertifiedExpansion<R>,
    #[cfg(test)]
    pub(super) cap_minus_deficit: CertifiedExpansion<R>,
    #[cfg(test)]
    pub(super) overlap_gap: CertifiedExpansion<R>,
}
