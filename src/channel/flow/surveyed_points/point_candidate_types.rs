use super::{
    dry_vacuum_types::{DryVacuumEdges, DryVacuumPossibilities},
    root_invocation::RootInvocationWitness,
    source_sonic_types::{SonicSide, SourceSonicCertificate},
    wave_curve_types::CurveValue,
    wet_wave_types::WetWaveDomain,
};
#[cfg(test)]
use crate::channel::section::PrismaticAreaPointRecovery;
use crate::channel::section::SupportedAreaQuery;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::CertifiedExpansion;
#[cfg(test)]
use solvr::common::arithmetic::Interval;

pub(super) struct ExteriorPoint<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) area: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
    #[cfg(test)]
    pub(super) velocity: CertifiedExpansion<R>,
    #[cfg(test)]
    pub(super) recovery: PrismaticAreaPointRecovery<R>,
}
pub(super) struct DryPoint<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) area: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
    #[cfg(test)]
    pub(super) depth: Interval<R>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum PointCandidateStatus {
    Complete = 0,
    Pending = 1,
    RegimeUnresolved = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum PointRegime {
    BothDry = 0,
    LeftWet = 1,
    RightWet = 2,
    Vacuum = 3,
    Touching = 4,
    Wet = 5,
    #[cfg(test)]
    Unresolved = 6,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PointRegion {
    LeftFan,
    Star,
    RightFan,
    Unlocated,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PointPendingReason {
    WetRoot,
    WavePositions,
    StarState,
    FanCertificate,
}
#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct PointPending<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) region: PointRegion,
    pub(super) reason: PointPendingReason,
}
/// Rows index the complete physical star operands in the owning result's wet-wave evidence.
pub(super) struct PointStar<R: Runtime<DType = DType>> {
    pub(super) wave: usize,
    pub(super) rows: Tensor<R>,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
}
#[cfg(test)]
pub(super) enum PointFanAuthority {
    DryVacuum,
    Wet(usize),
}
pub(super) struct PointFan<R: Runtime<DType = DType>> {
    pub(super) certificate: SourceSonicCertificate<R>,
    pub(super) active: Tensor<R>,
    #[cfg(test)]
    pub(super) authority: PointFanAuthority,
}
#[cfg(test)]
pub(super) struct PointFanEvidence<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) membership: super::wet_fan_membership::WetFanMembership<R>,
    pub(super) physical_status: Tensor<R>,
}
#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct PointDryEvidence<R: Runtime<DType = DType>> {
    pub(super) regime: Tensor<R>,
    pub(super) status: Tensor<R>,
    pub(super) invariant: CurveValue<R>,
    pub(super) edges: [DryVacuumEdges<R>; 2],
    pub(super) possibilities: DryVacuumPossibilities<R>,
}
pub(super) enum PointCandidate<R: Runtime<DType = DType>> {
    Exterior(Box<ExteriorPoint<R>>),
    Star(PointStar<R>),
    Fan(Box<PointFan<R>>),
    Dry(Box<DryPoint<R>>),
    Pending(#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))] PointPending<R>),
}
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "tests read the parent rows and the callback and point counts, which are moves and counters production already holds"
    )
)]
pub(super) struct PointCandidateCertificates<R: Runtime<DType = DType>> {
    pub(super) invocation_witness: RootInvocationWitness,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) regime: Tensor<R>,
    pub(super) status: Tensor<R>,
    pub(super) wet_status: Tensor<R>,
    #[cfg(test)]
    pub(super) depth_status: Tensor<R>,
    #[cfg(test)]
    pub(super) wet_evidence: Vec<super::wet_root_types::WetRootEvidence<R>>,
    #[cfg(test)]
    pub(super) sonic_status: [Tensor<R>; 2],
    pub(super) dry_vacuum: PointDryEvidence<R>,
    #[cfg(test)]
    pub(super) fan_evidence: Vec<PointFanEvidence<R>>,
    pub(super) waves: Vec<WetWaveDomain<R>>,
    pub(super) candidates: Vec<PointCandidate<R>>,
    pub(super) callback_points: usize,
    pub(super) point_limit: usize,
}
