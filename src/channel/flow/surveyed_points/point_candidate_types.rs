use super::{
    dry_vacuum_types::{DryVacuumEdges, DryVacuumPossibilities},
    root_invocation::RootInvocationWitness,
    source_sonic_types::{SonicSide, SourceSonicCertificate},
    wave_curve_types::CurveValue,
    wet_wave_types::WetWaveDomain,
};
use crate::channel::section::{PrismaticAreaPointRecovery, SupportedAreaQuery};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct ExteriorPoint<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) area: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
    pub(super) velocity: CertifiedExpansion<R>,
    pub(super) recovery: PrismaticAreaPointRecovery<R>,
}
pub(super) struct DryPoint<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) area: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
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
pub(super) enum PointFanAuthority {
    DryVacuum,
    Wet(usize),
}
pub(super) struct PointFan<R: Runtime<DType = DType>> {
    pub(super) certificate: SourceSonicCertificate<R>,
    pub(super) active: Tensor<R>,
    pub(super) authority: PointFanAuthority,
}
pub(super) struct PointFanEvidence<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) membership: super::wet_fan_membership::WetFanMembership<R>,
    pub(super) physical_status: Tensor<R>,
}
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
    Pending(PointPending<R>),
}
pub(super) struct PointCandidateCertificates<R: Runtime<DType = DType>> {
    pub(super) invocation_witness: RootInvocationWitness,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) regime: Tensor<R>,
    pub(super) status: Tensor<R>,
    pub(super) wet_status: Tensor<R>,
    pub(super) depth_status: Tensor<R>,
    pub(super) wet_evidence: Vec<super::wet_root_types::WetRootEvidence<R>>,
    pub(super) sonic_status: [Tensor<R>; 2],
    pub(super) dry_vacuum: PointDryEvidence<R>,
    pub(super) fan_evidence: Vec<PointFanEvidence<R>>,
    pub(super) waves: Vec<WetWaveDomain<R>>,
    pub(super) candidates: Vec<PointCandidate<R>>,
    pub(super) callback_points: usize,
    pub(super) point_limit: usize,
}
