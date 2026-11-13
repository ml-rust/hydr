use super::{
    source_sonic_types::{SonicSide, SourceSonicCertificate},
    wave_curve_types::CurveValue,
};
use crate::channel::section::{PrismaticAreaPointRecovery, SupportedAreaQuery};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum DryVacuumRegime {
    BothDry = 0,
    LeftWet = 1,
    RightWet = 2,
    Vacuum = 3,
    Touching = 4,
    NotApplicable = 5,
    Unresolved = 6,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum DryVacuumStatus {
    Complete = 0,
    Pending = 1,
    NotApplicable = 2,
    RegimeUnresolved = 3,
}

pub(super) struct DryVacuumEdges<R: Runtime<DType = DType>> {
    pub(super) head: CurveValue<R>,
    pub(super) front: CurveValue<R>,
    pub(super) head_exact_zero: Tensor<R>,
    pub(super) front_exact_zero: Tensor<R>,
}
pub(super) struct DryVacuumPossibilities<R: Runtime<DType = DType>> {
    pub(super) left_exterior: Tensor<R>,
    pub(super) left_fan: Tensor<R>,
    pub(super) dry: Tensor<R>,
    pub(super) right_fan: Tensor<R>,
    pub(super) right_exterior: Tensor<R>,
}
pub(super) struct DryVacuumExterior<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) area: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
    pub(super) velocity: CertifiedExpansion<R>,
    pub(super) recovery: PrismaticAreaPointRecovery<R>,
}
pub(super) struct DryVacuumDry<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) area: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
    pub(super) depth: Interval<R>,
}
pub(super) struct DryVacuumPendingFan<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) source_status: Tensor<R>,
}
/// Sonic payloads gain fan authority only inside this admitted candidate set.
pub(super) enum DryVacuumCandidate<R: Runtime<DType = DType>> {
    Exterior(Box<DryVacuumExterior<R>>),
    Dry(Box<DryVacuumDry<R>>),
    Fan(Box<SourceSonicCertificate<R>>),
    PendingFan(Box<DryVacuumPendingFan<R>>),
}
pub(super) struct DryVacuumCertificates<R: Runtime<DType = DType>> {
    pub(super) invocation_witness: super::root_invocation::RootInvocationWitness,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) regime: Tensor<R>,
    pub(super) status: Tensor<R>,
    pub(super) invariant: CurveValue<R>,
    pub(super) edges: [DryVacuumEdges<R>; 2],
    pub(super) possibilities: DryVacuumPossibilities<R>,
    pub(super) candidates: Vec<DryVacuumCandidate<R>>,
    pub(super) sonic_status: [Tensor<R>; 2],
    pub(super) callback_points: usize,
    pub(super) point_limit: usize,
}
