use super::{point_candidate_types::PointCandidateCertificates, source_sonic_types::SonicSide};
use crate::channel::section::{PressureEndpoint, SupportedAreaQuery};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum PointFluxStatus {
    Accepted = 0,
    PendingCandidates = 1,
    PendingPositiveArea = 2,
    FluxBudgetExceeded = 3,
    /// Only the controller reports this: the sonic continuation reached its work ceiling.
    WorkLimit = 4,
    /// Only the controller reports this: the refined sonic root stays unresolved.
    RootBracketUnresolved = 5,
    /// Only the controller reports this: a possible wave branch lacks a certified speed.
    WaveEnvelopeUnresolved = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PointFluxKind {
    Exterior(SonicSide),
    Star,
    Fan(SonicSide),
    Dry,
}

#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct PointPressure<R: Runtime<DType = DType>> {
    pub(super) source_anchor: CertifiedExpansion<R>,
    pub(super) increment: PressureEndpoint<R>,
    pub(super) evaluation: Interval<R>,
    pub(super) selected_anchor: CertifiedExpansion<R>,
    pub(super) selected_increment: CertifiedExpansion<R>,
    pub(super) proposal_arithmetic: CertifiedExpansion<R>,
    pub(super) proposal: CertifiedExpansion<R>,
    pub(super) proposal_contained: Tensor<R>,
    pub(super) chosen: CertifiedExpansion<R>,
    pub(super) selection_change: CertifiedExpansion<R>,
    pub(super) evaluation_discrepancy: Interval<R>,
}

/// Every field belongs to one candidate and one selected conserved state.
#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct SelectedPointFlux<R: Runtime<DType = DType>> {
    pub(super) candidate: usize,
    pub(super) kind: PointFluxKind,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) area: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
    pub(super) velocity: CertifiedExpansion<R>,
    pub(super) momentum: CertifiedExpansion<R>,
    pub(super) depth: Interval<R>,
    pub(super) pressure: Option<PointPressure<R>>,
    pub(super) area_evaluation: CertifiedExpansion<R>,
    pub(super) area_discrepancy: Interval<R>,
    pub(super) area_projection: Interval<R>,
    pub(super) discharge_evaluation: CertifiedExpansion<R>,
    pub(super) discharge_invariant_velocity: CertifiedExpansion<R>,
    pub(super) discharge_product: Interval<R>,
    pub(super) discharge_physical: Interval<R>,
    pub(super) discharge_proposal: CertifiedExpansion<R>,
    pub(super) discharge_proposal_contained: Tensor<R>,
    pub(super) discharge_selection_change: CertifiedExpansion<R>,
    pub(super) discharge_discrepancy: Interval<R>,
    pub(super) velocity_arithmetic: CertifiedExpansion<R>,
    pub(super) momentum_arithmetic: CertifiedExpansion<R>,
    pub(super) momentum_evaluation: Interval<R>,
    pub(super) momentum_projection: Interval<R>,
}

#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct PointFluxCertificates<R: Runtime<DType = DType>> {
    pub(super) candidates: PointCandidateCertificates<R>,
    pub(super) status: Tensor<R>,
    /// Zero means the numeric hull is partial or absent and carries no complete physical certificate.
    pub(super) physical_valid: Tensor<R>,
    pub(super) joint_valid: Tensor<R>,
    pub(super) physical_water: Interval<R>,
    pub(super) physical_momentum: Interval<R>,
    pub(super) joint_water: Interval<R>,
    pub(super) joint_momentum: Interval<R>,
    pub(super) selected: Vec<SelectedPointFlux<R>>,
    pub(super) callback_points: usize,
    pub(super) point_limit: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum PointPositiveStatus {
    Supported = 0,
    PendingCandidates = 1,
    PendingPositiveArea = 2,
    NoProgress = 3,
    WorkLimit = 4,
    AccuracyFloor = 5,
    UnrepresentableTolerance = 6,
}

/// One explicit advance shares the original invocation and never retries internally.
#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct PointPositiveSupport<R: Runtime<DType = DType>> {
    pub(super) candidates: PointCandidateCertificates<R>,
    pub(super) status: Tensor<R>,
    pub(super) progressed: Tensor<R>,
    pub(super) attempts: Tensor<R>,
    pub(super) callback_points: usize,
    pub(super) additional_points: usize,
    pub(super) point_limit: usize,
}

/// Face-order fluxes after the sonic-fan continuation.
/// Each face keeps the values of its last selection, taken from one coherent trace.
#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct ResolvedPointFluxes<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    /// `PointFluxStatus` codes, including the controller-only codes.
    pub(super) status: Tensor<R>,
    pub(super) physical_valid: Tensor<R>,
    pub(super) joint_valid: Tensor<R>,
    pub(super) physical_water: Interval<R>,
    pub(super) physical_momentum: Interval<R>,
    pub(super) joint_water: Interval<R>,
    pub(super) joint_momentum: Interval<R>,
    /// I64 index into `selections` of the selection that owns each face.
    pub(super) selection: Tensor<R>,
    /// Every executed selection, in call order.
    pub(super) selections: Vec<PointFluxCertificates<R>>,
    /// Last sonic `PointQuadratureOutcome` per face. `NotApplicable` when never advanced.
    pub(super) sonic_outcome: Tensor<R>,
    /// Sonic panel level per face after its last advance. -1 when never advanced.
    pub(super) sonic_level: Tensor<R>,
    /// Signal speeds from the owning selection's certificates: lower is left, upper is right.
    /// Faces with `wave_valid` zero hold [0, 0], which certifies nothing.
    pub(super) wave_speed_m_per_s: Interval<R>,
    pub(super) wave_valid: Tensor<R>,
    pub(super) positive_support: Option<PointPositiveSupport<R>>,
    /// Controller rounds in which at least one face advanced.
    pub(super) rounds: usize,
    pub(super) callback_points: usize,
    pub(super) point_limit: usize,
}
