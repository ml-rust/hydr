use crate::channel::section::SupportedAreaQuery;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum WetStarStatus {
    Certified = 0,
    Unavailable = 1,
    Unresolved = 2,
}

/// Endpoint diagnostics preserve source anchors and every discarded arithmetic term.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "tests read the discharge change and unchanged mask, which are moves from the endpoint terms"
    )
)]
pub(super) struct WetStarSideEndpoint<R: Runtime<DType = DType>> {
    #[cfg(test)]
    pub(super) source_area: CertifiedExpansion<R>,
    pub(super) source_velocity: CertifiedExpansion<R>,
    pub(super) source_discharge: CertifiedExpansion<R>,
    pub(super) area_difference: CertifiedExpansion<R>,
    pub(super) velocity_change: CertifiedExpansion<R>,
    pub(super) velocity: CertifiedExpansion<R>,
    pub(super) discharge_change: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
    pub(super) unchanged: Tensor<R>,
    pub(super) available: Tensor<R>,
}

pub(super) struct WetStarEndpoint<R: Runtime<DType = DType>> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "tests read the area, a move from the endpoint terms"
        )
    )]
    pub(super) area: CertifiedExpansion<R>,
    pub(super) sides: [WetStarSideEndpoint<R>; 2],
    pub(super) waves: super::paired_endpoint::EndpointWaveEvidence<R>,
}

/// Range operands include the complete root domain, without discharge endpoint-extrema assumptions.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "tests read the operands, which are moves from the range computation"
    )
)]
pub(super) struct WetStarSideRange<R: Runtime<DType = DType>> {
    pub(super) area_difference: CertifiedExpansion<R>,
    pub(super) velocity_change: CertifiedExpansion<R>,
    pub(super) discharge_change: CertifiedExpansion<R>,
    pub(super) discharge: CertifiedExpansion<R>,
}

/// Physical root-state enclosures certify no wave position, selected state, or accepted flux.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "tests read the side operands and side intervals, which are moves from the range computation"
    )
)]
pub(super) struct WetStarDomain<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) endpoints: [WetStarEndpoint<R>; 2],
    pub(super) side_operands: [WetStarSideRange<R>; 2],
    pub(super) side_velocity: [Interval<R>; 2],
    pub(super) side_discharge: [Interval<R>; 2],
    #[cfg(test)]
    pub(super) velocity: Interval<R>,
    pub(super) discharge: Interval<R>,
    pub(super) status: Tensor<R>,
    /// Every row of one star domain shares this Simpson panel level.
    pub(super) level: super::source_binding::SourcePanelLevel,
}
