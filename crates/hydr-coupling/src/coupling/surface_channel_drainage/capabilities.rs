//! Capability declaration of the simultaneous 3-way surface-channel-drainage coordinator.

use super::{SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION, SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, Regime,
    RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus, RuntimeSupport, SourceKind, Summary,
    Unsupported, UnsupportedReason,
};

const DOC: &str = "docs/surface-channel-drainage.md";

const fn unsupported(
    id: &'static str,
    reason: UnsupportedReason,
    summary: &'static str,
    doc: &'static str,
) -> Unsupported {
    Unsupported {
        id,
        reason,
        summary,
        doc,
    }
}

static SURFACE_CHANNEL_DRAINAGE: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::SurfaceChannelDrainage,
    contract: Some(SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION),
    snapshot_schema: Some(SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::FusedMulAdd,
        ArithmeticRequirement::RoundToNearestSqrt,
        ArithmeticRequirement::CorrectlyRoundedElementary,
    ],
    runtimes: &[RuntimeSupport {
        runtime: RuntimeKind::Cpu,
        status: RuntimeStatus::Reference,
        evidence: "surface_channel_drainage_three_way_exchange_rollback_and_restart",
    }],
    endpoints: &[],
    regimes: &[
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::WetSubcritical,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::Dry,
        },
    ],
    sources: &[SourceKind::ExchangeVolume],
    options: &[],
    nested: &[ComponentId::SurfaceChannels, ComponentId::SwmmDrainage],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "duplicate-parameter-authority",
            UnsupportedReason::RejectedByAdmission,
            "A SWMM link owned by native [CONTROLS] or claimed twice, or a repeated inlet id or (cell, node) pair, rejects.",
            DOC,
        ),
        unsupported(
            "datum-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "A vertical datum mismatch between setup and SWMM drainage, or an inlet rim below node invert or surface bed, rejects.",
            DOC,
        ),
        unsupported(
            "gpu-execution",
            UnsupportedReason::OutsideModel,
            "The nested pinned EPA SWMM v5.2.4 worker executes on host CPU only.",
            DOC,
        ),
    ],
    limits: &[
        Limit {
            id: "three-way-midpoint-closure",
            summary: "Couples 2D surface + 1D channels/pools/structures with EPA SWMM v5.2.4 drainage over a common step via outer midpoint-head iteration and inner surface-channel fixed-point closure.",
            doc: DOC,
        },
        Limit {
            id: "pinned-swmm-v5-2-4",
            summary: "Requires the pinned EPA SWMM v5.2.4 worker built via tools/reference/swmm_drainage_build.py.",
            doc: DOC,
        },
    ],
    summary: Summary {
        equations: "Simultaneous 2D shallow-water surface flow, 1D river channels (with pools, structures, and runoff), and 1D dynamic-wave EPA SWMM v5.2.4 piped drainage.",
        method: "Outer midpoint-head inlet iteration over inner SurfaceChannelBatch fixed-point closure with exact step alignment, donor limiting, SWMM rollback, and atomic 3-way checkpoints.",
        assessment: "Reference CPU 3-way coordinator verified for simultaneous river overbank spill, sewer capture/surcharge, transactional rollback, and restart.",
    },
};

/// Returns the capability declaration of the 3-way surface-channel-drainage coordinator.
pub fn surface_channel_drainage_capabilities() -> &'static ComponentCapabilities {
    &SURFACE_CHANNEL_DRAINAGE
}
