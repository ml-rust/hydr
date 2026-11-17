//! Capability declaration of the 2D surface coupled to channel reaches through bank weirs.

use super::{SURFACE_CHANNEL_CONTRACT_VERSION, SURFACE_CHANNEL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines links, exchange, limits and scope.
const SURFACE_CHANNELS_DOC: &str = "docs/surface-channels.md";

/// Boundary kinds each side of a link admits.
const BANK_WEIR: &[BoundaryKind] = &[BoundaryKind::SuppressedRectangularWeir];

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

const fn weir_regime(regime: Regime) -> RegimeSupport {
    RegimeSupport {
        scope: RegimeScope::Boundary(BoundaryKind::SuppressedRectangularWeir),
        regime,
    }
}

static SURFACE_CHANNELS: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::SurfaceChannels,
    contract: Some(SURFACE_CHANNEL_CONTRACT_VERSION),
    snapshot_schema: Some(SURFACE_CHANNEL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::RoundToNearestSqrt,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "selected_cpu_client_executes_surface_channel_overflow_reversal_and_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_surface_channel_overflow_reversal_and_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_surface_channel_f64_without_fallback",
        },
    ],
    endpoints: &[
        Endpoint {
            id: EndpointId::SurfaceCell,
            admitted: BANK_WEIR,
        },
        Endpoint {
            id: EndpointId::Lateral,
            admitted: BANK_WEIR,
        },
    ],
    regimes: &[
        weir_regime(Regime::FreeWeirFlow),
        weir_regime(Regime::SubmergedWeirFlow),
        weir_regime(Regime::BelowCrest),
    ],
    sources: &[
        SourceKind::Rain,
        SourceKind::ImperviousLand,
        SourceKind::ModifiedGreenAmpt,
        SourceKind::Evaporation,
        SourceKind::Withdrawal,
        SourceKind::LateralInflow,
        SourceKind::LateralWithdrawal,
    ],
    options: &[],
    nested: &[
        ComponentId::SurfaceSolver,
        ComponentId::Soil,
        ComponentId::ChannelReachRectangular,
        ComponentId::ChannelReachTriangular,
    ],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "unobstructed-connection",
            UnsupportedReason::RejectedByAdmission,
            "A crest below the surface bed or the channel bed rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "surveyed-reach-link",
            UnsupportedReason::RejectedByAdmission,
            "A link to a surveyed reach rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "duplicate-links",
            UnsupportedReason::RejectedByAdmission,
            "A repeated link identity or a repeated surface cell, reach and channel cell rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "inactive-surface-cell",
            UnsupportedReason::RejectedByAdmission,
            "A link to an inactive surface cell or a cell outside the grid rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "non-unit-tangent",
            UnsupportedReason::RejectedByAdmission,
            "A nonfinite reach tangent or one off unit length rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "datum-or-gravity-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "A reach with another vertical datum or other gravity bits rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "step-and-retry-limits",
            UnsupportedReason::RejectedByAdmission,
            "Common step or retry limits outside the child limits reject, and so does a common interval below the minimum step.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "reserved-binding-prefix",
            UnsupportedReason::RejectedByAdmission,
            "A caller lateral binding or input under the `surface-exchange:` prefix rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "non-symmetric-weir",
            UnsupportedReason::RejectedByAdmission,
            "A weir coefficient without symmetric reversal evidence rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "missing-link-evidence",
            UnsupportedReason::RejectedByAdmission,
            "Empty mapping or storage-exclusion evidence rejects.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "integrated-lateral-span",
            UnsupportedReason::RejectedByAdmission,
            "An integrated caller lateral volume off the exact common interval rejects or requests recomputation.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "reach-end-port",
            UnsupportedReason::NotAttached,
            "Reach ends exchange no water with the surface; links attach at bank cells only.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "pool-attachment",
            UnsupportedReason::NotAttached,
            "Level pools attach through channel/pool or coupled level pools, not here.",
            "docs/channel-pools.md",
        ),
        unsupported(
            "structure-attachment",
            UnsupportedReason::NotAttached,
            "Hydraulic structures attach through coupled level pools, not here.",
            "docs/coupled-level-pools.md",
        ),
        unsupported(
            "tensor-forcing",
            UnsupportedReason::NotAttached,
            "Channel forcing is host data only; no device-resident forcing interval exists.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "subcycling",
            UnsupportedReason::OutsideModel,
            "Every child advances the same common interval; no child subcycles.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "normal-momentum",
            UnsupportedReason::OutsideModel,
            "Exchange water carries only the velocity along the reach tangent; normal momentum dissipates.",
            SURFACE_CHANNELS_DOC,
        ),
        unsupported(
            "gpu-numerical-qualification",
            UnsupportedReason::Unqualified,
            "CUDA runs execute, but no test qualifies their numerical accuracy.",
            "docs/integration.md",
        ),
        unsupported(
            "measured-event-accuracy",
            UnsupportedReason::Unqualified,
            "No measured-event assessment establishes field accuracy.",
            "docs/acceptance.md",
        ),
    ],
    limits: &[
        Limit {
            id: "momentum-closure",
            summary: "Surface water enters a reach at its post-transport tangent velocity; channel water leaves at its accepted-start velocity.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "caller-link-evidence",
            summary: "The caller attests that linked cells share a bank and that no water is stored twice.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "original-donor-water",
            summary: "Each donor gives only water it holds at the accepted start, under one shared factor.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "picard-convergence",
            summary: "Plain fixed-point iteration can stall on stiff drains; the coordinator then halves the common interval.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "child-endpoint-policies",
            summary: "Surface exterior faces and reach ends take the nested component's policies.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "first-order-time",
            summary: "Time integration remains first order, without an energy guarantee for the complete calculation.",
            doc: SURFACE_CHANNELS_DOC,
        },
    ],
    summary: Summary {
        equations: "2D shallow-water surface and 1D channel reaches exchanging water through suppressed rectangular bank weirs with Villemonte submergence",
        method: "fixed-point iteration of end stages and signed link volumes over one common interval, midpoint-head weir volumes, one shared factor per donor, and nested child admission",
        assessment: "selected spill, reversal, donor-limit, retry and restart cases; field accuracy remains unestablished",
    },
};

/// Returns the surface/channel coordinator's declared scope, independent of application
/// eligibility.
///
/// Execution success establishes no field accuracy.
pub fn surface_channel_capabilities() -> &'static ComponentCapabilities {
    &SURFACE_CHANNELS
}
