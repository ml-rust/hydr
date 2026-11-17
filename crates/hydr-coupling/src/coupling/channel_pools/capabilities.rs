//! Capability declaration of channel reaches connected to finite level pools.

use super::{CHANNEL_POOL_CONTRACT_VERSION, CHANNEL_POOL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines connections, forcing, limits and scope.
const CHANNEL_POOLS_DOC: &str = "docs/channel-pools.md";

/// Boundary kinds a connected reach endpoint admits.
const CONNECTED: &[BoundaryKind] = &[BoundaryKind::PoolConnection];

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

static CHANNEL_POOLS: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::ChannelPools,
    contract: Some(CHANNEL_POOL_CONTRACT_VERSION),
    snapshot_schema: Some(CHANNEL_POOL_SNAPSHOT_SCHEMA),
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
            evidence: "native_finite_pool_forcing_preserves_views_aliases_signed_zero_and_destination_restarts",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_executes_finite_pool_non_knot_cuts_mixed_connections_and_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_finite_channel_pool_f64_without_fallback",
        },
    ],
    endpoints: &[
        Endpoint {
            id: EndpointId::ReachLeft,
            admitted: CONNECTED,
        },
        Endpoint {
            id: EndpointId::ReachRight,
            admitted: CONNECTED,
        },
    ],
    regimes: &[
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::PoolConnection),
            regime: Regime::WetSubcritical,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::WetSubcritical,
        },
    ],
    sources: &[SourceKind::LateralInflow, SourceKind::LateralWithdrawal],
    options: &[],
    nested: &[
        ComponentId::ChannelReachRectangular,
        ComponentId::ChannelReachTriangular,
        ComponentId::LevelPool,
    ],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "surveyed-reach-attachment",
            UnsupportedReason::RejectedByAdmission,
            "Surveyed reaches reject; advance them as standalone reaches.",
            CHANNEL_POOLS_DOC,
        ),
        unsupported(
            "dry-connections",
            UnsupportedReason::RejectedByAdmission,
            "A connection with a dry pool or a dry connected cell rejects.",
            CHANNEL_POOLS_DOC,
        ),
        unsupported(
            "supercritical-connections",
            UnsupportedReason::RejectedByAdmission,
            "Supercritical flow through a connection rejects in either direction.",
            CHANNEL_POOLS_DOC,
        ),
        unsupported(
            "critical-transitions",
            UnsupportedReason::OutsideModel,
            "No critical transitions at connections.",
            CHANNEL_POOLS_DOC,
        ),
        unsupported(
            "jets-and-oscillating-surfaces",
            UnsupportedReason::OutsideModel,
            "No jets or oscillating pool surfaces.",
            CHANNEL_POOLS_DOC,
        ),
        unsupported(
            "pool-structures",
            UnsupportedReason::NotAttached,
            "Pool structures attach through coupled level pools, not here.",
            "docs/coupled-level-pools.md",
        ),
        unsupported(
            "local-pool-sources",
            UnsupportedReason::NotAttached,
            "Local pool rain and losses attach through coupled level pools, not here.",
            "docs/coupled-level-pools.md",
        ),
        unsupported(
            "junctions",
            UnsupportedReason::OutsideModel,
            "No general channel junctions.",
            CHANNEL_POOLS_DOC,
        ),
        unsupported(
            "inferred-connections",
            UnsupportedReason::OutsideModel,
            "No inferred geographic connections or overlapping represented storage.",
            CHANNEL_POOLS_DOC,
        ),
        unsupported(
            "surface-solver",
            UnsupportedReason::NotAttached,
            "exists as a standalone component; not attached to this coordinator",
            "docs/integration.md",
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
            id: "unconnected-endpoint-policies",
            summary: "Unconnected endpoints take the nested reach's endpoint policies.",
            doc: CHANNEL_POOLS_DOC,
        },
        Limit {
            id: "quiescent-pool-head",
            summary: "Pools stay approximately level with negligible approach velocity, and exits mix completely.",
            doc: CHANNEL_POOLS_DOC,
        },
        Limit {
            id: "original-donor-water",
            summary: "Each connection exports only eligible water present at the accepted start.",
            doc: CHANNEL_POOLS_DOC,
        },
        Limit {
            id: "first-order-time",
            summary: "Time integration remains first order, without an energy guarantee for the complete calculation.",
            doc: CHANNEL_POOLS_DOC,
        },
    ],
    summary: Summary {
        equations: "1D shallow-water reaches exchanging water with level pools through quiescent-reservoir connections with entrance loss",
        method: "fixed-point iteration of midpoint pool heads over common intervals, paired signed transfers, and nested reach and pool admission",
        assessment: "selected feedback, refinement, reversal and restart cases; field accuracy remains unestablished",
    },
};

/// Returns the channel/pool coordinator's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn channel_pool_capabilities() -> &'static ComponentCapabilities {
    &CHANNEL_POOLS
}
