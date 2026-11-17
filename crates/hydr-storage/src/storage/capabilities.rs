//! Capability declaration of the standalone level pool.

use super::{LEVEL_POOL_CONTRACT_VERSION, LEVEL_POOL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, Regime,
    RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus, RuntimeSupport, SourceKind, Summary,
    Unsupported, UnsupportedReason,
};

/// Path of the guide that defines pool curves, inputs, sources and limits.
const LEVEL_POOL_DOC: &str = "docs/level-pool.md";

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

static LEVEL_POOL: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::LevelPool,
    contract: Some(LEVEL_POOL_CONTRACT_VERSION),
    snapshot_schema: Some(LEVEL_POOL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "ordinary_decimal_exchange_retains_original_transfers_and_exact_twosum_witnesses",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_level_pool_exchange_and_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_level_pool_f64_without_fallback",
        },
    ],
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
    sources: &[
        SourceKind::ExchangeVolume,
        SourceKind::LocalPoolRain,
        SourceKind::LocalPoolLoss,
    ],
    options: &[],
    nested: &[],
    attachable_to: &[ComponentId::CoupledLevelPools, ComponentId::ChannelPools],
    unsupported: &[
        unsupported(
            "curve-extrapolation",
            UnsupportedReason::RejectedByAdmission,
            "Storage outside the closed curve volume range rejects without extrapolation.",
            LEVEL_POOL_DOC,
        ),
        unsupported(
            "absorbed-positive-exchange",
            UnsupportedReason::RejectedByAdmission,
            "A positive exchange that leaves rounded storage unchanged rejects.",
            LEVEL_POOL_DOC,
        ),
        unsupported(
            "donor-overdraft",
            UnsupportedReason::RejectedByAdmission,
            "Outgoing volume above the donor storage rejects without clipping.",
            LEVEL_POOL_DOC,
        ),
        unsupported(
            "outlet-structures",
            UnsupportedReason::NotAttached,
            "Outlet structures and controls attach through coupled level pools.",
            "docs/coupled-level-pools.md",
        ),
        unsupported(
            "geometry-events",
            UnsupportedReason::OutsideModel,
            "No geometry events or changing storage curves.",
            LEVEL_POOL_DOC,
        ),
        unsupported(
            "soil-groundwater",
            UnsupportedReason::OutsideModel,
            "Bottom demand is a supplied rate, without a soil or groundwater process.",
            LEVEL_POOL_DOC,
        ),
        unsupported(
            "within-interval-trajectory",
            UnsupportedReason::OutsideModel,
            "No continuous storage trajectory or knot-event splitting within an interval.",
            LEVEL_POOL_DOC,
        ),
        unsupported(
            "gpu-numerical-qualification",
            UnsupportedReason::Unqualified,
            "CUDA runs execute, but only the CPU reference qualifies numerical accuracy.",
            LEVEL_POOL_DOC,
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
            id: "storage-evidence-attested",
            summary: "The caller attests connected level water and separation from hydraulic cells.",
            doc: LEVEL_POOL_DOC,
        },
        Limit {
            id: "frozen-source-area",
            summary: "Rain and evaporation use the area frozen at the accepted start, first order in time.",
            doc: LEVEL_POOL_DOC,
        },
        Limit {
            id: "source-step-resolution",
            summary: "Each interval process volume stays within the maximum curve volume, or the interval rejects.",
            doc: LEVEL_POOL_DOC,
        },
    ],
    summary: Summary {
        equations: "volume balance of connected, approximately level storage with a monotone stage-volume curve",
        method: "ordered TwoSum exchange and frozen-area source updates, Solvr linear stage interpolation, independent physical and arithmetic gates",
        assessment: "selected analytical exchange, source-refinement and restart cases; field accuracy remains unestablished",
    },
};

/// Returns the level pool's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn level_pool_capabilities() -> &'static ComponentCapabilities {
    &LEVEL_POOL
}
