//! Capability declaration of free runoff outlets advanced with their receiving pools.

use super::{RUNOFF_POOL_CONTRACT_VERSION, RUNOFF_POOL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, RuntimeKind, RuntimeStatus, RuntimeSupport, SourceKind, Summary, Unsupported,
    UnsupportedReason,
};

/// Path of the guide that defines outlet bindings, forcing, retries and limits.
const RUNOFF_POOLS_DOC: &str = "docs/runoff-pools.md";

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

static RUNOFF_POOLS: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::RunoffPools,
    contract: Some(RUNOFF_POOL_CONTRACT_VERSION),
    snapshot_schema: Some(RUNOFF_POOL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::RoundToNearestSqrt,
        ArithmeticRequirement::FusedMulAdd,
        ArithmeticRequirement::CorrectlyRoundedElementary,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "selected_cpu_client_pairs_native_runoff_and_preserves_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_pairs_native_runoff_and_preserves_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_runoff_pools_f64_without_fallback",
        },
    ],
    endpoints: &[Endpoint {
        id: EndpointId::Outlet,
        admitted: &[BoundaryKind::RunoffOutletNoFeedback],
    }],
    regimes: &[],
    sources: &[
        SourceKind::Rain,
        SourceKind::ImperviousLand,
        SourceKind::ModifiedGreenAmpt,
        SourceKind::Evaporation,
        SourceKind::LocalPoolRain,
        SourceKind::LocalPoolLoss,
    ],
    options: &[],
    nested: &[ComponentId::Subcatchment, ComponentId::CoupledLevelPools],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "unmapped-outlet",
            UnsupportedReason::RejectedByAdmission,
            "A potential outlet or receiving boundary left without its pair rejects.",
            RUNOFF_POOLS_DOC,
        ),
        unsupported(
            "duplicate-outlet",
            UnsupportedReason::RejectedByAdmission,
            "An outlet lane or receiving boundary bound twice rejects.",
            RUNOFF_POOLS_DOC,
        ),
        unsupported(
            "backwater-feedback",
            UnsupportedReason::OutsideModel,
            "No downstream backwater, tailwater feedback or reversed runoff through an outlet.",
            RUNOFF_POOLS_DOC,
        ),
        unsupported(
            "external-runon",
            UnsupportedReason::OutsideModel,
            "No external runon, additional prescribed pool inflows or disposal routes.",
            RUNOFF_POOLS_DOC,
        ),
        unsupported(
            "catchment-graphs",
            UnsupportedReason::OutsideModel,
            "No inter-catchment routing graphs or geometric nonoverlap proofs.",
            RUNOFF_POOLS_DOC,
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
            id: "free-outlet-attested",
            summary: "The caller attests free drainage and separate water ownership for each outlet.",
            doc: RUNOFF_POOLS_DOC,
        },
        Limit {
            id: "common-interval-recalculation",
            summary: "A shorter pool interval recalculates both children from accepted state, within the retry limit.",
            doc: RUNOFF_POOLS_DOC,
        },
        Limit {
            id: "independent-child-budgets",
            summary: "Child iteration, conservation and soil budgets stay independent of the common water budget.",
            doc: RUNOFF_POOLS_DOC,
        },
    ],
    summary: Summary {
        equations: "subcatchment runoff and connected level-pool volume balances sharing exact outlet volumes",
        method: "common trial intervals, original outlet tensors as pool interval volumes, full recalculation on retry, nested child admission",
        assessment: "selected exact export, recalculation, source-order and restart cases; field accuracy remains unestablished",
    },
};

/// Returns the runoff/pool coordinator's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn runoff_pool_capabilities() -> &'static ComponentCapabilities {
    &RUNOFF_POOLS
}
