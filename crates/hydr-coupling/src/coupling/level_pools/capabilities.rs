//! Capability declaration of level pools joined through structures.

use super::{COUPLED_LEVEL_POOL_CONTRACT_VERSION, COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines connections, operations, sources, inflows and limits.
const COUPLED_DOC: &str = "docs/coupled-level-pools.md";

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

static COUPLED_LEVEL_POOLS: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::CoupledLevelPools,
    contract: Some(COUPLED_LEVEL_POOL_CONTRACT_VERSION),
    snapshot_schema: Some(COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "selected_cpu_client_executes_coupled_pool_exchange_and_trusted_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_coupled_pool_exchange_and_trusted_restart",
        },
    ],
    endpoints: &[
        Endpoint {
            id: EndpointId::PoolConnection,
            admitted: &[
                BoundaryKind::SubmergedOrifice,
                BoundaryKind::SuppressedRectangularWeir,
                BoundaryKind::PumpStoppedIsolated,
                BoundaryKind::PumpRunning,
                BoundaryKind::GateFixedOpening,
            ],
        },
        Endpoint {
            id: EndpointId::Lateral,
            admitted: &[
                BoundaryKind::PrescribedInflowRate,
                BoundaryKind::PrescribedInflowVolume,
            ],
        },
    ],
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
    sources: &[SourceKind::LocalPoolRain, SourceKind::LocalPoolLoss],
    options: &[],
    nested: &[ComponentId::LevelPool],
    attachable_to: &[ComponentId::RunoffPools],
    unsupported: &[
        unsupported(
            "implicit-pump-operation",
            UnsupportedReason::RejectedByAdmission,
            "A pump without an explicit running or stopped-isolated operation rejects.",
            COUPLED_DOC,
        ),
        unsupported(
            "mismatched-structure-operation",
            UnsupportedReason::RejectedByAdmission,
            "A gate without an opening, or a passive law without a Passive operation, rejects.",
            COUPLED_DOC,
        ),
        unsupported(
            "external-withdrawals",
            UnsupportedReason::OutsideModel,
            "No external withdrawals or runoff feedback from receiving-pool levels.",
            COUPLED_DOC,
        ),
        unsupported(
            "measured-replacement-selection",
            UnsupportedReason::OutsideModel,
            "Inflows arrive already selected, without measured-replacement selection.",
            COUPLED_DOC,
        ),
        unsupported(
            "dynamic-operations",
            UnsupportedReason::NotAttached,
            "Operations stay static; control schedules, trajectories and actuator frames are separate components.",
            "docs/operating-controls.md",
        ),
        unsupported(
            "surface-solver",
            UnsupportedReason::NotAttached,
            "exists as a standalone component; not attached to these pools",
            "docs/integration.md",
        ),
        unsupported(
            "closure-event-localization",
            UnsupportedReason::OutsideModel,
            "Availability scaling locates no separate connection closure events.",
            COUPLED_DOC,
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
            id: "inflow-ownership-attested",
            summary: "The caller attests external origin and no duplicated water for each inflow.",
            doc: COUPLED_DOC,
        },
        Limit {
            id: "structure-hydraulic-support",
            summary: "Running pumps need wet support, and submerged orifices need water above the crown at both endpoints.",
            doc: COUPLED_DOC,
        },
        Limit {
            id: "post-source-donor-budget",
            summary: "Withdrawals draw on post-source storage; incoming transfers and inflows fund no earlier removal.",
            doc: COUPLED_DOC,
        },
    ],
    summary: Summary {
        equations: "volume balance of connected level pools exchanging water through checked structure laws",
        method: "midpoint-head Picard iteration with bounded interval halving, one signed volume per structure, and nested level-pool accounting",
        assessment: "selected analytical exchange, refinement, source, inflow and restart cases; field accuracy remains unestablished",
    },
};

/// Returns the coupled level pools' declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn coupled_level_pool_capabilities() -> &'static ComponentCapabilities {
    &COUPLED_LEVEL_POOLS
}
