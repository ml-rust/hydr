//! Capability declaration of the 2D surface solver.

use super::SNAPSHOT_SCHEMA;
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

/// Version of the public numerical semantics and in-memory snapshot contract.
pub const CONTRACT_VERSION: &str = "hydr-swe-numr-cpu-1";

/// Summary of a component that exists on its own but has no attachment to this solver.
const STANDALONE: &str = "exists as a standalone component; not attached to this solver";

/// Path of the guide that defines the surface boundary policies.
const BOUNDARIES_DOC: &str = "docs/boundaries.md";

/// Path of the guide that defines the surface equations and excluded physics.
const NUMERICS_DOC: &str = "docs/numerics.md";

/// Path of the guide that defines the local land sources and their limits.
const LAND_SOURCES_DOC: &str = "docs/land-sources.md";

static CAPABILITIES: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::SurfaceSolver,
    contract: Some(CONTRACT_VERSION),
    snapshot_schema: Some(SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "selected_cpu_client_preserves_still_water_and_trusted_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_preserves_still_water_and_trusted_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_surface_solver_f64_without_fallback",
        },
    ],
    endpoints: &[Endpoint {
        id: EndpointId::ExteriorFace,
        admitted: &[
            BoundaryKind::Wall,
            BoundaryKind::FluxPort,
            BoundaryKind::Stage,
            BoundaryKind::Discharge,
            BoundaryKind::Exterior,
            BoundaryKind::Transmissive,
        ],
    }],
    regimes: &[
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::Stage),
            regime: Regime::WetSubcritical,
        },
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::Discharge),
            regime: Regime::WetSubcritical,
        },
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::Exterior),
            regime: Regime::WetSubcritical,
        },
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::Exterior),
            regime: Regime::Supercritical,
        },
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::Exterior),
            regime: Regime::Dry,
        },
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::Transmissive),
            regime: Regime::Dry,
        },
        RegimeSupport {
            scope: RegimeScope::Boundary(BoundaryKind::Transmissive),
            regime: Regime::Supercritical,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::WetSubcritical,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::Supercritical,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::Dry,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::WetDryFront,
        },
    ],
    sources: &[
        SourceKind::Rain,
        SourceKind::ImperviousLand,
        SourceKind::ModifiedGreenAmpt,
        SourceKind::Evaporation,
        SourceKind::Withdrawal,
    ],
    options: &[],
    nested: &[ComponentId::Soil],
    attachable_to: &[ComponentId::SurfaceChannels],
    unsupported: &[
        Unsupported {
            id: "stage-outside-wet-subcritical",
            reason: UnsupportedReason::RejectedByAdmission,
            summary: "Stage traces with a dry, critical or supercritical interior or trace reject.",
            doc: BOUNDARIES_DOC,
        },
        Unsupported {
            id: "discharge-outside-wet-subcritical",
            reason: UnsupportedReason::RejectedByAdmission,
            summary: "Discharge traces outside the connected wet-subcritical branch reject.",
            doc: BOUNDARIES_DOC,
        },
        Unsupported {
            id: "transmissive-subcritical-or-incoming",
            reason: UnsupportedReason::RejectedByAdmission,
            summary: "Transmissive faces with subcritical, critical or incoming wet flow reject.",
            doc: BOUNDARIES_DOC,
        },
        Unsupported {
            id: "channel-pools",
            reason: UnsupportedReason::NotAttached,
            summary: STANDALONE,
            doc: "docs/channel-pools.md",
        },
        Unsupported {
            id: "level-pools",
            reason: UnsupportedReason::NotAttached,
            summary: STANDALONE,
            doc: "docs/level-pool.md",
        },
        Unsupported {
            id: "coupled-level-pools",
            reason: UnsupportedReason::NotAttached,
            summary: STANDALONE,
            doc: "docs/coupled-level-pools.md",
        },
        Unsupported {
            id: "runoff-pools",
            reason: UnsupportedReason::NotAttached,
            summary: STANDALONE,
            doc: "docs/runoff-pools.md",
        },
        Unsupported {
            id: "reservoir",
            reason: UnsupportedReason::NotAttached,
            summary: STANDALONE,
            doc: "docs/reservoir.md",
        },
        Unsupported {
            id: "subcatchment",
            reason: UnsupportedReason::NotAttached,
            summary: STANDALONE,
            doc: "docs/subcatchment.md",
        },
        Unsupported {
            id: "structures",
            reason: UnsupportedReason::NotAttached,
            summary: STANDALONE,
            doc: "docs/structures.md",
        },
        Unsupported {
            id: "groundwater",
            reason: UnsupportedReason::OutsideModel,
            summary: "No groundwater or finite subsurface reservoir exchange.",
            doc: LAND_SOURCES_DOC,
        },
        Unsupported {
            id: "other-catchment-hydrology",
            reason: UnsupportedReason::OutsideModel,
            summary: "No catchment process beyond the declared rain, soil, evaporation and withdrawal.",
            doc: LAND_SOURCES_DOC,
        },
        Unsupported {
            id: "subgrid-terrain",
            reason: UnsupportedReason::OutsideModel,
            summary: "One bed elevation per cell, without subgrid storage or connections.",
            doc: NUMERICS_DOC,
        },
        Unsupported {
            id: "nonhydrostatic-flow",
            reason: UnsupportedReason::OutsideModel,
            summary: "No 3D or nonhydrostatic flow and no vertical jets.",
            doc: NUMERICS_DOC,
        },
        Unsupported {
            id: "sediment",
            reason: UnsupportedReason::OutsideModel,
            summary: "No sediment, erosion or debris transport.",
            doc: NUMERICS_DOC,
        },
        Unsupported {
            id: "additional-forces",
            reason: UnsupportedReason::OutsideModel,
            summary: "No wind, Coriolis, atmospheric pressure gradient or horizontal eddy viscosity.",
            doc: NUMERICS_DOC,
        },
        Unsupported {
            id: "gpu-numerical-qualification",
            reason: UnsupportedReason::Unqualified,
            summary: "GPU runs execute, but no test qualifies their numerical accuracy.",
            doc: "docs/integration.md",
        },
        Unsupported {
            id: "measured-event-accuracy",
            reason: UnsupportedReason::Unqualified,
            summary: "No measured-event assessment establishes field accuracy.",
            doc: "docs/acceptance.md",
        },
    ],
    limits: &[
        Limit {
            id: "flux-port-provider-attested",
            summary: "The provider attests flux-port source completeness and wave-speed bounds.",
            doc: BOUNDARIES_DOC,
        },
        Limit {
            id: "soil-water-adjustment-budget",
            summary: "Soil arithmetic reconciliation stays within a cumulative water-adjustment budget.",
            doc: LAND_SOURCES_DOC,
        },
    ],
    summary: Summary {
        equations: "full 2D hydrostatic depth-averaged shallow-water equations with Manning resistance",
        method: "first-order finite volumes, hydrostatic side corrections, bracketed-bound HLL, forward Euler and split sources",
        assessment: "selected synthetic analytical, conservation, refinement, restart, and flat wet/dry Clawpack comparisons; application accuracy remains unestablished",
    },
};

/// Returns the surface solver's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn surface_solver_capabilities() -> &'static ComponentCapabilities {
    &CAPABILITIES
}
