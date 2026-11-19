//! Capability declarations of the rectangular, triangular and surveyed channel reaches.

use super::{CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, OptionSet, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines reach geometry, endpoints, sources and limits.
const CHANNELS_DOC: &str = "docs/channels.md";

/// Summary of a component that exists on its own but has no attachment to this reach.
const STANDALONE: &str = "exists as a standalone component; not attached to this reach";

/// Endpoint policies of every reach geometry.
const SCALAR_REACH_ADMITTED: &[BoundaryKind] = &[
    BoundaryKind::Wall,
    BoundaryKind::Exterior,
    BoundaryKind::Stage,
    BoundaryKind::Discharge,
    BoundaryKind::QuiescentReservoir,
    BoundaryKind::FluxPort,
];

const fn regime(scope: RegimeScope, regime: Regime) -> RegimeSupport {
    RegimeSupport { scope, regime }
}

const fn runtimes(
    cpu: &'static str,
    cuda: &'static str,
    wgpu: &'static str,
) -> [RuntimeSupport; 3] {
    [
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: cpu,
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: cuda,
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: wgpu,
        },
    ]
}

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

const EXTERIOR_WET_SUBCRITICAL: RegimeSupport = regime(
    RegimeScope::Boundary(BoundaryKind::Exterior),
    Regime::WetSubcritical,
);
const EXTERIOR_SUPERCRITICAL: RegimeSupport = regime(
    RegimeScope::Boundary(BoundaryKind::Exterior),
    Regime::Supercritical,
);
const EXTERIOR_DRY: RegimeSupport =
    regime(RegimeScope::Boundary(BoundaryKind::Exterior), Regime::Dry);
const INTERIOR_WET_SUBCRITICAL: RegimeSupport =
    regime(RegimeScope::Interior, Regime::WetSubcritical);
const INTERIOR_SUPERCRITICAL: RegimeSupport = regime(RegimeScope::Interior, Regime::Supercritical);
const INTERIOR_DRY: RegimeSupport = regime(RegimeScope::Interior, Regime::Dry);
const INTERIOR_WET_DRY_FRONT: RegimeSupport = regime(RegimeScope::Interior, Regime::WetDryFront);
const FLUX_PORT_WET_SUBCRITICAL: RegimeSupport = regime(
    RegimeScope::Boundary(BoundaryKind::FluxPort),
    Regime::WetSubcritical,
);
const FLUX_PORT_SUPERCRITICAL: RegimeSupport = regime(
    RegimeScope::Boundary(BoundaryKind::FluxPort),
    Regime::Supercritical,
);
const FLUX_PORT_DRY: RegimeSupport =
    regime(RegimeScope::Boundary(BoundaryKind::FluxPort), Regime::Dry);
const FLUX_PORT_WET_DRY_FRONT: RegimeSupport = regime(
    RegimeScope::Boundary(BoundaryKind::FluxPort),
    Regime::WetDryFront,
);

/// Regimes of every reach geometry.
const SCALAR_REACH_REGIMES: &[RegimeSupport] = &[
    regime(
        RegimeScope::Boundary(BoundaryKind::Stage),
        Regime::WetSubcritical,
    ),
    regime(
        RegimeScope::Boundary(BoundaryKind::Discharge),
        Regime::WetSubcritical,
    ),
    regime(
        RegimeScope::Boundary(BoundaryKind::QuiescentReservoir),
        Regime::WetSubcritical,
    ),
    EXTERIOR_WET_SUBCRITICAL,
    EXTERIOR_SUPERCRITICAL,
    EXTERIOR_DRY,
    FLUX_PORT_WET_SUBCRITICAL,
    FLUX_PORT_SUPERCRITICAL,
    FLUX_PORT_DRY,
    FLUX_PORT_WET_DRY_FRONT,
    INTERIOR_WET_SUBCRITICAL,
    INTERIOR_SUPERCRITICAL,
    INTERIOR_DRY,
    INTERIOR_WET_DRY_FRONT,
];

const SOURCES: &[SourceKind] = &[SourceKind::LateralInflow, SourceKind::LateralWithdrawal];

/// Arithmetic of rectangular and surveyed reaches: IEEE basic, FMA and the Manning power.
const MANNING_ARITHMETIC: &[ArithmeticRequirement] = &[
    ArithmeticRequirement::F64Ieee754Basic,
    ArithmeticRequirement::NearestEvenGradualUnderflow,
    ArithmeticRequirement::FusedMulAdd,
    ArithmeticRequirement::CorrectlyRoundedElementary,
];

const STAGE_OUTSIDE_WET_SUBCRITICAL: Unsupported = unsupported(
    "stage-outside-wet-subcritical",
    UnsupportedReason::RejectedByAdmission,
    "Stage endpoints with a dry, critical or supercritical interior or trace reject.",
    CHANNELS_DOC,
);
const DISCHARGE_OUTSIDE_WET_SUBCRITICAL: Unsupported = unsupported(
    "discharge-outside-wet-subcritical",
    UnsupportedReason::RejectedByAdmission,
    "Discharge endpoints outside the connected wet-subcritical branch reject.",
    CHANNELS_DOC,
);
const RESERVOIR_OUTSIDE_WET_SUBCRITICAL: Unsupported = unsupported(
    "quiescent-reservoir-outside-wet-subcritical",
    UnsupportedReason::RejectedByAdmission,
    "QuiescentReservoir endpoints with a dry, critical or supercritical interior or trace reject.",
    CHANNELS_DOC,
);
const FLUX_PORT_SLOW_SIGNAL: Unsupported = unsupported(
    "flux-port-speed-below-interior",
    UnsupportedReason::RejectedByAdmission,
    "FluxPort endpoints whose signal speed lies below the end cell's |u| + c reject.",
    CHANNELS_DOC,
);
const FLUX_PORT_DRY_EXPORT: Unsupported = unsupported(
    "flux-port-dry-export",
    UnsupportedReason::RejectedByAdmission,
    "FluxPort endpoints that export water from a dry end cell reject.",
    CHANNELS_DOC,
);
const INTEGRATED_ENDPOINT_DISCHARGE: Unsupported = unsupported(
    "integrated-endpoint-discharge",
    UnsupportedReason::OutsideModel,
    "Discharge endpoints take a held rate; no integrated endpoint volume.",
    CHANNELS_DOC,
);
const TRANSMISSIVE_ENDPOINTS: Unsupported = unsupported(
    "transmissive-endpoints",
    UnsupportedReason::OutsideModel,
    "No transmissive endpoint policy.",
    CHANNELS_DOC,
);
const JUNCTIONS: Unsupported = unsupported(
    "junctions",
    UnsupportedReason::OutsideModel,
    "One unbranched reach, without general junctions.",
    CHANNELS_DOC,
);
const STRUCTURES: Unsupported = unsupported(
    "structures",
    UnsupportedReason::NotAttached,
    STANDALONE,
    "docs/structures.md",
);
const VARYING_GEOMETRY: Unsupported = unsupported(
    "varying-geometry",
    UnsupportedReason::OutsideModel,
    "No mixed sections, longitudinally varying profiles or changing geometry.",
    CHANNELS_DOC,
);
const LOCAL_LOSSES: Unsupported = unsupported(
    "contraction-expansion-losses",
    UnsupportedReason::OutsideModel,
    "No separate abrupt contraction or expansion losses.",
    CHANNELS_DOC,
);
const DRAINAGE: Unsupported = unsupported(
    "pressurized-drainage",
    UnsupportedReason::OutsideModel,
    "No drainage-network, surcharge or pressurized-pipe flow.",
    CHANNELS_DOC,
);
const TRANSVERSE_FLOW: Unsupported = unsupported(
    "transverse-flow",
    UnsupportedReason::OutsideModel,
    "One depth-averaged longitudinal velocity, without transverse jets or strong curvature.",
    CHANNELS_DOC,
);
const GPU_QUALIFICATION: Unsupported = unsupported(
    "gpu-numerical-qualification",
    UnsupportedReason::Unqualified,
    "CUDA runs execute, but no test qualifies their numerical accuracy.",
    "docs/integration.md",
);
const MEASURED_EVENT_ACCURACY: Unsupported = unsupported(
    "measured-event-accuracy",
    UnsupportedReason::Unqualified,
    "No measured-event assessment establishes field accuracy.",
    "docs/acceptance.md",
);

/// Unsupported uses of rectangular and triangular reaches.
const SCALAR_REACH_UNSUPPORTED: &[Unsupported] = &[
    STAGE_OUTSIDE_WET_SUBCRITICAL,
    DISCHARGE_OUTSIDE_WET_SUBCRITICAL,
    RESERVOIR_OUTSIDE_WET_SUBCRITICAL,
    FLUX_PORT_SLOW_SIGNAL,
    FLUX_PORT_DRY_EXPORT,
    INTEGRATED_ENDPOINT_DISCHARGE,
    TRANSMISSIVE_ENDPOINTS,
    JUNCTIONS,
    STRUCTURES,
    VARYING_GEOMETRY,
    LOCAL_LOSSES,
    DRAINAGE,
    TRANSVERSE_FLOW,
    GPU_QUALIFICATION,
    MEASURED_EVENT_ACCURACY,
];

/// Coordinators that nest rectangular and triangular reaches.
const SCALAR_REACH_PARENTS: &[ComponentId] =
    &[ComponentId::ChannelPools, ComponentId::SurfaceChannels];

const LATERAL_OWNERSHIP: Limit = Limit {
    id: "lateral-ownership-attested",
    summary: "The caller attests that lateral inputs duplicate no other water.",
    doc: CHANNELS_DOC,
};
const RESERVOIR_HELD_ELEVATION: Limit = Limit {
    id: "quiescent-reservoir-held-elevation",
    summary: "QuiescentReservoir holds the supplied elevation and advances no reservoir storage.",
    doc: CHANNELS_DOC,
};
const FLUX_PORT_PROVIDER: Limit = Limit {
    id: "flux-port-provider-rates",
    summary: "FluxPort rates and the exterior invariant come from the provider; the reach checks the speed bound against its end cell only.",
    doc: CHANNELS_DOC,
};

const ENDPOINTS: [Endpoint; 2] = [
    Endpoint {
        id: EndpointId::ReachLeft,
        admitted: SCALAR_REACH_ADMITTED,
    },
    Endpoint {
        id: EndpointId::ReachRight,
        admitted: SCALAR_REACH_ADMITTED,
    },
];

static RECTANGULAR: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::ChannelReachRectangular,
    contract: Some(CHANNEL_CONTRACT_VERSION),
    snapshot_schema: Some(CHANNEL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: MANNING_ARITHMETIC,
    runtimes: &runtimes(
        "selected_cpu_client_advances_rectangular_channels_and_complete_restart",
        "selected_cuda_client_advances_rectangular_channels_and_complete_restart",
        "selected_wgpu_client_rejects_channel_f64_without_fallback",
    ),
    endpoints: &ENDPOINTS,
    regimes: SCALAR_REACH_REGIMES,
    sources: SOURCES,
    options: &[],
    nested: &[],
    attachable_to: SCALAR_REACH_PARENTS,
    unsupported: SCALAR_REACH_UNSUPPORTED,
    limits: &[
        LATERAL_OWNERSHIP,
        RESERVOIR_HELD_ELEVATION,
        FLUX_PORT_PROVIDER,
        Limit {
            id: "scalar-endpoint-square-roots",
            summary: "Stage, Discharge and QuiescentReservoir endpoints also need correctly rounded F64 square roots.",
            doc: CHANNELS_DOC,
        },
    ],
    summary: Summary {
        equations: "1D hydrostatic shallow-water equations for one unbranched rectangular reach with Manning resistance",
        method: "first-order finite volumes, Roberts-Wilson pressure reconstruction, width-aware HLL with bracketed wave bounds, forward Euler and split Manning friction",
        assessment: "selected analytical, conservation, refinement, scalar-endpoint, flux-port and restart cases; field accuracy remains unestablished",
    },
};

static TRIANGULAR: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::ChannelReachTriangular,
    contract: Some(CHANNEL_CONTRACT_VERSION),
    snapshot_schema: Some(CHANNEL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::RoundToNearestSqrt,
        ArithmeticRequirement::FusedMulAdd,
        ArithmeticRequirement::CorrectlyRoundedElementary,
    ],
    runtimes: &runtimes(
        "selected_cpu_client_advances_triangular_channels_sources_and_complete_restart",
        "selected_cuda_client_advances_triangular_channels_sources_and_complete_restart",
        "selected_wgpu_client_rejects_triangular_f64_without_fallback",
    ),
    endpoints: &ENDPOINTS,
    regimes: SCALAR_REACH_REGIMES,
    sources: SOURCES,
    options: &[OptionSet {
        id: "triangular-reconstruction",
        values: &["first-order-reference", "limited-linear"],
        default: Some("limited-linear"),
    }],
    nested: &[],
    attachable_to: SCALAR_REACH_PARENTS,
    unsupported: SCALAR_REACH_UNSUPPORTED,
    limits: &[
        LATERAL_OWNERSHIP,
        RESERVOIR_HELD_ELEVATION,
        FLUX_PORT_PROVIDER,
    ],
    summary: Summary {
        equations: "1D hydrostatic shallow-water equations for one unbranched prismatic triangular reach with Manning resistance",
        method: "finite volumes with limited-linear or first-order reconstruction, barotropic Godunov sampling with certified flux brackets, forward Euler and split Manning friction",
        assessment: "selected analytical, conservation, refinement, scalar-endpoint, flux-port and restart cases; field accuracy remains unestablished",
    },
};

static SURVEYED: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::ChannelReachSurveyed,
    contract: Some(CHANNEL_CONTRACT_VERSION),
    snapshot_schema: Some(CHANNEL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: MANNING_ARITHMETIC,
    runtimes: &runtimes(
        "native_strided_sources_copy_operands_and_replay_complete_packages",
        "cuda_surveyed_sources_and_restart_use_actual_selected_device",
        "wgpu_rejects_surveyed_f64_without_fallback",
    ),
    endpoints: &ENDPOINTS,
    regimes: SCALAR_REACH_REGIMES,
    sources: SOURCES,
    options: &[],
    nested: &[],
    attachable_to: &[ComponentId::SurfaceChannels],
    unsupported: &[
        STAGE_OUTSIDE_WET_SUBCRITICAL,
        DISCHARGE_OUTSIDE_WET_SUBCRITICAL,
        RESERVOIR_OUTSIDE_WET_SUBCRITICAL,
        FLUX_PORT_SLOW_SIGNAL,
        FLUX_PORT_DRY_EXPORT,
        INTEGRATED_ENDPOINT_DISCHARGE,
        unsupported(
            "surveyed-channel-pool-attachment",
            UnsupportedReason::RejectedByAdmission,
            "Channel-to-pool connections reject surveyed reaches.",
            "docs/channel-pools.md",
        ),
        TRANSMISSIVE_ENDPOINTS,
        JUNCTIONS,
        STRUCTURES,
        VARYING_GEOMETRY,
        LOCAL_LOSSES,
        DRAINAGE,
        TRANSVERSE_FLOW,
        unsupported(
            "surveyed-convergence",
            UnsupportedReason::Unqualified,
            "No spatial, temporal or wet/dry-front convergence acceptance for surveyed reaches.",
            CHANNELS_DOC,
        ),
        GPU_QUALIFICATION,
        MEASURED_EVENT_ACCURACY,
    ],
    limits: &[
        LATERAL_OWNERSHIP,
        RESERVOIR_HELD_ELEVATION,
        FLUX_PORT_PROVIDER,
        Limit {
            id: "surveyed-work-ceilings",
            summary: "Surveyed root, integration and native query work stops at the configured ceilings.",
            doc: CHANNELS_DOC,
        },
        Limit {
            id: "surveyed-trace-certificates",
            summary: "Surveyed scalar traces and their signal speeds come from certified wave curves. An unresolved enclosure within the work ceilings rejects.",
            doc: CHANNELS_DOC,
        },
    ],
    summary: Summary {
        equations: "1D hydrostatic shallow-water equations for one unbranched horizontal surveyed reach with Manning resistance",
        method: "first-order finite volumes, certified surveyed-section interface fluxes, certified surveyed endpoint traces, forward Euler and split Manning friction with a bounded hydraulic radius",
        assessment: "functional CPU and CUDA source and restart cases with selected scalar-endpoint and flux-port cases; reach convergence and field accuracy remain unestablished",
    },
};

/// Returns the rectangular reach's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn rectangular_reach_capabilities() -> &'static ComponentCapabilities {
    &RECTANGULAR
}

/// Returns the triangular reach's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn triangular_reach_capabilities() -> &'static ComponentCapabilities {
    &TRIANGULAR
}

/// Returns the surveyed reach's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn surveyed_reach_capabilities() -> &'static ComponentCapabilities {
    &SURVEYED
}
