//! Capability declaration of the Modified Green-Ampt soil component.

use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines soil parameters, memory, inputs and limits.
const SOIL_DOC: &str = "docs/soil.md";

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

static SOIL: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::Soil,
    contract: None,
    snapshot_schema: None,
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "selected_cpu_client_executes_soil_trial_without_mutating_accepted_state",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_soil_trial_without_mutating_accepted_state",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_client_rejects_f64_profile",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[SourceKind::ModifiedGreenAmpt],
    options: &[],
    nested: &[],
    attachable_to: &[
        ComponentId::SurfaceSolver,
        ComponentId::Subcatchment,
        ComponentId::SurfaceChannels,
    ],
    unsupported: &[
        unsupported(
            "unrepresentable-derived-parameters",
            UnsupportedReason::RejectedByAdmission,
            "Parameters whose derived capacity or unit bridge rounds away reject.",
            SOIL_DOC,
        ),
        unsupported(
            "inactive-pervious-cell",
            UnsupportedReason::RejectedByAdmission,
            "Soil on an inactive hydraulic cell rejects; inactive cells take impervious land.",
            "docs/land-sources.md",
        ),
        unsupported(
            "mismatched-land-state",
            UnsupportedReason::RejectedByAdmission,
            "Antecedent memory whose variant differs from its land cell rejects.",
            "docs/land-sources.md",
        ),
        unsupported(
            "collapsed-positive-infiltration",
            UnsupportedReason::RejectedByAdmission,
            "Positive infiltration whose interval depth rounds to zero rejects.",
            SOIL_DOC,
        ),
        unsupported(
            "standalone-checkpoint",
            UnsupportedReason::OutsideModel,
            "No standalone checkpoint package; soil memory persists inside surface-solver and subcatchment packages.",
            SOIL_DOC,
        ),
        unsupported(
            "original-green-ampt",
            UnsupportedReason::OutsideModel,
            "Only the modified method; no original Green-Ampt model.",
            SOIL_DOC,
        ),
        unsupported(
            "surface-water-coupling",
            UnsupportedReason::NotAttached,
            "Surface-water availability, evaporation and water ledgers attach through land sources.",
            "docs/land-sources.md",
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
            id: "raw-empirical-infiltration",
            summary: "Outputs are raw interval infiltration, without surface-water availability or lifetime ledgers.",
            doc: SOIL_DOC,
        },
        Limit {
            id: "complete-native-memory",
            summary: "Reconstruction needs all five native state fields and identical checked parameters.",
            doc: SOIL_DOC,
        },
    ],
    summary: Summary {
        equations: "EPA SWMM 5.2.4 Modified Green-Ampt infiltration with upper-zone moisture recovery",
        method: "native-unit branches, Solvr batched Newton with the native F2min fallback, immutable trials",
        assessment: "pinned EPA reference corpus within frozen tolerances; field accuracy remains unestablished",
    },
};

/// Returns the soil component's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn soil_capabilities() -> &'static ComponentCapabilities {
    &SOIL
}
