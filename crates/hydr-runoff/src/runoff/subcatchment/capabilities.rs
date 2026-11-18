//! Capability declaration of the standalone three-subarea subcatchment.

use super::{SUBCATCHMENT_CONTRACT_VERSION, SUBCATCHMENT_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, OptionSet, Precision, RuntimeKind, RuntimeStatus, RuntimeSupport, SourceKind, Summary,
    Unsupported, UnsupportedReason,
};

/// Path of the guide that defines subareas, sources, routing and limits.
const SUBCATCHMENT_DOC: &str = "docs/subcatchment.md";

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

static SUBCATCHMENT: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::Subcatchment,
    contract: Some(SUBCATCHMENT_CONTRACT_VERSION),
    snapshot_schema: Some(SUBCATCHMENT_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::FusedMulAdd,
        ArithmeticRequirement::CorrectlyRoundedElementary,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "repeated_trials_leave_every_accepted_component_field_unchanged",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_subcatchment_routing_and_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_subcatchment_f64_without_fallback",
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
    ],
    options: &[OptionSet {
        id: "subcatchment-routing",
        values: &[
            "direct",
            "impervious-to-pervious",
            "pervious-to-impervious-storage",
        ],
        default: None,
    }],
    nested: &[ComponentId::Reservoir, ComponentId::Soil],
    attachable_to: &[ComponentId::RunoffPools],
    unsupported: &[
        unsupported(
            "soil-support-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "Soil parameters without positive pervious area, or pervious area without them, reject.",
            SUBCATCHMENT_DOC,
        ),
        unsupported(
            "routing-without-receiver",
            UnsupportedReason::RejectedByAdmission,
            "A positive routing fraction toward an absent receiving subarea rejects.",
            SUBCATCHMENT_DOC,
        ),
        unsupported(
            "runon-to-absent-subarea",
            UnsupportedReason::RejectedByAdmission,
            "Positive external runon onto an absent subarea rejects.",
            SUBCATCHMENT_DOC,
        ),
        unsupported(
            "catchment-graphs",
            UnsupportedReason::OutsideModel,
            "No inter-catchment routing graphs.",
            SUBCATCHMENT_DOC,
        ),
        unsupported(
            "tailwater-feedback",
            UnsupportedReason::OutsideModel,
            "No tailwater feedback on subarea exports.",
            SUBCATCHMENT_DOC,
        ),
        unsupported(
            "other-catchment-hydrology",
            UnsupportedReason::OutsideModel,
            "No groundwater, LID, snow or water quality.",
            SUBCATCHMENT_DOC,
        ),
        unsupported(
            "surface-solver",
            UnsupportedReason::NotAttached,
            "exists as a standalone component; not attached to this subcatchment",
            "docs/integration.md",
        ),
        unsupported(
            "gpu-numerical-qualification",
            UnsupportedReason::Unqualified,
            "CUDA runs execute, but only the CPU reference qualifies numerical accuracy.",
            SUBCATCHMENT_DOC,
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
            id: "caller-owned-runon",
            summary: "External runon arrives as authoritative destination volumes; the caller owns rainfall support and runon allocation.",
            doc: SUBCATCHMENT_DOC,
        },
        Limit {
            id: "soil-water-adjustment-budget",
            summary: "Soil arithmetic reconciliation stays within a cumulative water-adjustment budget.",
            doc: SUBCATCHMENT_DOC,
        },
        Limit {
            id: "first-order-source-priority",
            summary: "Source priority, integrated export and same-interval routing are first order and differ from native SWMM bookkeeping.",
            doc: SUBCATCHMENT_DOC,
        },
    ],
    summary: Summary {
        equations: "storage balance of three subareas with rain, Modified Green-Ampt infiltration, capped evaporation and nonlinear reservoir drainage",
        method: "fixed source priority, nested soil and reservoir admission, two topological routing phases with paired transfer volumes",
        assessment: "selected smooth-routing refinement against SciPy DOP853 and matching-input EPA soil calls; field accuracy remains unestablished",
    },
};

/// Returns the subcatchment's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn subcatchment_capabilities() -> &'static ComponentCapabilities {
    &SUBCATCHMENT
}
