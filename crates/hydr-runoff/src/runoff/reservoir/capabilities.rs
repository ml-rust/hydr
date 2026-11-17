//! Capability declaration of the standalone nonlinear reservoir.

use super::{RESERVOIR_CONTRACT_VERSION, RESERVOIR_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, RuntimeKind, RuntimeStatus, RuntimeSupport, Summary, Unsupported,
    UnsupportedReason,
};

/// Path of the guide that defines drainage branches, inputs, outputs and limits.
const RESERVOIR_DOC: &str = "docs/reservoir.md";

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

static RESERVOIR: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::Reservoir,
    contract: Some(RESERVOIR_CONTRACT_VERSION),
    snapshot_schema: Some(RESERVOIR_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[RuntimeSupport {
        runtime: RuntimeKind::Cpu,
        status: RuntimeStatus::Reference,
        evidence: "immutable_repeated_trials_preserve_the_accepted_batch",
    }],
    endpoints: &[Endpoint {
        id: EndpointId::Outlet,
        admitted: &[BoundaryKind::RunoffOutletNoFeedback],
    }],
    regimes: &[],
    sources: &[],
    options: &[],
    nested: &[],
    attachable_to: &[ComponentId::Subcatchment],
    unsupported: &[
        unsupported(
            "ambiguous-closure",
            UnsupportedReason::RejectedByAdmission,
            "Zero roughness with zero width or slope rejects as an ambiguous closure.",
            RESERVOIR_DOC,
        ),
        unsupported(
            "absorbed-drainage-excess",
            UnsupportedReason::RejectedByAdmission,
            "A positive drainage excess that rounds back to depression storage rejects.",
            RESERVOIR_DOC,
        ),
        unsupported(
            "soil-and-evaporation",
            UnsupportedReason::NotAttached,
            "Soil and evaporation attach through the subcatchment, not here.",
            "docs/subcatchment.md",
        ),
        unsupported(
            "internal-routing",
            UnsupportedReason::NotAttached,
            "Paired internal routing attaches through the subcatchment, not here.",
            "docs/subcatchment.md",
        ),
        unsupported(
            "surface-solver",
            UnsupportedReason::NotAttached,
            "exists as a standalone component; not attached to this reservoir",
            "docs/integration.md",
        ),
        unsupported(
            "native-adaptive-stepping",
            UnsupportedReason::OutsideModel,
            "No SWMM adaptive ODE controller or endpoint-discharge volume bookkeeping.",
            RESERVOIR_DOC,
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
            id: "parameter-selected-drainage",
            summary: "Roughness, width and slope select nonlinear drainage, closed storage or instantaneous overflow.",
            doc: RESERVOIR_DOC,
        },
        Limit {
            id: "constant-interval-supply",
            summary: "Each element receives one constant nonnegative incoming rate over the exact interval.",
            doc: RESERVOIR_DOC,
        },
        Limit {
            id: "first-order-implicit-storage",
            summary: "The storage equation is first-order implicit; refinement evidence covers selected recession cases.",
            doc: RESERVOIR_DOC,
        },
    ],
    summary: Summary {
        equations: "storage balance with the native-derived nonlinear depth-discharge law above depression storage",
        method: "first-order implicit storage update, Solvr batched bisection of the excess-depth root, independent volume gates",
        assessment: "selected analytical recession refinement and branch cases; field accuracy remains unestablished",
    },
};

/// Returns the standalone reservoir's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn reservoir_capabilities() -> &'static ComponentCapabilities {
    &RESERVOIR
}
