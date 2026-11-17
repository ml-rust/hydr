//! Capability declaration of sampled controls with ideal instantaneous pump response.

use super::{CONTROLLED_PUMP_CONTRACT_VERSION, CONTROLLED_PUMP_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, OptionSet, Precision,
    RuntimeKind, RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines the response model, operation, restoration and limits.
const CONTROLLED_PUMP_DOC: &str = "docs/controlled-pumps.md";

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

static CONTROLLED_PUMP: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::ControlledPump,
    contract: Some(CONTROLLED_PUMP_CONTRACT_VERSION),
    snapshot_schema: Some(CONTROLLED_PUMP_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "ideal_response_applies_selected_stopped_and_running_operations_on_the_decision_right",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_controlled_pump_operations_and_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_controlled_pump_f64_without_fallback",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[],
    options: &[OptionSet {
        id: "pump-response-model",
        values: &["ideal-instantaneous"],
        default: None,
    }],
    nested: &[ComponentId::OperatingControl],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "incomplete-response-definitions",
            UnsupportedReason::RejectedByAdmission,
            "Response definitions that do not match every controlled pump once, in order, reject.",
            CONTROLLED_PUMP_DOC,
        ),
        unsupported(
            "missing-response-evidence",
            UnsupportedReason::RejectedByAdmission,
            "Empty response provenance or ideal-response applicability rejects.",
            CONTROLLED_PUMP_DOC,
        ),
        unsupported(
            "non-pump-controller",
            UnsupportedReason::RejectedByAdmission,
            "A controller bound to a non-pump structure rejects.",
            CONTROLLED_PUMP_DOC,
        ),
        unsupported(
            "motor-dynamics",
            UnsupportedReason::OutsideModel,
            "No acceleration, startup delay, coast-down or independent response memory.",
            CONTROLLED_PUMP_DOC,
        ),
        unsupported(
            "native-control-authority",
            UnsupportedReason::OutsideModel,
            "No native SWMM control authority or parity.",
            CONTROLLED_PUMP_DOC,
        ),
        unsupported(
            "accepted-transfer",
            UnsupportedReason::NotAttached,
            "Operation feeds instantaneous proposals through operating sources; no donor budget or accepted transfer.",
            CONTROLLED_PUMP_DOC,
        ),
        unsupported(
            "gpu-numerical-qualification",
            UnsupportedReason::Unqualified,
            "CUDA runs execute, but only the CPU reference qualifies numerical accuracy.",
            CONTROLLED_PUMP_DOC,
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
            id: "memoryless-response",
            summary: "The response holds no state, so it adopts advanced checked controls unchanged.",
            doc: CONTROLLED_PUMP_DOC,
        },
        Limit {
            id: "right-side-operation",
            summary: "Actual operation equals the accepted command on each decision's right side.",
            doc: CONTROLLED_PUMP_DOC,
        },
    ],
    summary: Summary {
        equations: "actual pump operation equal to the accepted sampled command",
        method: "latch-selected original stopped or running literals, nested control trials, no response state",
        assessment: "selected analytical qualification, operation and restart cases; field accuracy remains unestablished",
    },
};

/// Returns the controlled pump's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn controlled_pump_capabilities() -> &'static ComponentCapabilities {
    &CONTROLLED_PUMP
}
