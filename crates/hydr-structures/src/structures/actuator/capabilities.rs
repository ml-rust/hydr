//! Capability declaration of ideal finite-rate gate actuators.

use super::{GATE_ACTUATOR_CONTRACT_VERSION, GATE_ACTUATOR_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines rates, targets, opening bounds and limits.
const ACTUATOR_DOC: &str = "docs/gate-actuators.md";

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

static GATE_ACTUATOR: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::GateActuator,
    contract: Some(GATE_ACTUATOR_CONTRACT_VERSION),
    snapshot_schema: Some(GATE_ACTUATOR_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::FusedMulAdd,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "selected_cpu_client_executes_gate_motion_reversal_and_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_gate_motion_reversal_and_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_gate_actuator_f64_without_fallback",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[],
    options: &[],
    nested: &[],
    attachable_to: &[ComponentId::ControlledGate],
    unsupported: &[
        unsupported(
            "non-gate-structure",
            UnsupportedReason::RejectedByAdmission,
            "An actuator bound to an unknown structure or a non-gate law rejects.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "duplicate-authority",
            UnsupportedReason::RejectedByAdmission,
            "Two actuators on one gate, or a repeated actuator identity, reject.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "missing-target-packet",
            UnsupportedReason::RejectedByAdmission,
            "A target event without one ordered packet per actuator, or a packet between events, rejects.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "stale-target",
            UnsupportedReason::RejectedByAdmission,
            "A packet issued before its target event rejects; no implicit hold applies.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "future-target",
            UnsupportedReason::RejectedByAdmission,
            "A packet available after its issue time rejects.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "target-outside-range",
            UnsupportedReason::RejectedByAdmission,
            "A target opening outside the bound gate range rejects.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "mechanical-dynamics",
            UnsupportedReason::OutsideModel,
            "No acceleration, inertia, backlash, deadband or load-dependent rates.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "continuous-feedback",
            UnsupportedReason::OutsideModel,
            "No continuous feedback from hydraulic state to opening.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "native-actuator-parity",
            UnsupportedReason::OutsideModel,
            "No native SWMM actuator behavior.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "hydraulic-consumption",
            UnsupportedReason::NotAttached,
            "Opening feeds structure proposals through operating sources; no accepted water transfer.",
            "docs/operating-sources.md",
        ),
        unsupported(
            "gpu-numerical-qualification",
            UnsupportedReason::Unqualified,
            "CUDA runs execute, but only the CPU reference qualifies numerical accuracy.",
            ACTUATOR_DOC,
        ),
        unsupported(
            "measured-event-accuracy",
            UnsupportedReason::Unqualified,
            "No measured-event assessment establishes mechanical calibration or field accuracy.",
            "docs/acceptance.md",
        ),
    ],
    limits: &[
        Limit {
            id: "ideal-directional-rates",
            summary: "Opening follows constant opening and closing rates within the gate range; a zero rate stalls that direction.",
            doc: ACTUATOR_DOC,
        },
        Limit {
            id: "certified-opening-bounds",
            summary: "Every position keeps finite certified bounds; rounded center equality never proves arrival or closure.",
            doc: ACTUATOR_DOC,
        },
        Limit {
            id: "explicit-accuracy-allowances",
            summary: "Opening and arrival-window allowances admit receipts; they never change rates or support.",
            doc: ACTUATOR_DOC,
        },
    ],
    summary: Summary {
        equations: "ideal rate-limited opening toward causal targets, capped at the target",
        method: "anchored fused multiply-add motion over TwoSum elapsed time, certified arrival windows, immutable trials",
        assessment: "pinned original-bit CPU reference histories; mechanical calibration and field accuracy remain unestablished",
    },
};

/// Returns the gate actuator's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn gate_actuator_capabilities() -> &'static ComponentCapabilities {
    &GATE_ACTUATOR
}
