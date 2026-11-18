//! Capability declaration of sampled controls owning ideal gate response.

use super::{CONTROLLED_GATE_CONTRACT_VERSION, CONTROLLED_GATE_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines binding, decisions, restoration and limits.
const CONTROLLED_GATE_DOC: &str = "docs/controlled-gates.md";

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

static CONTROLLED_GATE: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::ControlledGate,
    contract: Some(CONTROLLED_GATE_CONTRACT_VERSION),
    snapshot_schema: Some(CONTROLLED_GATE_SNAPSHOT_SCHEMA),
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
            evidence: "dyadic_switches_and_reversal_enclose_frozen_physical_positions_across_layouts",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_controlled_gate_motion_and_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_controlled_gate_f64_without_fallback",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[],
    options: &[],
    nested: &[ComponentId::OperatingControl, ComponentId::GateActuator],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "advanced-child",
            UnsupportedReason::RejectedByAdmission,
            "A control or actuator child with accepted intervals rejects; restore a composite snapshot instead.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "foreign-structure-model",
            UnsupportedReason::RejectedByAdmission,
            "Children built on differing structure definitions reject.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "mismatched-gate-subset",
            UnsupportedReason::RejectedByAdmission,
            "Controllers and actuators that do not bind the same ordered gates one to one reject.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "calendar-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "Initial, decision, target or terminal times that differ by F64 bits reject.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "initial-target-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "An initial command numerically different from the actuator's initial target rejects.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "mechanical-dynamics",
            UnsupportedReason::OutsideModel,
            "No mechanical dynamics or native actuator parity.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "hydraulic-feedback",
            UnsupportedReason::OutsideModel,
            "No continuous trigger completeness or endogenous hydraulic feedback.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "pump-response",
            UnsupportedReason::NotAttached,
            "Pump response attaches through controlled pumps, not here.",
            "docs/controlled-pumps.md",
        ),
        unsupported(
            "accepted-hydraulic-transfer",
            UnsupportedReason::NotAttached,
            "Opening feeds structure proposals through operating sources; no accepted transfer or discharge enclosure.",
            CONTROLLED_GATE_DOC,
        ),
        unsupported(
            "gpu-numerical-qualification",
            UnsupportedReason::Unqualified,
            "CUDA runs execute, but only the CPU reference qualifies numerical accuracy.",
            CONTROLLED_GATE_DOC,
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
            id: "atomic-acceptance",
            summary: "Both children accept together; a rejected or dropped candidate changes neither.",
            doc: CONTROLLED_GATE_DOC,
        },
        Limit {
            id: "target-separate-from-opening",
            summary: "Requested targets and actual openings stay separate; flow proposals read actual opening only.",
            doc: CONTROLLED_GATE_DOC,
        },
        Limit {
            id: "binding-child-policies",
            summary: "Child qualification timing, opening uncertainty and arrival windows stay binding.",
            doc: CONTROLLED_GATE_DOC,
        },
    ],
    summary: Summary {
        equations: "sampled hysteresis targets driving ideal rate-limited gate opening",
        method: "one common calendar, old target through the decision's left state, new command on its right, atomic child acceptance",
        assessment: "selected analytical switching, reversal and restart cases; field accuracy remains unestablished",
    },
};

/// Returns the controlled gate's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn controlled_gate_capabilities() -> &'static ComponentCapabilities {
    &CONTROLLED_GATE
}
