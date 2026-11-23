//! Capability declaration of sampled operating controls.

use super::{OPERATING_CONTROL_CONTRACT_VERSION, OPERATING_CONTROL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines thresholds, samples, qualification and limits.
const CONTROL_DOC: &str = "docs/operating-controls.md";

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

static OPERATING_CONTROL: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::OperatingControl,
    contract: Some(OPERATING_CONTROL_CONTRACT_VERSION),
    snapshot_schema: Some(OPERATING_CONTROL_SNAPSHOT_SCHEMA),
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
            evidence: "endpoint_commands_apply_to_the_right_interval_only",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_operating_control_relay_and_complete_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_operating_control_f64_without_fallback",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[],
    options: &[],
    nested: &[],
    attachable_to: &[
        ComponentId::ControlledGate,
        ComponentId::ControlledPump,
        ComponentId::SurfaceChannels,
    ],
    unsupported: &[
        unsupported(
            "inverted-thresholds",
            UnsupportedReason::RejectedByAdmission,
            "A lower stage at or above the upper stage, or a negative delay, rejects.",
            CONTROL_DOC,
        ),
        unsupported(
            "law-incompatible-command",
            UnsupportedReason::RejectedByAdmission,
            "A command outside the controlled device's law, speed policy or opening range rejects.",
            CONTROL_DOC,
        ),
        unsupported(
            "missing-sample",
            UnsupportedReason::RejectedByAdmission,
            "A decision without one ordered sample per controller, or a sample between decisions, rejects.",
            CONTROL_DOC,
        ),
        unsupported(
            "foreign-sample",
            UnsupportedReason::RejectedByAdmission,
            "A sample whose controller, signal, support or datum differs from the declaration rejects.",
            CONTROL_DOC,
        ),
        unsupported(
            "nonfinite-sample",
            UnsupportedReason::RejectedByAdmission,
            "A sample with a nonfinite stage, observation or availability time rejects.",
            CONTROL_DOC,
        ),
        unsupported(
            "stale-sample",
            UnsupportedReason::RejectedByAdmission,
            "A sample observed before its decision rejects; no last-value hold applies.",
            CONTROL_DOC,
        ),
        unsupported(
            "future-sample",
            UnsupportedReason::RejectedByAdmission,
            "A sample observed after its decision, or available after its observation, rejects.",
            CONTROL_DOC,
        ),
        unsupported(
            "minimum-dwell",
            UnsupportedReason::OutsideModel,
            "No minimum on/off dwell.",
            CONTROL_DOC,
        ),
        unsupported(
            "continuous-event-localization",
            UnsupportedReason::OutsideModel,
            "No continuous threshold history or localization of crossings between samples.",
            CONTROL_DOC,
        ),
        unsupported(
            "rule-grammar-and-pid",
            UnsupportedReason::OutsideModel,
            "No generic rule grammar or PID control.",
            CONTROL_DOC,
        ),
        unsupported(
            "native-swmm-rules",
            UnsupportedReason::OutsideModel,
            "No native SWMM rule execution.",
            CONTROL_DOC,
        ),
        unsupported(
            "device-response",
            UnsupportedReason::NotAttached,
            "Commands request operation only; controlled gates and pumps supply actual response.",
            CONTROL_DOC,
        ),
        unsupported(
            "gpu-numerical-qualification",
            UnsupportedReason::Unqualified,
            "CUDA runs execute, but only the CPU reference qualifies numerical accuracy.",
            CONTROL_DOC,
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
            id: "gate-blocked-range",
            summary: "A blocked gate admits target commands up to its blocked maximum opening; a larger command rejects, never clamps.",
            doc: CONTROL_DOC,
        },
        Limit {
            id: "sampled-qualification",
            summary: "Qualification persists across supplied samples only, never across unobserved excursions.",
            doc: CONTROL_DOC,
        },
        Limit {
            id: "original-timestamp-timers",
            summary: "Timers age from original timestamps through TwoSum elapsed time, never repeated interval sums.",
            doc: CONTROL_DOC,
        },
        Limit {
            id: "simultaneous-update",
            summary: "All controllers read one original sample batch and update simultaneously.",
            doc: CONTROL_DOC,
        },
    ],
    summary: Summary {
        equations: "two-threshold hysteresis relay with high and low qualification delays",
        method: "sampled decisions on an original calendar, TwoSum timer ages, immutable trials",
        assessment: "pinned original-bit reference histories; field accuracy remains unestablished",
    },
};

/// Returns the sampled operating control's declared scope, independent of application eligibility.
///
/// Execution success establishes no field accuracy.
pub fn operating_control_capabilities() -> &'static ComponentCapabilities {
    &OPERATING_CONTROL
}
