//! Capability declaration of the pure-Rust Preissmann-slot pressurized-pipe drainage solver.

use super::{NATIVE_PIPE_CONTRACT_VERSION, NATIVE_PIPE_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

const DOC: &str = "docs/native-pipe-drainage.md";

static OUTFALL_BOUNDARIES: &[BoundaryKind] = &[BoundaryKind::Stage, BoundaryKind::Transmissive];

static NATIVE_PIPE_DRAINAGE: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::NativePipeDrainage,
    contract: Some(NATIVE_PIPE_CONTRACT_VERSION),
    snapshot_schema: Some(NATIVE_PIPE_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[RuntimeSupport {
        runtime: RuntimeKind::Cpu,
        status: RuntimeStatus::Reference,
        evidence: "native_pipe_gravity_surcharge_control_rollback_and_restart",
    }],
    endpoints: &[Endpoint {
        id: EndpointId::Outlet,
        admitted: OUTFALL_BOUNDARIES,
    }],
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
    sources: &[SourceKind::LateralInflow, SourceKind::LateralWithdrawal],
    options: &[],
    nested: &[],
    attachable_to: &[],
    unsupported: &[
        Unsupported {
            id: "invalid-pipe-network",
            reason: UnsupportedReason::RejectedByAdmission,
            summary: "Duplicate node/conduit ids, unknown node references, self-loops, or non-positive geometry reject at construction.",
            doc: DOC,
        },
        Unsupported {
            id: "tampered-pipe-snapshot",
            reason: UnsupportedReason::RejectedByAdmission,
            summary: "Snapshot import rejects mismatched schema, contract, datum, lengths, or non-finite/negative depths.",
            doc: DOC,
        },
    ],
    limits: &[Limit {
        id: "preissmann-slot-link-node",
        summary: "Uses link-node 1D dynamic-wave momentum with continuous Preissmann-slot pressurization; no external C SWMM binary is required.",
        doc: DOC,
    }],
    summary: Summary {
        equations: "1D dynamic-wave closed-conduit momentum and nodal continuity with continuous Preissmann surcharge slot.",
        method: "Pure-Rust link-node semi-implicit Manning friction solver with exact SI continuity accounting, link setting control, transactional rollback, and portable checkpoints.",
        assessment: "Zero-dependency pure-Rust pressurized-pipe solver verified for gravity flow, pressurized surcharge, link actuation, rollback, and bit-identical restart.",
    },
};

/// Returns the capability declaration of the pure-Rust pressurized-pipe drainage solver.
pub fn native_pipe_drainage_capabilities() -> &'static ComponentCapabilities {
    &NATIVE_PIPE_DRAINAGE
}
