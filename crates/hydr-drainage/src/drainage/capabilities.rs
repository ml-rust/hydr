//! Capability declaration of the pinned EPA SWMM v5.2.4 drainage component.

use super::{SWMM_DRAINAGE_CONTRACT_VERSION, SWMM_DRAINAGE_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, OptionSet, Precision, Regime,
    RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus, RuntimeSupport, SourceKind, Summary,
    Unsupported, UnsupportedReason,
};

const DRAINAGE_DOC: &str = "docs/drainage.md";

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

static SWMM_DRAINAGE: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::SwmmDrainage,
    contract: Some(SWMM_DRAINAGE_CONTRACT_VERSION),
    snapshot_schema: Some(SWMM_DRAINAGE_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
    ],
    runtimes: &[RuntimeSupport {
        runtime: RuntimeKind::Cpu,
        status: RuntimeStatus::Reference,
        evidence: "swmm_drainage_rollback_and_cold_restart_are_bit_identical",
    }],
    endpoints: &[],
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
    sources: &[SourceKind::ExchangeVolume],
    options: &[OptionSet {
        id: "surcharge-method",
        values: &["extran", "slot"],
        default: None,
    }],
    nested: &[],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "surcharge-method-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "Opening rejects if the declared surcharge method disagrees with the SWMM .inp SURCHARGE_METHOD.",
            DRAINAGE_DOC,
        ),
        unsupported(
            "missing-worker-binary",
            UnsupportedReason::RejectedByAdmission,
            "Opening rejects with BackendUnavailable when the pinned swmm-drainage-worker binary is absent.",
            DRAINAGE_DOC,
        ),
        unsupported(
            "tampered-snapshot",
            UnsupportedReason::RejectedByAdmission,
            "Snapshot import rejects mismatched schema, contract, .inp SHA-256, datum, surcharge method, or payload digest.",
            DRAINAGE_DOC,
        ),
        unsupported(
            "native-control-authority",
            UnsupportedReason::NotAttached,
            "Standalone drainage admits NativeSwmm control authority only; coupled hydr/SWMM authority attaches in surface-drainage coupling.",
            DRAINAGE_DOC,
        ),
        unsupported(
            "gpu-execution",
            UnsupportedReason::OutsideModel,
            "The pinned EPA SWMM v5.2.4 C worker executes on host CPU only.",
            DRAINAGE_DOC,
        ),
    ],
    limits: &[
        Limit {
            id: "pinned-swmm-v5-2-4",
            summary: "Requires the pinned EPA SWMM v5.2.4 worker built via tools/reference/swmm_drainage_build.py.",
            doc: DRAINAGE_DOC,
        },
        Limit {
            id: "si-boundary-conversion",
            summary: "Converts SWMM internal feet and cfs to canonical SI meters and cubic meters per second using exact 0.3048 m/ft.",
            doc: DRAINAGE_DOC,
        },
    ],
    summary: Summary {
        equations: "1D Saint-Venant dynamic-wave closed-conduit routing with EXTRAN or Preissmann slot pressurized surcharge (EPA SWMM v5.2.4).",
        method: "Isolated pinned EPA SWMM v5.2.4 worker process with canonical SI boundary conversion, transactional rollback, and complete hidden-state snapshots.",
        assessment: "Reference CPU-only backend; verified bit-identical across trial rollback and cold-process restart.",
    },
};

/// Returns the capability declaration of the pinned EPA SWMM v5.2.4 drainage component.
pub fn swmm_drainage_capabilities() -> &'static ComponentCapabilities {
    &SWMM_DRAINAGE
}
