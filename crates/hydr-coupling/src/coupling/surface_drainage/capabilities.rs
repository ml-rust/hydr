//! Capability declaration of the conservative 2D surface / SWMM drainage coordinator.

use super::{SURFACE_DRAINAGE_CONTRACT_VERSION, SURFACE_DRAINAGE_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, Regime,
    RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus, RuntimeSupport, SourceKind, Summary,
    Unsupported, UnsupportedReason,
};

const SURFACE_DRAINAGE_DOC: &str = "docs/surface-drainage.md";

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

static SURFACE_DRAINAGE: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::SurfaceDrainage,
    contract: Some(SURFACE_DRAINAGE_CONTRACT_VERSION),
    snapshot_schema: Some(SURFACE_DRAINAGE_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::FusedMulAdd,
        ArithmeticRequirement::CorrectlyRoundedElementary,
    ],
    runtimes: &[RuntimeSupport {
        runtime: RuntimeKind::Cpu,
        status: RuntimeStatus::Reference,
        evidence: "surface_drainage_capture_surcharge_rollback_and_restart_conserve_mass",
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
    options: &[],
    nested: &[ComponentId::SurfaceSolver, ComponentId::SwmmDrainage],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "duplicate-parameter-authority",
            UnsupportedReason::RejectedByAdmission,
            "A SWMM link controlled by native [CONTROLS] or claimed twice, or a repeated inlet id or (cell, node) pair, rejects.",
            SURFACE_DRAINAGE_DOC,
        ),
        unsupported(
            "datum-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "A vertical datum mismatch between setup and SWMM drainage, or an inlet rim below node invert or surface bed, rejects.",
            SURFACE_DRAINAGE_DOC,
        ),
        unsupported(
            "gpu-execution",
            UnsupportedReason::OutsideModel,
            "The nested pinned EPA SWMM v5.2.4 worker executes on host CPU only.",
            SURFACE_DRAINAGE_DOC,
        ),
    ],
    limits: &[
        Limit {
            id: "midpoint-inlet-closure",
            summary: "Uses midpoint-head weir/orifice inlet closure with proportional surface donor limiting and step halving on nonconvergence.",
            doc: SURFACE_DRAINAGE_DOC,
        },
        Limit {
            id: "pinned-swmm-v5-2-4",
            summary: "Requires the pinned EPA SWMM v5.2.4 worker built via tools/reference/swmm_drainage_build.py.",
            doc: SURFACE_DRAINAGE_DOC,
        },
    ],
    summary: Summary {
        equations: "2D shallow-water surface flow coupled to 1D dynamic-wave EPA SWMM v5.2.4 piped drainage through bidirectional weir/orifice inlets.",
        method: "Conservative midpoint-head fixed-point exchange with surface donor reservation, complete SWMM hidden-state rollback, and exclusive control-authority checks.",
        assessment: "Reference CPU coordinator verified for capture, pressurized surcharge, rollback, and dual-component restart.",
    },
};

/// Returns the capability declaration of the conservative surface-drainage coordinator.
pub fn surface_drainage_capabilities() -> &'static ComponentCapabilities {
    &SURFACE_DRAINAGE
}
