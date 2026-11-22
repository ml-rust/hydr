//! Capability declaration of the culvert inlet-control headwater query.

use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines inlet definitions, bands, admission and limits.
const CULVERT_INLET_DOC: &str = "docs/structures.md";

const fn unsupported(
    id: &'static str,
    reason: UnsupportedReason,
    summary: &'static str,
) -> Unsupported {
    Unsupported {
        id,
        reason,
        summary,
        doc: CULVERT_INLET_DOC,
    }
}

static CULVERT_INLET: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::CulvertInlet,
    contract: None,
    snapshot_schema: None,
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::RoundToNearestSqrt,
        ArithmeticRequirement::FusedMulAdd,
        ArithmeticRequirement::CorrectlyRoundedElementary,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "culvert_inlet_rows_return_single_row_bits_at_every_layout_width",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_matches_culvert_inlets_bit_for_bit",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_culvert_inlet_f64_without_fallback",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[],
    options: &[],
    nested: &[],
    attachable_to: &[ComponentId::Culvert],
    unsupported: &[
        unsupported(
            "unadmitted-inlet-definition",
            UnsupportedReason::RejectedByAdmission,
            "An HDS-5 code outside 1 to 26 rejects. So do explicit coefficients with a nonfinite or nonpositive K, M or c, or a nonfinite Y.",
        ),
        unsupported(
            "invalid-inlet-row",
            UnsupportedReason::RejectedByAdmission,
            "An empty row list, a nonfinite slope, a nonfinite or nonpositive gravity, a coded inlet on a barrel of another shape, or an elliptical, pipe-arch or flat-floor arch barrel rejects.",
        ),
        unsupported(
            "falling-transition",
            UnsupportedReason::RejectedByAdmission,
            "A row whose submerged headwater at X = 4 does not certifiably exceed its unsubmerged headwater at X = 3.5 rejects. So does a form-1 row whose critical depth at X = 3.5 is not below the crown.",
        ),
        unsupported(
            "invalid-query-input",
            UnsupportedReason::RejectedByAdmission,
            "A negative or nonfinite discharge, or a tensor of another width, dtype or device, rejects the whole call.",
        ),
        unsupported(
            "other-inlet-codes",
            UnsupportedReason::OutsideModel,
            "HDS-5 codes 27 to 57: corrugated metal boxes, ellipses, arches, pipe arches and tapered inlets.",
        ),
        unsupported(
            "culvert-discharge",
            UnsupportedReason::NotAttached,
            "Inlet headwater only; no outlet control, discharge from head or flow direction. The culvert structure law gives those.",
        ),
    ],
    limits: &[
        Limit {
            id: "no-inlet-storage",
            summary: "Queries hold no state and advance no storage; no snapshot or checkpoint exists.",
            doc: CULVERT_INLET_DOC,
        },
        Limit {
            id: "linear-transition",
            summary: "Between X = 3.5 and 4 the headwater is linear in X between the two limit headwaters. HDS-5 draws a hand curve there.",
            doc: CULVERT_INLET_DOC,
        },
        Limit {
            id: "slope-as-given",
            summary: "The slope term Ks S takes the signed slope as given. The structure law bounds adverse slopes.",
            doc: CULVERT_INLET_DOC,
        },
        Limit {
            id: "form-one-zero-discharge",
            summary: "At zero discharge a form-1 row returns D Ks S, the value of its equation, not zero.",
            doc: CULVERT_INLET_DOC,
        },
    ],
    summary: Summary {
        equations: "HDS-5 inlet control: unsubmerged forms 1 and 2, the submerged form, and a linear transition between X = 3.5 and 4",
        method: "F64 interval enclosures with correctly rounded log and exp powers, certified critical depth for the form-1 critical head, and the exact SI conversion Ku = 0.3048^(-1/2)",
        assessment: "independent 100-digit headwater curves and limit anchors lie inside every enclosure; SWMM rows agree within their fixed tolerances except SWMM's stale form-1 flows",
    },
};

/// Returns the culvert inlet query's declared scope, independent of application eligibility.
///
/// The enclosures establish no culvert discharge or field accuracy.
pub fn culvert_inlet_capabilities() -> &'static ComponentCapabilities {
    &CULVERT_INLET
}
