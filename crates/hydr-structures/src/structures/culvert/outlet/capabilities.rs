//! Capability declaration of the culvert outlet-control headwater query.

use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines outlet rows, the march, statuses and limits.
const CULVERT_OUTLET_DOC: &str = "docs/structures.md";

const fn unsupported(
    id: &'static str,
    reason: UnsupportedReason,
    summary: &'static str,
) -> Unsupported {
    Unsupported {
        id,
        reason,
        summary,
        doc: CULVERT_OUTLET_DOC,
    }
}

const fn limit(id: &'static str, summary: &'static str) -> Limit {
    Limit {
        id,
        summary,
        doc: CULVERT_OUTLET_DOC,
    }
}

static CULVERT_OUTLET: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::CulvertOutlet,
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
            evidence: "culvert_outlet_rows_return_single_row_bits_at_every_layout_width",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_matches_culvert_outlets_bit_for_bit",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_culvert_outlet_f64_without_fallback",
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
            "invalid-outlet-row",
            UnsupportedReason::RejectedByAdmission,
            "An empty row list, a nonfinite or nonpositive length, Manning n or gravity, a nonfinite or negative entrance loss, or a nonfinite slope rejects.",
        ),
        unsupported(
            "adverse-beyond-range",
            UnsupportedReason::RejectedByAdmission,
            "An adverse barrel whose outlet invert sits more than half the rise above its inlet invert rejects.",
        ),
        unsupported(
            "invalid-query-input",
            UnsupportedReason::RejectedByAdmission,
            "A negative or nonfinite discharge or tailwater, or a tensor of another width, dtype or device, rejects the whole call.",
        ),
        unsupported(
            "culvert-discharge",
            UnsupportedReason::NotAttached,
            "Outlet headwater only; no inlet control, discharge from head or flow direction. The culvert structure law gives those.",
        ),
    ],
    limits: &[
        limit(
            "no-outlet-storage",
            "Queries hold no state and advance no storage; barrel storage is not represented.",
        ),
        limit(
            "zero-tail-velocity",
            "The exit loss takes a tail velocity of zero, so the free profile starts at max(TW, yc).",
        ),
        limit(
            "critical-inlet",
            "A steep barrel whose subcritical profile reaches critical depth, or has none, takes critical depth at the inlet.",
        ),
        limit(
            "critical-at-crown",
            "A critical depth at or above the crown makes the outlet flow full at max(TW, D).",
        ),
        limit(
            "unresolved-rows",
            "A row whose march comparisons are not certified, whose critical depth lies below the search floor, or whose start sits too close to a normal-depth pole reports Unresolved with NaN bounds.",
        ),
        limit(
            "inlet-energy-dip",
            "Where (1 + ke) Fr^2 > 1 at the inlet depth, y + (1 + ke) V^2/(2g) falls as that depth rises, so HW_o can fall as the tailwater drowns a steep inlet.",
        ),
        limit(
            "critical-limit-width",
            "Next to a critical-depth limit the inlet depth moves with the square root of the profile length, so enclosures widen there.",
        ),
        limit(
            "jump-free-outlet",
            "A jump is located only when the subcritical profile is free at the outlet and friction falls with depth across the jump.",
        ),
    ],
    summary: Summary {
        equations: "HDS-5 full-barrel loss with Ku = 2g, direct-step profiles dx/dy = (1 - Fr^2)/(S - Sf) from max(TW, yc), and the momentum function for the jump",
        method: "F64 interval enclosures; certified Simpson over a graded partition in a smooth section coordinate with order-4 Taylor jets for the fourth derivative; certified bisection for the inlet depth and the jump",
        assessment: "independent 100-digit full-barrel rows and adaptive-quadrature profile rows lie inside every enclosure within their frozen tolerances",
    },
};

/// Returns the culvert outlet query's declared scope, independent of application eligibility.
///
/// The enclosures establish no culvert discharge or field accuracy.
pub fn culvert_outlet_capabilities() -> &'static ComponentCapabilities {
    &CULVERT_OUTLET
}
