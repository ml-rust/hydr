//! Capability declaration of the culvert barrel geometry and depth queries.

use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines barrel shapes, queries, statuses and limits.
const CULVERT_BARREL_DOC: &str = "docs/structures.md";

const fn unsupported(
    id: &'static str,
    reason: UnsupportedReason,
    summary: &'static str,
) -> Unsupported {
    Unsupported {
        id,
        reason,
        summary,
        doc: CULVERT_BARREL_DOC,
    }
}

static CULVERT_BARREL: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::CulvertBarrel,
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
            evidence: "culvert_barrel_queries_return_single_row_bits_at_every_layout_width",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_matches_culvert_barrels_bit_for_bit",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_culvert_barrel_f64_without_fallback",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[],
    options: &[],
    nested: &[],
    attachable_to: &[],
    unsupported: &[
        unsupported(
            "unrepresentable-barrel-dimensions",
            UnsupportedReason::RejectedByAdmission,
            "A nonfinite or nonpositive span, rise or given radius rejects. So does a rise whose 2^-30 fraction is subnormal, dimensions whose area, perimeter or moment scale overflows F64, or a given or derived radius whose cube overflows F64.",
        ),
        unsupported(
            "arc-barrel-dimensions",
            UnsupportedReason::RejectedByAdmission,
            "An elliptical, pipe-arch or flat-floor arch barrel whose dimensions break its tangent-arc rules rejects at construction. Each message names the rule, the values and the dimension to change.",
        ),
        unsupported(
            "arc-barrel-certificate",
            UnsupportedReason::RejectedByAdmission,
            "Batch construction certifies the depth slope floors and the conveyance peak of each elliptical, pipe-arch and flat-floor arch barrel. It rejects, naming the row, a barrel whose dimensions scaled to a unit rise fail admission, or whose conveyance has no certified single peak below the crown.",
        ),
        unsupported(
            "empty-barrel-batch",
            UnsupportedReason::RejectedByAdmission,
            "A batch without barrels rejects.",
        ),
        unsupported(
            "invalid-query-input",
            UnsupportedReason::RejectedByAdmission,
            "A negative or nonfinite depth or discharge, a nonpositive Manning n or gravity, a nonfinite slope, or a tensor of another width rejects the whole call.",
        ),
        unsupported(
            "range-reaching-crown",
            UnsupportedReason::RejectedByAdmission,
            "A depth range that is reversed or reaches the crown rejects. Split it at the crown.",
        ),
        unsupported(
            "other-barrel-shapes",
            UnsupportedReason::OutsideModel,
            "No corrugated metal box, long-span structural plate, true ellipse or tapered sections.",
        ),
        unsupported(
            "culvert-discharge",
            UnsupportedReason::NotAttached,
            "Geometry and depths only; no inlet control, outlet control, barrel profile or culvert discharge.",
        ),
    ],
    limits: &[
        Limit {
            id: "no-barrel-storage",
            summary: "Queries hold no state and advance no storage; no snapshot or checkpoint exists.",
            doc: CULVERT_BARREL_DOC,
        },
        Limit {
            id: "free-surface-depth-domain",
            summary: "Range geometry covers 0 <= lower <= upper < D. Depth searches cover [D 2^-30, top]; statuses report roots outside it.",
            doc: CULVERT_BARREL_DOC,
        },
        Limit {
            id: "full-state-constants-only",
            summary: "A depth at or above the crown returns the full-barrel constants; pressure flow and full-barrel friction stay with the caller.",
            doc: CULVERT_BARREL_DOC,
        },
    ],
    summary: Summary {
        equations: "circular, box and tangent-arc closed-conduit geometry, and on circular and box barrels critical depth Q^2 T = g A^3 and Manning normal depth on the rising conveyance limb",
        method: "F64 interval enclosures with correctly rounded acos and log, arc bands from each wall's chord and circular segment, and certified bisection in s = y/D",
        assessment: "independent 100-digit references lie inside every enclosure; CPU and CUDA bounds match bit for bit on circular and box barrels",
    },
};

/// Returns the culvert barrel queries' declared scope, independent of application eligibility.
///
/// The enclosures establish no culvert discharge or field accuracy.
pub fn culvert_barrel_capabilities() -> &'static ComponentCapabilities {
    &CULVERT_BARREL
}
