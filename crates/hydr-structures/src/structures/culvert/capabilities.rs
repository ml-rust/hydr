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
            "Batch construction certifies the depth slope floors and the conveyance peak of each elliptical, pipe-arch and flat-floor arch barrel. It rejects, naming the row, a barrel whose dimensions scaled by a power of two to a rise in [1, 2) fail admission, or whose conveyance has no certified single peak below the crown.",
        ),
        unsupported(
            "barrel-floor-domain",
            UnsupportedReason::RejectedByAdmission,
            "A floor rejects when its sediment depth is nonfinite or negative, reaches the rise, which is a closed barrel, or is zero on a barrel without its own floor. So does a nonfinite or nonpositive bed-to-wall Manning ratio, a ratio whose 3/2 power times the span leaves F64, and an open rise whose 2^-30 fraction is subnormal.",
        ),
        unsupported(
            "floored-barrel-queries",
            UnsupportedReason::RejectedByAdmission,
            "Critical and normal depth of a batch with a floored barrel reject, naming the row. Inlet-control rows, outlet-control rows and the culvert law reject a floored barrel.",
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
        Limit {
            id: "culvert-floor-roughness",
            summary: "A floored barrel reports its bed width and the weighted perimeter P_w + rho^(3/2) b of HDS-5 eq. 3.8 with exponent 2/3, for one Manning n per material. Depth searches, inlet control, outlet control and the culvert law take no floor yet, and the floor of a flat-floor arch without one takes the barrel's Manning n.",
            doc: CULVERT_BARREL_DOC,
        },
        Limit {
            id: "uniform-flat-bed",
            summary: "A floor is one flat bed at one depth along the whole barrel. Depths above a floor run up from the bed, and the open rise is D - h. The admitted depth is D - fl(D - h), within half an F64 step of D - h.",
            doc: CULVERT_BARREL_DOC,
        },
        Limit {
            id: "scaled-values-outside-f64",
            summary: "Geometry runs in each barrel's frame, scaled by a power of two to a rise in [1, 2), and scales back exactly. A metre area or moment outside the F64 range, as at a rise near 2^-900 m, rounds outward.",
            doc: CULVERT_BARREL_DOC,
        },
    ],
    summary: Summary {
        equations: "circular, box and tangent-arc closed-conduit geometry and their open sections above a flat bed, critical depth Q^2 T = g A^3 and Manning normal depth on the rising conveyance limb",
        method: "F64 interval enclosures with correctly rounded acos and log, arc bands from each wall's chord and circular segment, and certified bisection in s = y/D",
        assessment: "independent 100-digit references lie inside every enclosure; CPU and CUDA bounds match bit for bit on circular, box and tangent-arc barrels",
    },
};

/// Returns the culvert barrel queries' declared scope, independent of application eligibility.
///
/// The enclosures establish no culvert discharge or field accuracy.
pub fn culvert_barrel_capabilities() -> &'static ComponentCapabilities {
    &CULVERT_BARREL
}
