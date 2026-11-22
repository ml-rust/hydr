//! Capability declaration of the culvert structure law.

use hydr_core::capability::{
    ArithmeticRequirement, ComponentCapabilities, ComponentId, Limit, Precision, RuntimeKind,
    RuntimeStatus, RuntimeSupport, Summary, Unsupported, UnsupportedReason,
};

/// Path of the guide that defines the law, its inversion, direction rules and limits.
const CULVERT_DOC: &str = "docs/structures.md";

const fn unsupported(
    id: &'static str,
    reason: UnsupportedReason,
    summary: &'static str,
) -> Unsupported {
    Unsupported {
        id,
        reason,
        summary,
        doc: CULVERT_DOC,
    }
}

const fn limit(id: &'static str, summary: &'static str) -> Limit {
    Limit {
        id,
        summary,
        doc: CULVERT_DOC,
    }
}

static CULVERT: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::Culvert,
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
            evidence: "culvert_lanes_return_single_lane_bits_at_every_layout_width",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_matches_culvert_laws_bit_for_bit",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_culvert_law_f64_without_fallback",
        },
    ],
    endpoints: &[],
    regimes: &[],
    sources: &[],
    options: &[],
    nested: &[ComponentId::CulvertInlet, ComponentId::CulvertOutlet],
    attachable_to: &[ComponentId::SurfaceChannels, ComponentId::CoupledLevelPools],
    unsupported: &[
        unsupported(
            "invalid-culvert-definition",
            UnsupportedReason::RejectedByAdmission,
            "A nonfinite invert, a zero barrel count, no culvert end, a negative entrance loss, or an end whose inlet or outlet row the queries reject, rejects.",
        ),
        unsupported(
            "culvert-arc-barrel",
            UnsupportedReason::RejectedByAdmission,
            "StructureBatch::with_culverts rejects a culvert whose barrel is elliptical, pipe-arch or flat-floor arch, naming the element. Their inlet and outlet control are not yet available.",
        ),
        unsupported(
            "culvert-end-mismatch",
            UnsupportedReason::RejectedByAdmission,
            "A culvert must declare an end for each direction its policy admits and no other.",
        ),
        unsupported(
            "culvert-falling-transition",
            UnsupportedReason::RejectedByAdmission,
            "An end whose inlet headwater does not certifiably rise across its transition rejects, naming the element.",
        ),
        unsupported(
            "culvert-without-certified-client",
            UnsupportedReason::RejectedByAdmission,
            "StructureBatch::new rejects a culvert. StructureBatch::with_culverts admits it on a client with certified roots and quadrature.",
        ),
        unsupported(
            "buried-dry-end",
            UnsupportedReason::RejectedByAdmission,
            "A known dry end whose bed lies above its culvert invert rejects.",
        ),
        unsupported(
            "culvert-blockage",
            UnsupportedReason::OutsideModel,
            "No blockage, sediment or debris; the barrel is clean and fully open.",
        ),
    ],
    limits: &[
        limit(
            "culvert-barrel-storage",
            "The law holds no water. Barrel storage is not represented, and proposals keep no state.",
        ),
        limit(
            "culvert-level-zero",
            "A lane whose outlet still level max(TW - S L, 0) is not certified below the headwater carries exact zero under EqualHeads.",
        ),
        limit(
            "culvert-zero-headwater",
            "A headwater at or below HW_i(0) = D Ks S, or below a certified still level, gives exact zero discharge.",
        ),
        limit(
            "culvert-bidirectional-fall",
            "Each admitted direction keeps its adverse fall within half the rise, so a two-way culvert keeps its inverts within D/2.",
        ),
        limit(
            "culvert-unresolved-outlet",
            "Where the outlet march leaves its headwater unresolved near the inlet discharge, the proposal rejects as Numerical. With critical depth below the crown, a start friction equal to the slope or a full-barrel head equal to the crown at the inlet within rounding has no certified outcome or bounded hull. Critical depth below 2^-30 D and a start within 2^-34 of a normal-depth pole also stay unresolved.",
        ),
        limit(
            "culvert-sensitivity-cap",
            "Head sensitivity is capped at 2^256 per metre. An outlet lane whose march row is open, a tie or unresolved, or whose divisors are not certified positive, takes the cap.",
        ),
        limit(
            "culvert-outlet-cost",
            "Every flowing lane runs one outlet march. Each lane the inlet does not certainly govern adds one certified bisection with one outlet march per round. An outlet-controlled lane that is not a drowned full barrel adds one coarse march for its head sensitivity.",
        ),
    ],
    summary: Summary {
        equations: "HDS-5: discharge is min(Q_i, Q_o), the inverse of the larger of the inlet and outlet control headwaters at fixed tailwater",
        method: "closed-form inlet inverses per band and one certified bisection in the form-1 critical-depth fraction; a drowned full barrel in closed form and otherwise one certified bisection over the certified outlet march",
        assessment: "independent 100-digit inlet and outlet corpus rows return their discharge within the frozen round-trip tolerance",
    },
};

/// Returns the culvert structure law's declared scope, independent of application eligibility.
pub fn culvert_capabilities() -> &'static ComponentCapabilities {
    &CULVERT
}
