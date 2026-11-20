//! Declared rejections and scope boundaries of the surface/channel coordinator.

use super::SURFACE_CHANNELS_DOC;
use hydr_core::capability::{Unsupported, UnsupportedReason};

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

/// Admission rejections first, then scope boundaries.
pub(super) const UNSUPPORTED: &[Unsupported] = &[
    unsupported(
        "unobstructed-connection",
        UnsupportedReason::RejectedByAdmission,
        "A crest below the surface bed or the channel bed rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "port-section-mismatch",
        UnsupportedReason::RejectedByAdmission,
        "A port on a triangular reach end, or on a surveyed end whose section is not a rectangle, rejects. Only a rectangular section matches a strip of surface faces.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "port-width-mismatch",
        UnsupportedReason::RejectedByAdmission,
        "A port whose face widths sum to other than the end cell width, past 16 epsilon of that width, rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "port-face-ownership",
        UnsupportedReason::RejectedByAdmission,
        "A port face without exactly one active cell, on a caller open face, in two ports, off one face line, or with a gap rejects, and so do two ports on one reach end and an unknown reach.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "port-endpoint-policy",
        UnsupportedReason::RejectedByAdmission,
        "A caller policy at a ported reach end rejects, and so does a missing policy at an unported end.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "duplicate-links",
        UnsupportedReason::RejectedByAdmission,
        "A repeated link, port, structure, pool, pool-end, or scenario identity, or a repeated surface cell, reach and channel cell, rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "inactive-surface-cell",
        UnsupportedReason::RejectedByAdmission,
        "A link to an inactive surface cell or a cell outside the grid rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "non-unit-tangent",
        UnsupportedReason::RejectedByAdmission,
        "A nonfinite reach tangent or one off unit length rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "datum-or-gravity-mismatch",
        UnsupportedReason::RejectedByAdmission,
        "A reach, structure, pool curve, or control signal in another vertical datum, or a reach or structure law with other gravity bits, rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "step-and-retry-limits",
        UnsupportedReason::RejectedByAdmission,
        "Common step or retry limits outside the child limits reject, and so does a common interval below the minimum step.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "reserved-binding-prefix",
        UnsupportedReason::RejectedByAdmission,
        "A caller lateral binding or input under the `surface-exchange:` or `structure-exchange:` prefix rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "non-symmetric-weir",
        UnsupportedReason::RejectedByAdmission,
        "A bank weir coefficient without symmetric reversal evidence rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "missing-link-evidence",
        UnsupportedReason::RejectedByAdmission,
        "Empty mapping, connection, scenario, or storage-exclusion evidence rejects, and so does an empty identity.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "integrated-lateral-span",
        UnsupportedReason::RejectedByAdmission,
        "An integrated caller lateral volume or pool inflow volume off the exact common interval rejects or requests recomputation.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "structure-endpoint-identity",
        UnsupportedReason::RejectedByAdmission,
        "A structure endpoint on no active surface cell, no existing channel cell, or no pool, both endpoints on one node, or an element endpoint other than its node identity rejects, and so does a pool end on an unknown reach or pool.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "unobstructed-pool-connection",
        UnsupportedReason::RejectedByAdmission,
        "A structure attachment bed below its node bed or pool origin, a weir crest, gate sill, or orifice sill below either attachment bed, or a pool-end cell bed below the pool origin rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "non-symmetric-structure",
        UnsupportedReason::RejectedByAdmission,
        "A bidirectional pump, or a bidirectional structure law without symmetric reversal evidence, rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "reach-end-ownership",
        UnsupportedReason::RejectedByAdmission,
        "A reach end with a port and a pool end, or two pool ends, rejects, and so does a caller policy at a pool end.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "pool-support",
        UnsupportedReason::RejectedByAdmission,
        "Pool support, source policies, or inflows that differ from the pools, a repeated storage support, a zero-volume stage off the curve origin, or a pool that joins no structure and no pool end rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "operation-ownership",
        UnsupportedReason::RejectedByAdmission,
        "An operation that does not drive its law, a dynamic owner without its operating child, a structure with zero owners or two, a child off the common time, or a signal on no coordinator node, bound twice, or missing rejects.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "pool-capacity-exceeded",
        UnsupportedReason::RejectedByAdmission,
        "A pool that fills past its top curve knot rejects without retry; declare a spill structure or extend the curve.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "pump-without-suction",
        UnsupportedReason::RejectedByAdmission,
        "A running pump whose donor stands at or below its suction bound at the start rejects without retry.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "pump-dry-delivery",
        UnsupportedReason::RejectedByAdmission,
        "A running pump that delivers to a dry receiver without a declared outlet rejects without retry.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "pool-end-regime",
        UnsupportedReason::RejectedByAdmission,
        "A dry pool end at the accepted start, or a nonfinite or negative entrance loss, rejects; pool ends need wet subcritical support.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "orifice-partial-submergence",
        UnsupportedReason::RejectedByAdmission,
        "An orifice with a side below its crown rejects; partial orifice flow is no admitted law.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "tensor-forcing",
        UnsupportedReason::NotAttached,
        "Channel and pool forcing is host data only; no device-resident forcing interval exists.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "structure-geometry-events",
        UnsupportedReason::NotAttached,
        "Sills, crests, areas, and blockage stay fixed over the run; no geometry event changes them.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "runoff-attachment",
        UnsupportedReason::NotAttached,
        "Runoff attaches to level pools through runoff pools, not to coordinator pools.",
        "docs/runoff-pools.md",
    ),
    unsupported(
        "culvert-inlet-outlet-control",
        UnsupportedReason::OutsideModel,
        "No culvert inlet or outlet control; an orifice approximates a culvert and does not satisfy culvert support.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "blockage-probability",
        UnsupportedReason::OutsideModel,
        "No blockage probability or inferred blockage; a fixed scenario is caller geometry with evidence.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "pump-dynamics",
        UnsupportedReason::OutsideModel,
        "No pump priming, net positive suction head, or motor dynamics.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "continuous-control",
        UnsupportedReason::OutsideModel,
        "Controls decide at sampled times only; no continuous feedback, minimum dwell, or native SWMM rules.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "pool-port",
        UnsupportedReason::OutsideModel,
        "No port carries momentum into a pool; a pool holds no momentum.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "subcycling",
        UnsupportedReason::OutsideModel,
        "Every child advances the same common interval; no child subcycles.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "normal-momentum",
        UnsupportedReason::OutsideModel,
        "Bank link water carries only the velocity along the reach tangent; its normal momentum dissipates. Ports carry normal momentum.",
        SURFACE_CHANNELS_DOC,
    ),
    unsupported(
        "gpu-numerical-qualification",
        UnsupportedReason::Unqualified,
        "CUDA runs execute, but no test qualifies their numerical accuracy.",
        "docs/integration.md",
    ),
    unsupported(
        "measured-event-accuracy",
        UnsupportedReason::Unqualified,
        "No measured-event assessment establishes field accuracy.",
        "docs/acceptance.md",
    ),
];
