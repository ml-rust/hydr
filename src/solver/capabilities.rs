/// Version of the public numerical semantics and in-memory snapshot contract.
pub const CONTRACT_VERSION: &str = "hydr-swe-numr-cpu-1";

/// Declared implementation scope. Execution success establishes no field accuracy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Public contract identity.
    pub contract: &'static str,
    /// Governing equations.
    pub equations: &'static str,
    /// Numerical scheme identity.
    pub method: &'static str,
    /// Arithmetic and execution envelope.
    pub execution: &'static str,
    /// Arithmetic format used by every hydraulic state tensor.
    pub precision: &'static str,
    /// Admitted local sources.
    pub sources: &'static [&'static str],
    /// Admitted boundaries.
    pub boundaries: &'static [&'static str],
    /// Hydraulic structures attached to this solver. Standalone constitutive proposals are separate.
    pub structures: &'static [&'static str],
    /// Available physical outputs and numerical diagnostics.
    pub outputs: &'static [&'static str],
    /// Scope of the evidence accompanying this capability revision.
    pub assessment: &'static str,
    /// Unsupported capabilities and physics.
    pub unsupported: &'static [&'static str],
}

/// Returns the CPU reference's implemented scope, independent of application eligibility.
pub fn capabilities() -> Capabilities {
    Capabilities {
        contract: CONTRACT_VERSION,
        equations: "full 2D hydrostatic depth-averaged shallow-water equations with Manning resistance",
        method: "first-order finite volumes, hydrostatic side corrections, bracketed-bound HLL, forward Euler and split sources",
        execution: "numr CPU runtime, sequential f64 tensors, fixed traversal within the same build and execution environment",
        precision: "IEEE 754 binary64 (f64)",
        sources: &[
            "interval-constant cell rain with zero horizontal momentum",
            "explicit impervious or pinned Modified Green-Ampt land with complete native antecedent state",
            "atomic rain, returned native infiltration, capped evaporation, and prescribed withdrawal with donor momentum",
            "proved soil arithmetic reconciliation under an independent cumulative numerical water-adjustment budget",
        ],
        boundaries: &[
            "explicit genuine impermeable walls",
            "source-complete compatible constant-rate flux ports including pressure and conservative wave bounds",
            "wet-subcritical compatible stage traces with incoming tangent data",
            "wet-subcritical prescribed numerical discharge traces with incoming tangent data",
            "complete wet or dry exterior reservoir states through the numerical Riemann solver",
            "explicit dry or strictly supercritical-outgoing transmissive boundaries",
        ],
        structures: &[],
        outputs: &[
            "cell-supported depth",
            "wet-cell water elevation",
            "wet-cell depth-averaged velocity and discharge per width",
            "explicit dry/unsupported status",
            "accepted interval face fluxes and volumes",
            "component water ledger and momentum-reset diagnostics",
            "accepted complete native soil state, independent sink ledgers, per-cell source receipts, and conservative correction charges",
            "accepted physical boundary traces and compatibility residuals",
            "per-face final numerical signal speeds and accepted transmissive state witnesses",
            "complete versioned continuation packages",
        ],
        assessment: "selected synthetic analytical, conservation, refinement, restart, and flat wet/dry Clawpack comparisons; application accuracy remains unestablished",
        unsupported: &[
            "scalar stage/discharge traces outside wet-subcritical regimes",
            "finite external reservoir or component coupling",
            "groundwater and finite subsurface reservoir coupling",
            "catchment hydrology beyond the declared rain, soil, evaporation, and withdrawal mechanisms",
            "structures attached to this solver",
            "networks and coupled exchanges",
            "subgrid terrain",
            "3D and nonhydrostatic flow",
            "vertical jets",
            "sediment and erosion",
            "wind and Coriolis",
            "atmospheric pressure gradients and horizontal eddy viscosity",
            "GPU",
            "differentiation",
            "field accuracy guarantees",
        ],
    }
}
