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
    /// Admitted hydraulic structures. Empty until their numerical gates pass.
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
            "prescribed land loss with donor momentum",
        ],
        boundaries: &[
            "explicit genuine impermeable walls",
            "source-complete compatible constant-rate flux ports including pressure and conservative wave bounds",
            "wet-subcritical compatible stage traces with incoming tangent data",
            "wet-subcritical prescribed numerical discharge traces with incoming tangent data",
        ],
        structures: &[],
        outputs: &[
            "cell-supported depth",
            "wet-cell water elevation",
            "wet-cell depth-averaged velocity and discharge per width",
            "explicit dry/unsupported status",
            "accepted interval face fluxes and volumes",
            "component water ledger and momentum-reset diagnostics",
            "accepted physical boundary traces and compatibility residuals",
            "complete versioned continuation packages",
        ],
        assessment: "selected synthetic analytical, conservation, refinement, restart, and flat wet/dry Clawpack comparisons; application accuracy remains unestablished",
        unsupported: &[
            "dry, critical, and supercritical characteristic boundaries or exterior-state reservoir closures",
            "soil infiltration/recovery models",
            "hydrology",
            "structures",
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
