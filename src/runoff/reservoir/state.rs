//! Explicit element identity, antecedent depth, and interval input.

use super::ReservoirParameters;

/// Identity and immutable physical definition of one storage element.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirElement {
    /// Unique nonempty identity without control characters. This asserts no GIS disjointness.
    pub id: String,
    /// Checked physical storage and closure parameters.
    pub parameters: ReservoirParameters,
}

/// Complete antecedent reservoir state in SI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReservoirState {
    /// Nonnegative represented depth in metres.
    pub depth_m: f64,
}

/// Constant incoming supply for the exact requested interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReservoirInput {
    /// Nonnegative incoming rate in metres/second, independent of provider provenance.
    pub incoming_rate_m_per_s: f64,
}
