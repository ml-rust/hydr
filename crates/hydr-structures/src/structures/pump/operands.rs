//! Immutable evaluated pump operands and reviewed normalization evidence.

/// Complete evaluated running/shutoff pump operands. Isolated proposals expose no instance.
#[derive(Clone, Debug, PartialEq)]
pub struct PumpHydraulicOperands {
    pub(crate) speed_ratio: f64,
    pub(crate) receiving_minus_donor_head_m: f64,
    pub(crate) receiving_minus_donor_head_low_m: f64,
    pub(crate) reference_head_m: f64,
    pub(crate) normalization_error_bound_m: f64,
    pub(crate) reference_head_lower_m: f64,
    pub(crate) reference_head_upper_m: f64,
    pub(crate) reference_discharge_m3_per_s: f64,
    pub(crate) division_exact: bool,
}
impl PumpHydraulicOperands {
    /// Original positive speed ratio.
    pub fn speed_ratio(&self) -> f64 {
        self.speed_ratio
    }
    /// High component of the oriented lift, in metres.
    ///
    /// The lift is receiving stage minus donor stage. With a free outlet the receiving term is
    /// `max(receiving stage, outlet)`, and a dry receiver's bed stands in for its stage.
    pub fn receiving_minus_donor_head_m(&self) -> f64 {
        self.receiving_minus_donor_head_m
    }
    /// Signed TwoSum subtraction witness, in metres.
    pub fn receiving_minus_donor_head_low_m(&self) -> f64 {
        self.receiving_minus_donor_head_low_m
    }
    /// Actual unchanged normalized Solvr query, in metres.
    pub fn reference_head_m(&self) -> f64 {
        self.reference_head_m
    }
    /// Conservative absolute normalized-head uncertainty, in metres.
    pub fn normalization_error_bound_m(&self) -> f64 {
        self.normalization_error_bound_m
    }
    /// Outward complete normalized-head lower endpoint, in metres.
    pub fn reference_head_lower_m(&self) -> f64 {
        self.reference_head_lower_m
    }
    /// Outward complete normalized-head upper endpoint, in metres.
    pub fn reference_head_upper_m(&self) -> f64 {
        self.reference_head_upper_m
    }
    /// Original reference-speed interpolated discharge magnitude.
    pub fn reference_discharge_m3_per_s(&self) -> f64 {
        self.reference_discharge_m3_per_s
    }
    /// Reference/zero/power-of-two certificate removes division rounding only, retaining head-low uncertainty.
    pub fn division_exact(&self) -> bool {
        self.division_exact
    }
}
