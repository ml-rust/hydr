//! Evaluated gate geometry and independent arithmetic-consistency evidence.

/// Original opening and represented hydraulic geometry for an active gate proposal.
#[derive(Clone, Debug, PartialEq)]
pub struct GateHydraulicOperands {
    pub(super) opening_m: f64,
    pub(super) crown_high_m: f64,
    pub(super) crown_low_m: f64,
    pub(super) crown_overflow: bool,
    pub(super) submerged_height_m: f64,
    pub(super) submerged_height_error_bound_m: f64,
    pub(super) free_height_m: f64,
    pub(super) free_height_error_bound_m: f64,
    pub(super) submerged_head_gap_m: f64,
    pub(super) submerged_head_gap_error_bound_m: f64,
    pub(super) free_lower_head_m: f64,
    pub(super) free_lower_head_error_bound_m: f64,
    pub(super) free_upper_head_m: f64,
    pub(super) free_upper_head_error_bound_m: f64,
    pub(super) free_head_ratio: f64,
    pub(super) free_integral_factor_k: f64,
    pub(super) closure_residual_m: f64,
    pub(super) closure_uncertainty_bound_m: f64,
    pub(super) closure_physical_allowance_m: f64,
    pub(super) log_discharge_magnitude: f64,
}
impl GateHydraulicOperands {
    /// Original supplied relative opening, in metres.
    pub fn opening_m(&self) -> f64 {
        self.opening_m
    }
    /// Exact finite crown expansion, absent when sill plus opening overflows positively.
    pub fn crown_expansion_m(&self) -> Option<(f64, f64)> {
        if self.crown_overflow {
            None
        } else {
            Some((self.crown_high_m, self.crown_low_m))
        }
    }
    /// Represented submerged strip height, exactly zero when the strip is absent.
    pub fn submerged_height_m(&self) -> f64 {
        self.submerged_height_m
    }
    /// Independent absolute error bound for represented submerged height, in metres.
    pub fn submerged_height_error_bound_m(&self) -> f64 {
        self.submerged_height_error_bound_m
    }
    /// Represented free strip height, exactly zero when the strip is absent.
    pub fn free_height_m(&self) -> f64 {
        self.free_height_m
    }
    /// Independent absolute error bound for represented free height, in metres.
    pub fn free_height_error_bound_m(&self) -> f64 {
        self.free_height_error_bound_m
    }
    /// Direct original wet-stage difference, present only for a submerged strip.
    pub fn submerged_head_gap_m(&self) -> Option<f64> {
        (self.submerged_height_m > 0.0).then_some(self.submerged_head_gap_m)
    }
    /// Independent absolute error bound for the submerged pressure gap, in metres.
    pub fn submerged_head_gap_error_bound_m(&self) -> Option<f64> {
        (self.submerged_height_m > 0.0).then_some(self.submerged_head_gap_error_bound_m)
    }
    /// Represented lower free-strip pressure head, present only for a free strip.
    pub fn free_lower_head_m(&self) -> Option<f64> {
        (self.free_height_m > 0.0).then_some(self.free_lower_head_m)
    }
    /// Independent absolute error bound for the lower free pressure head, in metres.
    pub fn free_lower_head_error_bound_m(&self) -> Option<f64> {
        (self.free_height_m > 0.0).then_some(self.free_lower_head_error_bound_m)
    }
    /// Represented upper free-strip pressure head, including exact zero at donor wetting.
    pub fn free_upper_head_m(&self) -> Option<f64> {
        (self.free_height_m > 0.0).then_some(self.free_upper_head_m)
    }
    /// Independent absolute error bound for the upper free pressure head, in metres.
    pub fn free_upper_head_error_bound_m(&self) -> Option<f64> {
        (self.free_height_m > 0.0).then_some(self.free_upper_head_error_bound_m)
    }
    /// Represented upper-to-lower free pressure-head ratio, within the closed range zero to one.
    pub fn free_head_ratio(&self) -> Option<f64> {
        (self.free_height_m > 0.0).then_some(self.free_head_ratio)
    }
    /// Stable free-strip factor `1+r/(1+sqrt(r))`, present only for a free strip.
    pub fn free_integral_factor_k(&self) -> Option<f64> {
        (self.free_height_m > 0.0).then_some(self.free_integral_factor_k)
    }
    /// Signed represented strip-closure discrepancy center, in metres.
    pub fn closure_residual_m(&self) -> f64 {
        self.closure_residual_m
    }
    /// Outward uncertainty around the closure discrepancy center, in metres.
    pub fn closure_uncertainty_bound_m(&self) -> f64 {
        self.closure_uncertainty_bound_m
    }
    /// Independent strip-length and target-error allowance for closure, in metres.
    pub fn closure_physical_allowance_m(&self) -> f64 {
        self.closure_physical_allowance_m
    }
    /// Natural logarithm of the final proposed SI discharge magnitude.
    pub fn log_discharge_magnitude(&self) -> f64 {
        self.log_discharge_magnitude
    }
}
