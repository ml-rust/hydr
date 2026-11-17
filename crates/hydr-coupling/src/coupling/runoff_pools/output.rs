use super::{RunoffPoolForcingInterval, RunoffPoolOutletBinding};

/// Derived child accounting. Internal outlet traffic never enlarges the physical tolerance scale.
#[derive(Clone, Debug, PartialEq)]
pub struct RunoffPoolWaterBalance {
    /// Original interval storage, or original lifetime storage, in cubic metres.
    pub storage_before_m3: f64,
    /// Complete current storage in both domains, in cubic metres.
    pub storage_after_m3: f64,
    /// Actual native and pool rainfall, in cubic metres.
    pub rain_volume_m3: f64,
    /// Actual native infiltration, in cubic metres.
    pub infiltration_volume_m3: f64,
    /// Actual capped native and pool evaporation, in cubic metres.
    pub evaporation_volume_m3: f64,
    /// Actual supplied pool bottom loss, in cubic metres.
    pub bottom_loss_volume_m3: f64,
    /// Explicit signed native soil water correction, in cubic metres.
    pub soil_water_adjustment_volume_m3: f64,
    /// Paired internal runoff export, excluded from external physical terms.
    pub internal_outlet_volume_m3: f64,
    /// Independent physical water residual, in cubic metres.
    pub water_residual_m3: f64,
    /// Conservative child and scalar arithmetic envelope, in cubic metres.
    pub roundoff_bound_m3: f64,
    /// Largest storage or physical rain/loss term. Transfers and numerical soil correction are excluded.
    pub physical_scale_m3: f64,
}
/// One accepted common interval with original forcing and complete immutable pairing identity.
#[derive(Clone, Debug, PartialEq)]
pub struct RunoffPoolStepDiagnostics {
    /// Common accepted start, in seconds.
    pub start_s: f64,
    /// Actual common accepted end, in seconds.
    pub end_s: f64,
    /// Original requested end before common duration limits and recalculation.
    pub requested_end_s: f64,
    /// Complete common recalculations before acceptance.
    pub retries: usize,
    /// Original covered meteorology and local source operands.
    pub forcing: RunoffPoolForcingInterval,
    /// Exact immutable outlet mapping in declared order.
    pub outlets: Vec<RunoffPoolOutletBinding>,
    /// Derived original interval physical accounting.
    pub balance: RunoffPoolWaterBalance,
}
