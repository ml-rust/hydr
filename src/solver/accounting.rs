use super::FaceTransfer;
use crate::contract::{BoundaryFace, EngineError};
pub use crate::soil::SoilSourceReceipt;

/// Cumulative component water accounting from the original initial state, in m³.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterLedger {
    /// Original represented storage.
    pub initial_storage: f64,
    /// Current represented storage.
    pub storage: f64,
    /// Integrated rain input.
    pub rain: f64,
    /// Integrated prescribed land losses.
    pub withdrawal: f64,
    /// Lifetime native-model infiltration removed from surface water.
    pub infiltration: f64,
    /// Lifetime availability-limited evaporation.
    pub evaporation: f64,
    /// Integrated inward port water.
    pub boundary_in: f64,
    /// Integrated outward port water.
    pub boundary_out: f64,
    /// Signed reported numerical water adjustments.
    pub water_adjustment: f64,
    /// Conservative cumulative absolute adjustment budget consumption.
    pub water_adjustment_abs_bound: f64,
    /// Signed cumulative x/y momentum removed by film resets, in m⁴/s.
    pub momentum_removed: [f64; 2],
    /// Signed cumulative storage residual, in m³.
    pub residual: f64,
}

impl WaterLedger {
    pub(super) fn new(storage: f64) -> Self {
        Self {
            initial_storage: storage,
            storage,
            rain: 0.0,
            withdrawal: 0.0,
            infiltration: 0.0,
            evaporation: 0.0,
            boundary_in: 0.0,
            boundary_out: 0.0,
            water_adjustment: 0.0,
            water_adjustment_abs_bound: 0.0,
            momentum_removed: [0.0; 2],
            residual: 0.0,
        }
    }

    pub(super) fn accepted(&self, step: &StepDiagnostics) -> Result<Self, EngineError> {
        let mut result = self.clone();
        result.storage = step.storage_after;
        result.rain += step.rain_volume;
        result.withdrawal += step.withdrawal_volume;
        result.infiltration += step.infiltration_volume;
        result.evaporation += step.evaporation_volume;
        result.water_adjustment += step.water_adjustment_volume;
        result.water_adjustment_abs_bound = crate::soil::admission::add_adjustment_charge(
            self.water_adjustment_abs_bound,
            step.water_adjustment_abs_charge,
        )
        .map_err(crate::tensor::land::static_error)?;
        result.boundary_in += step.boundary_in;
        result.boundary_out += step.boundary_out;
        for axis in 0..2 {
            result.momentum_removed[axis] += step.momentum_removed[axis];
        }
        result.residual = (result.storage - result.initial_storage)
            - (result.rain + result.boundary_in
                - result.boundary_out
                - result.withdrawal
                - result.infiltration
                - result.evaporation
                + result.water_adjustment);
        if [
            result.rain,
            result.withdrawal,
            result.infiltration,
            result.evaporation,
            result.water_adjustment,
            result.water_adjustment_abs_bound,
            result.boundary_in,
            result.boundary_out,
            result.residual,
            result.momentum_removed[0],
            result.momentum_removed[1],
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(EngineError::rejected(
                "ledger",
                "cumulative accounting overflow: reduce the horizon or input scale",
            ));
        }
        Ok(result)
    }
}

/// Physical state and compatibility residuals at an accepted open boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryTraceDiagnostics {
    /// Exterior face using the trace.
    pub face: BoundaryFace,
    /// Trace depth in metres.
    pub depth: f64,
    /// Outward normal trace velocity in m/s.
    pub normal_velocity: f64,
    /// Trace velocity along the positive other coordinate in m/s.
    pub tangential_velocity: f64,
    /// Achieved signed outward water rate in m³/s.
    pub outward_water_rate: f64,
    /// Wave-curve discharge minus requested discharge in m²/s. Stage traces record zero.
    pub discharge_residual_per_width: f64,
    /// Corrected normal velocity minus compatible wave-curve velocity in m/s.
    pub characteristic_velocity_residual: f64,
    /// Conservative normal signal-speed bound in m/s.
    pub signal_speed: f64,
}

/// Post-source interior state admitted by an accepted transmissive boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct TransmissiveDiagnostics {
    /// Active exterior face receiving this copied state.
    pub face: BoundaryFace,
    /// Represented interior depth in metres.
    pub depth: f64,
    /// Outward-normal discharge per width in m²/s.
    pub outward_discharge_per_width: f64,
    /// Discharge along the positive other global coordinate in m²/s.
    pub tangential_discharge_per_width: f64,
}

/// Numerical and physical accounting for exactly one accepted interval.
#[derive(Clone, Debug, PartialEq)]
pub struct StepDiagnostics {
    /// Accepted starting time in seconds.
    pub start: f64,
    /// Accepted ending time in seconds.
    pub end: f64,
    /// Accepted elapsed interval in seconds.
    pub dt: f64,
    /// Number of discarded and recalculated trial intervals.
    pub retries: usize,
    /// CFL upper interval bound in seconds. Infinity denotes no wave restriction.
    pub cfl_limit: f64,
    /// All-outgoing donor depletion bound in seconds. Infinity denotes no outflow.
    pub depletion_limit: f64,
    /// Storage before the step in m³.
    pub storage_before: f64,
    /// Storage after the step in m³.
    pub storage_after: f64,
    /// Integrated rainfall in m³.
    pub rain_volume: f64,
    /// Integrated prescribed land loss in m³.
    pub withdrawal_volume: f64,
    /// Removed native-model infiltration in m³.
    pub infiltration_volume: f64,
    /// Availability-limited evaporation in m³.
    pub evaporation_volume: f64,
    /// Reported numerical water created by soil reconciliation in m³.
    pub soil_water_adjustment_volume: f64,
    /// Total signed reported numerical water adjustment in m³.
    pub water_adjustment_volume: f64,
    /// Conservative absolute adjustment budget charge in m³.
    pub water_adjustment_abs_charge: f64,
    /// Inward boundary water in m³.
    pub boundary_in: f64,
    /// Outward boundary water in m³.
    pub boundary_out: f64,
    /// Signed water residual in m³.
    pub residual: f64,
    /// Residual divided by max(storage, external volumes, 1 m³).
    pub normalized_residual: f64,
    /// Maximum absolute per-cell arithmetic water residual in m³.
    pub max_local_residual: f64,
    /// Signed x/y momentum removed by film resets, in m⁴/s.
    pub momentum_removed: [f64; 2],
    /// Numerical flux support over this interval. No instantaneous endpoint flux is implied.
    pub faces: Vec<FaceTransfer>,
    /// Physical Stage/Discharge traces used over this accepted interval.
    pub boundary_traces: Vec<BoundaryTraceDiagnostics>,
    /// Post-source trial witnesses for accepted transmissive boundaries.
    pub transmissive_states: Vec<TransmissiveDiagnostics>,
    /// Ordered full-grid accepted source witnesses, including inactive zero bookkeeping.
    pub sources: Vec<SourceReceipt>,
}

/// Full-grid source algebra at an accepted interval, with all depths in metres.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceReceipt {
    /// Global row-major cell identity.
    pub cell: usize,
    /// Interval-start ponded water.
    pub ponded_depth_m: f64,
    /// Added rain depth.
    pub rain_depth_m: f64,
    /// Original soil-model infiltration, without an availability cap.
    pub infiltration_depth_m: f64,
    /// Availability-limited evaporation after infiltration.
    pub evaporation_depth_m: f64,
    /// Prescribed withdrawal after evaporation.
    pub withdrawal_depth_m: f64,
    /// Explicit soil numerical correction, distinct from rainfall.
    pub water_adjustment_depth_m: f64,
    /// Native model and bound evidence for pervious cells only.
    pub soil: Option<SoilSourceReceipt>,
}
