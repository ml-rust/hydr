use super::{BoundaryFace, EngineError, FaceTransfer};

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
    /// Integrated inward port water.
    pub boundary_in: f64,
    /// Integrated outward port water.
    pub boundary_out: f64,
    /// Numerical water adjustments. This reference rejects negative water and always records zero.
    pub water_adjustment: f64,
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
            boundary_in: 0.0,
            boundary_out: 0.0,
            water_adjustment: 0.0,
            momentum_removed: [0.0; 2],
            residual: 0.0,
        }
    }

    pub(super) fn accepted(&self, step: &StepDiagnostics) -> Result<Self, EngineError> {
        let mut result = self.clone();
        result.storage = step.storage_after;
        result.rain += step.rain_volume;
        result.withdrawal += step.withdrawal_volume;
        result.boundary_in += step.boundary_in;
        result.boundary_out += step.boundary_out;
        for axis in 0..2 {
            result.momentum_removed[axis] += step.momentum_removed[axis];
        }
        result.residual = (result.storage - result.initial_storage)
            - (result.rain + result.boundary_in - result.boundary_out - result.withdrawal);
        if [
            result.rain,
            result.withdrawal,
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
}
