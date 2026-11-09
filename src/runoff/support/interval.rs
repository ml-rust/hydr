//! Exact immutable admitted packets and explicit rain suppression evidence.

use super::definition::CompiledPlan;
use crate::runoff::SubcatchmentInput;
use crate::solver::{BoundaryForcing, ForcingInterval};
use std::sync::Arc;

/// Complete physical input packet for one exact original interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportIntervalInput {
    /// Requested elapsed start in seconds.
    pub start: f64,
    /// Requested elapsed end in seconds.
    pub end: f64,
    /// Prepared surface observations over exactly the same interval.
    pub surface: SurfaceObservationInput,
    /// Exactly one request per active modelled contributor.
    pub subcatchments: Vec<SubcatchmentContributionInput>,
    /// Exactly one physical value per declared boundary binding.
    pub boundaries: Vec<BoundBoundaryInput>,
}

/// Prepared observations, including rain retained for suppressed support.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceObservationInput {
    /// Exact bound input identity.
    pub input_identity: String,
    /// Original elapsed start in seconds.
    pub start: f64,
    /// Original elapsed end in seconds.
    pub end: f64,
    /// Row-major observed rain in m/s.
    pub rain: Vec<f64>,
    /// Row-major potential evaporation in m/s, passed unchanged.
    pub potential_evaporation: Vec<f64>,
    /// Row-major withdrawal in m/s, passed unchanged.
    pub withdrawal: Vec<f64>,
}

/// One active modelled input without implicit fixed-volume rescaling.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentContributionInput {
    /// Exact active modelled source identity.
    pub contribution_id: String,
    /// Exact immutable input identity.
    pub input_identity: String,
    /// Original elapsed start in seconds.
    pub start: f64,
    /// Original elapsed end in seconds.
    pub end: f64,
    /// Original numerical forcing, including fixed destination volumes.
    pub input: SubcatchmentInput,
}

/// Complete physical data for one immutable boundary identity.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundBoundaryInput {
    /// Exact boundary binding identity.
    pub boundary_id: String,
    /// Exact expected input identity.
    pub input_identity: String,
    /// Original elapsed start in seconds.
    pub start: f64,
    /// Original elapsed end in seconds.
    pub end: f64,
    /// Uncomposed physical value passed unchanged to the numerical consumer.
    pub value: BoundaryForcing,
}

/// Immutable evidence for an active cell's explicitly suppressed rain.
#[derive(Clone, Debug, PartialEq)]
pub struct RainSuppressionReceipt {
    /// Row-major active cell index.
    pub cell: usize,
    /// Complete declared support atom identities.
    pub atoms: Vec<String>,
    /// Observed prepared rainfall rate in m/s.
    pub observed_rain_m_per_s: f64,
    /// Explicit injected zero rainfall rate in m/s.
    pub injected_rain_m_per_s: f64,
    /// Selected primary source owning this complete footprint.
    pub selected_contribution_id: String,
}

/// Complete checked packet with a private immutable plan-instance token.
#[derive(Clone, Debug)]
pub struct AdmittedInterval {
    pub(super) plan: Arc<CompiledPlan>,
    pub(super) forcing: ForcingInterval,
    pub(super) active_inputs: Vec<SubcatchmentInput>,
    pub(super) suppression: Vec<RainSuppressionReceipt>,
}
impl AdmittedInterval {
    /// Exact elapsed start in seconds.
    pub fn start(&self) -> f64 {
        self.forcing.start
    }
    /// Exact elapsed end in seconds.
    pub fn end(&self) -> f64 {
        self.forcing.end
    }
    /// Immutable compiled forcing, with explicit suppressed rain zeroes.
    pub fn forcing(&self) -> &ForcingInterval {
        &self.forcing
    }
    /// Immutable actual modelled inputs in selected original definition order.
    pub fn subcatchment_inputs(&self) -> &[SubcatchmentInput] {
        &self.active_inputs
    }
    /// Immutable observed/injected rain evidence for suppressed active cells.
    pub fn suppression_receipts(&self) -> &[RainSuppressionReceipt] {
        &self.suppression
    }
}
