use crate::coupling::CoupledLevelPoolSetup;
use crate::runoff::{Subarea, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentState};
use crate::storage::{LevelPoolSources, LevelPoolState};

/// Free native outlet routed once to one declared receiving boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunoffPoolOutletBinding {
    /// Exact native subcatchment identity.
    pub subcatchment_id: String,
    /// Positive-area physical outlet role.
    pub subarea: Subarea,
    /// Exact receiving boundary identity from the pool setup.
    pub boundary_id: String,
    /// Evidence that downstream water cannot change this outlet's discharge.
    pub free_outlet_assertion: String,
    /// Evidence excluding represented pool storage, local pool rain, and duplicate runoff support.
    pub nonoverlap_assertion: String,
}
/// Common numerical limits, independent from each child's admission budgets.
#[derive(Clone, Debug, PartialEq)]
pub struct RunoffPoolConfig {
    /// Smallest common duration, in seconds.
    pub minimum_step_s: f64,
    /// Largest common duration, in seconds.
    pub maximum_step_s: f64,
    /// Maximum complete recalculations after the initial attempt.
    pub retry_limit: usize,
    /// Absolute parent physical water budget, in cubic metres.
    pub mass_absolute_m3: f64,
    /// Relative budget against external physical terms and represented storage.
    pub mass_relative: f64,
}
/// Explicit child definitions and complete free-outlet ownership, without external runon or disposal.
#[derive(Clone, Debug, PartialEq)]
pub struct RunoffPoolSetup {
    /// Immutable native geometry, soil, and routing in numerical order.
    pub subcatchments: Vec<SubcatchmentDefinition>,
    /// Native runoff and separate soil-adjustment controls.
    pub runoff_config: SubcatchmentConfig,
    /// Complete receiving pool definitions and boundary evidence.
    pub pools: CoupledLevelPoolSetup,
    /// Every potential native outlet and every receiving boundary, paired once.
    pub outlets: Vec<RunoffPoolOutletBinding>,
    /// Common duration, retry, and physical accounting budgets.
    pub config: RunoffPoolConfig,
}
/// Complete antecedent state at one immutable common origin.
#[derive(Clone, Debug, PartialEq)]
pub struct RunoffPoolInitialState {
    /// Common nonnegative original time, in seconds.
    pub time_s: f64,
    /// Native surface depths and complete soil memory.
    pub subcatchments: Vec<SubcatchmentState>,
    /// Authoritative receiving-pool storage.
    pub pools: Vec<LevelPoolState>,
}
/// Complete original meteorology, without an external runon field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunoffPoolMeteorology {
    /// Nonnegative rainfall rate over represented native support, in metres per second.
    pub rain_rate_m_per_s: f64,
    /// Nonnegative potential evaporation rate, in metres per second.
    pub potential_evaporation_m_per_s: f64,
}
/// Complete constant meteorology and optional local pool sources over reusable time coverage.
#[derive(Clone, Debug, PartialEq)]
pub struct RunoffPoolForcingInterval {
    /// Earliest covered time, in seconds.
    pub start_s: f64,
    /// Latest covered time, including the original requested endpoint.
    pub end_s: f64,
    /// Complete meteorology in immutable subcatchment order.
    pub subcatchments: Vec<RunoffPoolMeteorology>,
    /// Complete native local sources, present exactly when pool source policies exist.
    pub pool_sources: Option<Vec<LevelPoolSources>>,
}
