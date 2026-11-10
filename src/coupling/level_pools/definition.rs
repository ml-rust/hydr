use crate::storage::{LevelPoolConfig, LevelPoolElement, LevelPoolSourcePolicy, LevelPoolState};
use crate::structures::{StructureElement, StructureOperation};

/// Physical meaning of a pool's zero-volume curve limit.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPoolSupport {
    /// Exact pool identity.
    pub pool_id: String,
    /// Physical zero-volume stage limit, matching the first curve knot, in metres.
    pub zero_volume_stage_m: f64,
    /// Evidence supporting this limit for the connected level pool.
    pub zero_volume_limit_assertion: String,
}

/// Known local bed and physical connection to one pool.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPoolAttachment {
    /// Exact pool identity, also used by the structure endpoint.
    pub pool_id: String,
    /// Local bed elevation on the pool's vertical reference, in metres.
    pub bed_elevation_m: f64,
    /// Evidence that this attachment samples the declared pool.
    pub physical_connection_assertion: String,
}

/// One structure's ordered connection to two distinct pools.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPoolConnection {
    /// Exact structure identity.
    pub structure_id: String,
    /// Positive-transfer source endpoint.
    pub endpoint_a: CoupledPoolAttachment,
    /// Positive-transfer receiving endpoint.
    pub endpoint_b: CoupledPoolAttachment,
}

/// Explicit numerical budgets and bounded work for a coupled interval.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolConfig {
    /// Existing individual-pool accounting and inverse-curve checks.
    pub storage: LevelPoolConfig,
    /// Maximum absolute head iteration residual, in metres.
    pub head_absolute_m: f64,
    /// Maximum absolute signed-transfer iteration residual, in m³.
    pub transfer_absolute_m3: f64,
    /// Absolute source-adjusted water-residual budget, in m³.
    pub mass_absolute_m3: f64,
    /// Relative source-adjusted water-residual budget against actual storage and throughput.
    pub mass_relative: f64,
    /// Smallest admitted trial duration, in seconds.
    pub minimum_step_s: f64,
    /// Largest admitted trial duration, in seconds.
    pub maximum_step_s: f64,
    /// Maximum Picard updates per attempted interval.
    pub iteration_limit: usize,
    /// Maximum interval halvings after the initial attempt.
    pub retry_limit: usize,
}

/// Immutable physical setup for connected pools.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolSetup {
    /// Native immutable local-source evidence, absent when local sources are disabled.
    pub source_policies: Option<Vec<LevelPoolSourcePolicy>>,
    /// Complete prescribed external inflow bindings in immutable boundary order.
    pub external_inflows: Vec<super::CoupledPoolInflowBinding>,
    /// Complete pool curves and storage assertions in numerical order.
    pub pools: Vec<LevelPoolElement>,
    /// One physical zero-volume declaration per ordered pool.
    pub pool_support: Vec<CoupledPoolSupport>,
    /// Existing checked structure definitions in numerical order.
    pub structures: Vec<StructureElement>,
    /// One explicit static operation per ordered structure.
    pub operations: Vec<StructureOperation>,
    /// One ordered pair of attachments per structure.
    pub connections: Vec<CoupledPoolConnection>,
    /// Explicit dimensional budgets and work limits.
    pub config: CoupledLevelPoolConfig,
}

/// Explicit starting time and authoritative pool volumes.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolInitialState {
    /// Finite nonnegative elapsed time in seconds.
    pub time_s: f64,
    /// One initial volume per ordered pool.
    pub states: Vec<LevelPoolState>,
}
