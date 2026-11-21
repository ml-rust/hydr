//! Caller declarations of native runoff attached to the coordinator.

use crate::runoff::{SubcatchmentConfig, SubcatchmentDefinition, SupportPlanInput};

/// Native runoff owned by the coordinator: its support plan, definitions, and outlets.
///
/// The coordinator compiles the plan against the surface domain and open faces. The plan
/// selects the active modelled sources and suppresses rain over their support.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelRunoff {
    /// Complete contributing-support declaration over the surface domain.
    pub support: SupportPlanInput,
    /// Every numerical definition the plan names, replaced ones included, in numerical order.
    pub definitions: Vec<SubcatchmentDefinition>,
    /// Numerical and soil-adjustment controls of the runoff child.
    pub config: SubcatchmentConfig,
    /// Exactly one destination per active modelled outlet, in any order.
    pub outlets: Vec<SurfaceChannelRunoffOutlet>,
}

/// One active outlet and the destination of every potential subarea lane it drains.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelRunoffOutlet {
    /// Exact `outlet_id` of one active modelled contribution.
    pub outlet_id: String,
    /// Receiving coordinator node or nodes.
    pub destination: SurfaceChannelRunoffDestination,
    /// Evidence that the receiving water cannot change this outlet's discharge.
    pub free_outlet: SurfaceChannelFreeOutlet,
}

/// Receiving coordinator nodes of one outlet.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceChannelRunoffDestination {
    /// Active surface cells by global row-major index.
    SurfaceCells {
        /// Nonempty distinct cells with positive weights.
        cells: Vec<SurfaceChannelRunoffShare>,
    },
    /// Cells of one reach by ordered cell index.
    ChannelCells {
        /// Exact setup reach identity.
        reach_id: String,
        /// Nonempty distinct cells with positive weights.
        cells: Vec<SurfaceChannelRunoffShare>,
    },
    /// One coordinator pool.
    Pool {
        /// Exact setup pool identity.
        pool_id: String,
    },
}

/// One receiving cell and its share of every lane of the outlet.
///
/// Weights normalize once at construction. A single target takes every lane volume unchanged.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceChannelRunoffShare {
    /// Global row-major surface cell index, or ordered cell index inside the reach.
    pub index: usize,
    /// Finite positive relative weight.
    pub weight: f64,
}

/// Free-outlet attestation of one outlet.
///
/// The outlet stays free while the receiving stage stays at or below the invert.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelFreeOutlet {
    /// Finite outlet invert in the surface vertical datum, in metres.
    pub invert_elevation_m: f64,
    /// Evidence that receiving water below the invert cannot change the outlet discharge.
    pub evidence: String,
}
