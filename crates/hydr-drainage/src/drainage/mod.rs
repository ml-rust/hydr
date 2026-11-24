//! Pinned EPA SWMM v5.2.4 piped drainage network solver.
//!
//! Exposes canonical SI state (`m`, `m³`, `m³/s`, `s`), transactional step rollback,
//! complete hidden-state snapshots, and explicit surcharge-method and control-authority
//! declarations.

mod capabilities;
mod definition;
mod engine;
mod error;
mod protocol;
mod state;

pub use capabilities::swmm_drainage_capabilities;
pub use definition::{SwmmControlAuthority, SwmmDrainageDefinition, SwmmSurchargeMethod};
pub use engine::{SwmmDrainage, SwmmDrainageTrial};
pub use error::DrainageError;
pub use state::{
    FT_TO_M, FT3_TO_M3, SWMM_DRAINAGE_CONTRACT_VERSION, SWMM_DRAINAGE_SNAPSHOT_SCHEMA,
    SwmmContinuityLedger, SwmmDrainageSnapshotData, SwmmDrainageState, SwmmInpUnitSystem,
    SwmmLinkCommand, SwmmLinkState, SwmmNetworkSummary, SwmmNodeState,
};
