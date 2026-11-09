//! Standalone volume-authoritative closed level-pool storage without outlet or atmospheric laws.

mod accounting;
mod batch;
mod curve;
pub(crate) mod definition;
mod error;
mod projection;
mod receipt;
mod snapshot;

pub use batch::{LevelPoolBatch, LevelPoolTrial};
pub use curve::{LevelPoolCurve, LevelPoolKnot};
pub use definition::{LevelPoolConfig, LevelPoolElement, LevelPoolInput, LevelPoolState};
pub use error::LevelPoolError;
pub use receipt::{
    LevelPoolElementLedger, LevelPoolOutput, LevelPoolReceipt, LevelPoolStepDiagnostics,
    LevelPoolTotals,
};
pub use snapshot::{LevelPoolSnapshot, LevelPoolSnapshotData};
