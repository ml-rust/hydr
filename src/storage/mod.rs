//! Standalone volume-authoritative level-pool storage with optional explicit frozen-area sources and no outlet law.

mod accounting;
mod batch;
mod coupling;
mod coupling_cut_bounds;
mod coupling_cuts;
mod curve;
pub(crate) mod definition;
mod error;
mod prepared_interval;
mod projection;
mod receipt;
mod snapshot;
mod source_policy;
mod source_projection;
mod source_receipt;
mod source_summary;
mod trial;

pub use batch::{LevelPoolBatch, LevelPoolTrial};
pub use curve::{LevelPoolCurve, LevelPoolKnot};
pub use definition::{LevelPoolConfig, LevelPoolElement, LevelPoolInput, LevelPoolState};
pub use error::LevelPoolError;
pub use receipt::{
    LevelPoolElementLedger, LevelPoolOutput, LevelPoolReceipt, LevelPoolStepDiagnostics,
    LevelPoolTotals,
};
pub use snapshot::{LevelPoolSnapshot, LevelPoolSnapshotData};
pub use source_policy::{LevelPoolSourcePolicy, LevelPoolSources};
pub use source_receipt::{
    LevelPoolCounterReceipt, LevelPoolDemandCapReceipt, LevelPoolSourceCounterReceipt,
    LevelPoolSourceLedger, LevelPoolSourceReceipt,
};

pub(crate) use prepared_interval::{LevelPoolSourcePlan, PreparedLevelPoolInterval};
pub(crate) use source_summary::LevelPoolIntervalSources;

pub(crate) use coupling_cuts::{LevelPoolCut, LevelPoolCutPlan};
