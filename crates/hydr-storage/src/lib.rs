//! Level-pool storage for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

use hydr_core::{numerics, tensor};

pub mod storage;

#[doc(hidden)]
pub mod internal {
    pub use crate::storage::coupling::{LevelPoolNativeAccounting, LevelPoolNativeInterval};
    pub use crate::storage::coupling_cuts::{
        LevelPoolCut, LevelPoolCutAvailability, LevelPoolCutPlan,
    };
    pub use crate::storage::prepared_interval::{LevelPoolSourcePlan, PreparedLevelPoolInterval};
    pub use crate::storage::source_summary::LevelPoolIntervalSources;
}
