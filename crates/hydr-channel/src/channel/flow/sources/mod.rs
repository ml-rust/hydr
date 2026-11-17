//! Lateral input assembly, donor momentum, and authoritative per-binding counters.

mod application;
mod arithmetic;
mod context;
mod expanded;
mod expanded_momentum;
mod preparation;
mod projection;
mod state;
mod types;

#[cfg(test)]
mod tests;

pub(super) use context::LateralContext;
pub(super) use types::{
    LateralReceipt, LateralState, LateralSummary, LateralTotals, PreparedLateral, SourceApplication,
};
