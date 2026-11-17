//! Prescribed external volumes and independent boundary ledgers.

mod accounting;
mod context;
mod definition;
mod preparation;
mod projection;
mod state;

pub(super) use accounting::{InflowCumulativeSummary, InflowIntervalSummary};
pub(super) use context::{BoundInflows, InflowContext};
pub use definition::{
    CoupledPoolInflow, CoupledPoolInflowBinding, CoupledPoolInflowLedger, CoupledPoolInflowReceipt,
    CoupledPoolInflowValue,
};
pub(super) use preparation::{InflowStep, PreparedInflows};
pub(super) use state::InflowState;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod native_tests;
