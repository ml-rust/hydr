//! Prescribed external volumes and independent boundary ledgers.

mod accounting;
mod context;
mod definition;
#[cfg(test)]
mod fixture;
mod preparation;
mod projection;
mod state;

pub(crate) use accounting::{InflowCumulativeSummary, InflowIntervalSummary};
pub(crate) use context::{BoundInflows, InflowContext};
pub use definition::{
    CoupledPoolInflow, CoupledPoolInflowBinding, CoupledPoolInflowLedger, CoupledPoolInflowReceipt,
    CoupledPoolInflowValue,
};
pub(crate) use preparation::{InflowStep, PreparedInflows};
pub(crate) use state::InflowState;

#[cfg(test)]
mod native_tests;
