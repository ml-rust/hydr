//! Exact F64 metadata codes shared by kernels and explicit output projection.

pub(crate) const ORIFICE: f64 = 1.0;
pub(crate) const FREE_WEIR: f64 = 2.0;
pub(crate) const SUBMERGED_WEIR: f64 = 3.0;
pub(crate) const BELOW_CREST: f64 = 4.0;
pub(crate) const EQUAL_HEADS: f64 = 5.0;
pub(crate) const PHYSICAL_ZERO: f64 = 0.0;
pub(crate) const ACTIVE: f64 = 1.0;
pub(crate) const BLOCKED: f64 = 2.0;
pub(crate) const PUMP_CURVE: f64 = 6.0;
pub(crate) const PUMP_ISOLATED: f64 = 7.0;
pub(crate) const PUMP_RUNNING: f64 = 3.0;
pub(crate) const PUMP_SHUTOFF: f64 = 4.0;
pub(crate) const PUMP_STOPPED: f64 = 5.0;
pub(crate) const GATE_CLOSED: f64 = 8.0;
pub(crate) const GATE_BELOW_SILL: f64 = 9.0;
pub(crate) const FREE_GATE: f64 = 10.0;
pub(crate) const MIXED_GATE: f64 = 11.0;
pub(crate) const SUBMERGED_GATE: f64 = 12.0;
