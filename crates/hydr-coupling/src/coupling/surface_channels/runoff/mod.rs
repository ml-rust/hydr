//! Native runoff attached to the coordinator: declarations, admission, the runoff child, and
//! its delivery into reaches and pools.
//!
//! The coordinator compiles one support plan from the runoff setup against the surface domain
//! and open faces. The plan selects the active modelled sources and suppresses rain over their
//! support. Each active outlet has one destination: surface cells, cells of one reach, or one
//! pool. Every potential subarea lane of the outlet delivers to each target of that destination.
//!
//! A channel target receives one synthesized inflow binding per lane and cell. A pool target
//! receives one synthesized inflow binding per lane. Both use the reserved prefix
//! `runoff-attachment:`. A surface target receives its lanes through the surface exchange
//! lanes: the link cells, then every runoff-only cell.
//!
//! Construction builds the runoff child at the common epoch. Each attempt trials the child from
//! its accepted state to the attempt end, after the surface step limits and before the pools
//! and the link fixed point. Runoff volumes stay fixed across the fixed point. Reaches take
//! them as interval-volume inflows with zero velocity. Pools take them as native interval
//! volumes beside the caller inflows. Surface cells take them as exchange inflow after
//! transport with zero velocity.

mod accounting;
mod admission;
mod attempt;
mod bindings;
mod child;
mod counters;
mod definition;
mod delivery;
mod forcing;
mod report;
mod restart;
mod surface;
mod tailwater;
mod targets;
mod topology;
mod writers;

#[cfg(test)]
mod basin;
#[cfg(test)]
mod cells;
#[cfg(test)]
mod rain_rows;
#[cfg(test)]
mod scenario;

pub(super) use accounting::RunoffTerms;
pub(super) use admission::{CheckedRunoff, check_runoff};
pub(super) use attempt::{RunoffAttempt, SurfaceRunoffStep};
pub(super) use bindings::{
    RUNOFF_ATTACHMENT_BINDING_PREFIX, append_lateral_bindings, runoff_pool_bindings,
};
pub(super) use child::build_child;
pub(super) use counters::RunoffCounters;
pub use definition::{
    SurfaceChannelFreeOutlet, SurfaceChannelRunoff, SurfaceChannelRunoffDestination,
    SurfaceChannelRunoffOutlet, SurfaceChannelRunoffShare,
};
pub(super) use delivery::{
    RunoffStart, pool_receipts, runoff_delivery, runoff_end, runoff_start, with_runoff,
};
pub(super) use forcing::check_schedule;
pub(super) use report::{
    RunoffStep, counter_data as runoff_counter_data, import_counters as import_runoff_counters,
};
pub(super) use restart::restored_child;
pub(super) use topology::RunoffTopology;
pub(super) use writers::check_caller_pool_inflows;
