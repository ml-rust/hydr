//! Native runoff attached to the coordinator: declarations, admission, and the runoff child.
//!
//! The coordinator compiles one support plan from the runoff setup against the surface domain
//! and open faces. The plan selects the active modelled sources and suppresses rain over their
//! support. Each active outlet has one destination: surface cells, cells of one reach, or one
//! pool. Every potential subarea lane of the outlet delivers to each target of that destination.
//!
//! A channel target receives one synthesized inflow binding per lane and cell. A pool target
//! receives one synthesized inflow binding per lane. Both use the reserved prefix
//! `runoff-attachment:`.
//!
//! Construction builds the runoff child at the common epoch. A trial on a setup with runoff
//! admits its forcing, then rejects, since stepping is not attached yet.

mod admission;
mod bindings;
mod child;
mod definition;
mod forcing;
mod targets;
mod writers;

#[cfg(test)]
mod scenario;

pub(super) use admission::{CheckedRunoff, check_runoff};
pub(super) use bindings::{RUNOFF_ATTACHMENT_BINDING_PREFIX, append_lateral_bindings};
pub(super) use child::build_child;
pub use definition::{
    SurfaceChannelFreeOutlet, SurfaceChannelRunoff, SurfaceChannelRunoffDestination,
    SurfaceChannelRunoffOutlet, SurfaceChannelRunoffShare,
};
pub(super) use forcing::{admit_forcing, check_schedule, stepping_not_attached};
pub(super) use writers::check_caller_pool_inflows;
