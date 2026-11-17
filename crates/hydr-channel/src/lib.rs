//! Open-channel reaches for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

use hydr_core::{contract, forcing, numerics, tensor};

pub mod channel;

#[doc(hidden)]
pub mod internal {
    pub use crate::channel::flow::coupling::{
        ChannelCouplingAccounting, ChannelCouplingEndpoints, ChannelCouplingExternalData,
        ChannelCouplingForcing, ChannelCouplingTrial, ChannelReservoirOperands,
    };
}
