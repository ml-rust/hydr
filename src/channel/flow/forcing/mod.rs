//! Original endpoint and lateral operands without a captured runtime client.

mod binding;
mod native;
mod projection;

pub(super) use native::{
    BoundChannelForcing, BoundLateralInputs, NativeLateralInput, NativeWaterAmount,
};
