//! Read access to the static coordinator operands, zero counters, and the transfer error map.

use super::Context;
use crate::channel::ChannelEnd;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::ports::PortCounters;
use crate::coupling::surface_channels::state::LinkCounters;
use crate::numerics::water_transfers::WaterTransferError;
use hydr_structures::internal::CouplingStructures;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R> {
    /// Zero lifetime link counters.
    pub fn zero_counters(&self) -> Result<LinkCounters<R>, SurfaceChannelError> {
        let zero = Tensor::<R>::zeros(&[self.link_count], DType::F64, self.client.device())?;
        Ok(LinkCounters {
            to_channel_m3: zero.clone(),
            to_surface_m3: zero.clone(),
            roundoff_bound_m3: zero,
            ports: self.zero_port_counters(),
        })
    }

    /// Rank-zero F64 `value` on the configured device.
    pub fn scalar(&self, value: f64) -> Result<Tensor<R>, SurfaceChannelError> {
        Ok(Tensor::<R>::full_scalar(
            &[],
            DType::F64,
            value,
            self.client.device(),
        )?)
    }

    /// Donor group count `S + C + P`.
    pub fn group_count(&self) -> usize {
        self.surface_count + self.slot_count + self.pool_count
    }

    /// Identity of the port at `end` of setup reach `reach`, if any.
    pub fn port_at(&self, reach: usize, end: ChannelEnd) -> Option<&str> {
        self.ports.as_ref()?.port_at(reach, end)
    }

    /// Port identities in ascending order. Empty without ports.
    pub fn port_ids(&self) -> &[String] {
        self.ports
            .as_ref()
            .map_or(&[][..], |ports| ports.ids.as_slice())
    }

    /// Zero lifetime port counters, one lane per port.
    pub fn zero_port_counters(&self) -> PortCounters {
        PortCounters::zero(self.port_ids().len())
    }

    /// True when a port binds either end of setup reach `reach`.
    pub fn is_ported(&self, reach: usize) -> bool {
        self.ports.as_ref().is_some_and(|ports| {
            ports
                .reach_ports
                .get(reach)
                .is_some_and(|ends| ends.iter().any(Option::is_some))
        })
    }

    /// Laws of every link. A coordinator with ports only has none.
    pub fn link_structures(&self) -> Result<&CouplingStructures<R>, SurfaceChannelError> {
        self.structures.as_ref().ok_or_else(|| {
            SurfaceChannelError::Numerical("a coordinator without links has no weir laws".into())
        })
    }
}

/// Maps a transfer-tree rejection onto the coordinator taxonomy.
pub(in super::super) fn transfer_error(error: WaterTransferError) -> SurfaceChannelError {
    match error {
        WaterTransferError::Input(message) => SurfaceChannelError::Input(message),
        WaterTransferError::Numerical(message) => SurfaceChannelError::Numerical(message),
        WaterTransferError::Runtime(error) => SurfaceChannelError::Runtime(error),
    }
}
