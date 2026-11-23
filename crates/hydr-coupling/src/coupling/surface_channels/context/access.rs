//! Read access to the static coordinator operands, zero counters, and the transfer error map.

use super::culverts::CulvertStops;
use super::{Context, NodeEnd};
use crate::channel::ChannelEnd;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::pool_ends::PoolEndCounters;
use crate::coupling::surface_channels::ports::{PortCounters, end_slot};
use crate::coupling::surface_channels::runoff::RunoffCounters;
use crate::coupling::surface_channels::state::LinkCounters;
use crate::coupling::surface_channels::structures::LinkLaw;
use crate::numerics::water_transfers::WaterTransferError;
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
            pool_ends: PoolEndCounters::zero(self.pool_end_count()),
            runoff: self
                .runoff
                .as_ref()
                .map(|topology| RunoffCounters::zero(&self.client, topology))
                .transpose()?,
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

    /// Pool-end count `E`.
    pub fn pool_end_count(&self) -> usize {
        self.pool_ends.as_ref().map_or(0, |ends| ends.ends.len())
    }

    /// Signed volume count of the iterate, `L + E`.
    pub fn transfer_count(&self) -> usize {
        self.link_count + self.pool_end_count()
    }

    /// Identity of the pool end at `end` of setup reach `reach`, if any.
    pub fn pool_end_at(&self, reach: usize, end: ChannelEnd) -> Option<&str> {
        let ends = self.pool_ends.as_ref()?;
        let position = ends.reach_ends.get(reach)?[end_slot(end)]?;
        ends.ends.get(position).map(|end| end.id.as_str())
    }

    /// True when a pool end owns either end of setup reach `reach`.
    pub fn is_pool_ended(&self, reach: usize) -> bool {
        self.pool_ends.as_ref().is_some_and(|ends| ends.owns(reach))
    }

    /// Invert stops for `laws` evaluated against this context's endpoint plan areas.
    pub(in super::super) fn build_culvert_stops(
        &self,
        laws: &[LinkLaw],
    ) -> Result<Option<CulvertStops<R>>, SurfaceChannelError> {
        let side_area = |link: usize, side: usize, stage: f64| -> f64 {
            match self.link_ends[link][side] {
                NodeEnd::Surface(lane) => self.lane_area_m2.get(lane).copied().unwrap_or(0.0),
                NodeEnd::Channel(slot) => self
                    .slot_plans
                    .get(slot)
                    .map_or(0.0, |plan| plan.area(stage)),
                NodeEnd::Pool(pool) => self
                    .pools
                    .as_ref()
                    .and_then(|parts| parts.plans.get(pool))
                    .map_or(0.0, |plan| plan.smallest(stage, plan.top())),
            }
        };
        CulvertStops::new(laws, side_area, self.client.device())
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
