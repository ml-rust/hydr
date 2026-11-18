//! Pool-end support at the accepted start and at the candidate end, and exact end pairing.

use crate::channel::ChannelReach;
use crate::coupling::links::{LinkError, pool_ends};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// One where a pool end's pool holds water above its end cell bed, `[E]`, for pool `volumes`.
    pub fn end_support(&self, volumes: &Tensor<R>) -> Result<Tensor<R>, SurfaceChannelError> {
        let ends = self.end_topology()?;
        let cuts = self
            .pools
            .as_ref()
            .and_then(|pools| pools.cuts.as_ref())
            .ok_or_else(|| {
                SurfaceChannelError::Numerical("pool ends lack their attachment cuts".into())
            })?;
        pool_ends::attachment_wet(&self.client, volumes, &cuts.upper, &ends.pools, &ends.cuts)
            .map_err(LinkError::surface_channel)
    }

    /// Wet support of every pool end at the accepted start, `[E]`, or `None` without pool ends.
    ///
    /// A pool end whose pool holds no water above the end cell bed, or whose end cell is dry,
    /// rejects with `pool-end-regime`. No shorter interval changes the accepted start, so the
    /// rejection is final.
    pub fn start_end_support(
        &self,
        reaches: &[ChannelReach<R>],
        pools: Option<&LevelPoolBatch<R>>,
        reach_ids: &[&str],
    ) -> Result<Option<Tensor<R>>, SurfaceChannelError> {
        let Some(topology) = &self.pool_ends else {
            return Ok(None);
        };
        let pools = pools.ok_or_else(|| {
            SurfaceChannelError::Numerical("pool ends lack their accepted pools".into())
        })?;
        let support = self.end_support(pools.volumes_tensor())?;
        let flags = support.try_to_vec::<f64>()?;
        for (end, flag) in topology.ends.iter().zip(&flags) {
            let reach_id = reach_ids[end.reach_index];
            let (_, wet) = reaches[end.reach_index]
                .coupling_cell_stages(&[end.cell])
                .map_err(|source| SurfaceChannelError::Channel {
                    reach_id: reach_id.to_string(),
                    source,
                })?;
            let cell_wet = wet.try_to_vec::<f64>()?.first().is_some_and(|&w| w > 0.0);
            if *flag <= 0.0 || !cell_wet {
                return Err(SurfaceChannelError::input(format!(
                    "pool-end-regime: pool end={:?} reach={reach_id:?} end={:?} is dry at the accepted start (pool wet above bed={}: {}, end cell wet: {cell_wet}): a pool end needs wet subcritical water on both sides; wet the pool and the end cell, or attach the reach end through a structure",
                    end.id,
                    end.end,
                    end.bed_m,
                    *flag > 0.0
                )));
            }
        }
        Ok(Some(support))
    }

    /// Rejects an active pool end without wet support at the candidate end of the interval.
    ///
    /// `start` and `end` hold the support flags at both interval ends and `volumes` the accepted
    /// pool-end volumes, each `[E]`. The rejection is retryable.
    pub fn check_end_support(
        &self,
        start: &Tensor<R>,
        end: &Tensor<R>,
        volumes: &Tensor<R>,
    ) -> Result<(), SurfaceChannelError> {
        pool_ends::check_support(&self.client, start, end, volumes)
            .map_err(LinkError::surface_channel)
    }

    /// Rejects a pool-end volume that differs in any bit from its accepted reach end volume.
    ///
    /// `volumes` holds every pool-end volume, positive into the pool, and `reaches` the accepted
    /// candidate reaches in setup order.
    pub fn check_end_pairing(
        &self,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        volumes: &[f64],
    ) -> Result<(), SurfaceChannelError> {
        let Some(topology) = &self.pool_ends else {
            return Ok(());
        };
        for (end, volume) in topology.ends.iter().zip(volumes) {
            let reach = &reaches[end.reach_index];
            let endpoints =
                reach
                    .coupling_endpoints()
                    .map_err(|source| SurfaceChannelError::Channel {
                        reach_id: reach_ids[end.reach_index].to_string(),
                        source,
                    })?;
            let outward = endpoints.outward_volume_m3.try_to_vec::<f64>()?;
            let paired = outward
                .get(end.side)
                .is_some_and(|v| v.to_bits() == volume.to_bits());
            if !paired {
                return Err(SurfaceChannelError::Numerical(format!(
                    "pool-end pairing: pool end={:?} volume={volume} differs from its reach end volume={:?}: inspect the pool-end completion",
                    end.id,
                    outward.get(end.side)
                )));
            }
        }
        if volumes.len() != topology.ends.len() {
            return Err(SurfaceChannelError::Numerical(
                "pool-end pairing lacks a pool-end volume".into(),
            ));
        }
        Ok(())
    }

    /// Pool-end operands. A coordinator without pool ends rejects.
    pub fn end_topology(&self) -> Result<&super::PoolEndTopology<R>, SurfaceChannelError> {
        self.pool_ends.as_ref().ok_or_else(|| {
            SurfaceChannelError::Numerical(
                "a coordinator without pool ends has no pool-end operands".into(),
            )
        })
    }
}
