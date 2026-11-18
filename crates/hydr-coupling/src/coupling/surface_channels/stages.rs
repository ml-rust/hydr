//! Stages and wet masks of every donor group: surface lanes, channel slots, then pools.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use crate::channel::ChannelReach;
use crate::solver::Solver;
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Stages and wet masks of every group: surface lanes, slots, then pools.
    pub fn node_stages(
        &self,
        surface: &Solver<R>,
        linked: &[&ChannelReach<R>],
        pools: Option<&LevelPoolBatch<R>>,
        reach_ids: &[&str],
    ) -> Result<(Tensor<R>, Tensor<R>), SurfaceChannelError> {
        let (stage, wet) = self.stages(surface, linked, reach_ids)?;
        let Some(pools) = pools else {
            return Ok((stage, wet));
        };
        let (pool_stage, pool_wet) = self.pool_stages(pools)?;
        let c = &self.client;
        Ok((
            c.cat(&[&stage, &pool_stage], 0)?,
            c.cat(&[&wet, &pool_wet], 0)?,
        ))
    }

    /// Accepted stages and wet masks: surface lanes, then slots.
    pub fn stages(
        &self,
        surface: &Solver<R>,
        linked: &[&ChannelReach<R>],
        reach_ids: &[&str],
    ) -> Result<(Tensor<R>, Tensor<R>), SurfaceChannelError> {
        let mut stages = Vec::new();
        let mut wets = Vec::new();
        if let Some(links) = &self.links {
            let (stage, wet) = surface.coupling_cell_stages(links)?;
            stages.push(stage);
            wets.push(wet);
        }
        for (entry, reach) in self.linked.iter().zip(linked) {
            let (stage, wet) = reach.coupling_cell_stages(&entry.cells).map_err(|source| {
                SurfaceChannelError::Channel {
                    reach_id: reach_ids[entry.reach].to_string(),
                    source,
                }
            })?;
            stages.push(stage);
            wets.push(wet);
        }
        let c = &self.client;
        if stages.is_empty() {
            let empty = Tensor::<R>::zeros(&[0], DType::F64, c.device())?;
            return Ok((empty.clone(), empty));
        }
        Ok((
            c.cat(&stages.iter().collect::<Vec<_>>(), 0)?,
            c.cat(&wets.iter().collect::<Vec<_>>(), 0)?,
        ))
    }
}
