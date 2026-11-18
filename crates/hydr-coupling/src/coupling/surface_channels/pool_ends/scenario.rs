//! Pool-end test fixtures: the structure tanks with the reach's right end on a constant-area pond.

use super::SurfaceChannelPoolEnd;
use crate::channel::{ChannelBoundary, ChannelEnd, ChannelLateralInput};
use crate::coupling::surface_channels::pools::scenario::{declaration, pond};
use crate::coupling::surface_channels::pools::{SurfaceChannelPoolForcing, SurfaceChannelPools};
use crate::coupling::surface_channels::structures::scenario::{
    Batch, REACH, TestResult, inflow, withdrawal,
};
use crate::coupling::surface_channels::{
    SurfaceChannelError, SurfaceChannelForcingInterval, SurfaceChannelReachForcing,
};

/// Plan area of the pond, in m².
pub const POND_AREA: f64 = 8.0;

/// Entrance loss of every fixture pool end.
pub const ENTRANCE_LOSS: f64 = 0.5;

/// Pool end `id` from the right end of the tank reach into pool `pool`.
pub fn pool_end(id: &str, pool: &str) -> SurfaceChannelPoolEnd {
    SurfaceChannelPoolEnd {
        id: id.into(),
        reach_id: REACH.into(),
        end: ChannelEnd::Right,
        pool_id: pool.into(),
        entrance_loss_coefficient: ENTRANCE_LOSS,
        physical_connection_assertion: "The reach outfall opens into the pond at its bed".into(),
        quiescent_reservoir_assertion: "The pond is quiescent and mixes the outfall water".into(),
    }
}

/// Declaration of one pond `pond` at origin `0.0`, 2 m deep, with pool ends `ends`.
pub fn ponded(ends: Vec<SurfaceChannelPoolEnd>) -> TestResult<SurfaceChannelPools> {
    let mut pools = declaration(vec![pond("pond", 0.0, POND_AREA, 2.0)?], false, Vec::new());
    pools.ends = ends;
    Ok(pools)
}

/// Forcing of one interval toward `end` with `right` at the reach's right end and caller
/// lateral rates `[inflow, withdrawal]`.
pub fn forcing(
    start: f64,
    end: f64,
    right: Option<ChannelBoundary>,
    [into, out]: [f64; 2],
) -> SurfaceChannelForcingInterval {
    let lateral: Vec<ChannelLateralInput> = vec![inflow(into), withdrawal(out)];
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Wall),
            right,
            lateral_inputs: lateral,
        }],
    }
}

/// One accepted interval toward `start + dt`, the pool owning the reach's right end, with caller
/// lateral rates `rates`.
pub fn advance(batch: &Batch, dt: f64, rates: [f64; 2]) -> Result<Batch, SurfaceChannelError> {
    let start = batch.time();
    let end = start + dt;
    let held = forcing(start, end, None, rates);
    Ok(batch
        .trial_with_pools_to(end, &held, &SurfaceChannelPoolForcing::none())?
        .into_next())
}

/// Pool-end volumes of the last accepted interval, positive into the pools, in m³.
pub fn end_volumes(batch: &Batch) -> TestResult<Vec<f64>> {
    let step = batch.last.as_ref().ok_or("an accepted interval")?;
    Ok(step.pool_ends.volume_m3.clone())
}

/// Fails unless every pool-end volume of the last interval equals its reach end volume in every
/// bit.
pub fn check_paired(batch: &Batch) -> TestResult {
    let topology = batch.context.pool_ends.as_ref().ok_or("the pool ends")?;
    let volumes = end_volumes(batch)?;
    assert_eq!(volumes.len(), topology.ends.len());
    for (end, volume) in topology.ends.iter().zip(&volumes) {
        let endpoints = batch.reaches[end.reach_index].coupling_endpoints()?;
        let outward = endpoints.outward_volume_m3.try_to_vec::<f64>()?;
        let reach = *outward.get(end.side).ok_or("the reach end volume")?;
        assert_eq!(
            reach.to_bits(),
            volume.to_bits(),
            "{}: {reach} {volume}",
            end.id
        );
    }
    Ok(())
}
