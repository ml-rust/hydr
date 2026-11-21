//! Pool-end test fixtures: the structure tanks with the reach's right end on a constant-area pond.

use super::SurfaceChannelPoolEnd;
use crate::channel::{ChannelBoundary, ChannelEnd, ChannelGeometry, ChannelLateralInput};
use crate::coupling::channel_pools::three_band_reach;
use crate::coupling::surface_channels::pools::SurfaceChannelPools;
use crate::coupling::surface_channels::pools::scenario::{declaration, pond, with_nodes};
use crate::coupling::surface_channels::structures::scenario::{
    Batch, DATUM, REACH, Tanks, TestResult, client, inflow, withdrawal,
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

/// The batch of one three-band surveyed reach cell at water depth `depth` and the pond at
/// volume `pond_volume`, with `pools` owning the reach's right end.
///
/// The cell is 2 m long. Its bottom band is 2 m wide with unit side slopes, so below depth 1 m
/// it holds `2 * depth * (2 + depth)`.
pub fn surveyed(
    depth: f64,
    pond_volume: f64,
    pools: SurfaceChannelPools,
) -> Result<Batch, SurfaceChannelError> {
    let input = |error: Box<dyn std::error::Error>| SurfaceChannelError::Input(error.to_string());
    let (mut setup, mut initial) = Tanks::at(0.0, depth).parts().map_err(input)?;
    let mut definition = three_band_reach(1, 2.0);
    definition.vertical_datum = DATUM.into();
    if let ChannelGeometry::PrismaticSurveyed { profile, .. } = &mut definition.geometry {
        profile.vertical_datum = DATUM.into();
    }
    definition.lateral_bindings = std::mem::take(&mut setup.reaches[0].definition.lateral_bindings);
    definition.config = setup.reaches[0].definition.config.clone();
    setup.reaches[0].definition = definition;
    initial.reaches[0].1.volume_m3 = vec![2.0 * depth * (2.0 + depth)];
    let (setup, initial) = with_nodes((setup, initial), pools, &[pond_volume], Vec::new());
    Batch::new(client(), setup, initial)
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
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: Vec::new(),
    }
}

/// One accepted interval toward `start + dt`, the pool owning the reach's right end, with caller
/// lateral rates `rates`.
pub fn advance(batch: &Batch, dt: f64, rates: [f64; 2]) -> Result<Batch, SurfaceChannelError> {
    let start = batch.time();
    let end = start + dt;
    let held = forcing(start, end, None, rates);
    Ok(batch.trial_to(end, &held)?.into_next())
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
