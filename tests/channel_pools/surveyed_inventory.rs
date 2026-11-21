//! Two-part inventory of a surveyed reach joined to a pool.
//!
//! Four 32 m cells of three bands start at rest at depth 1.5 m. One end joins a pool of
//! plan area `1`, `4`, or `16 m²`. The other end holds the rest state. Every accepted
//! interval of two `0.125 s` requests:
//! - at least one reach cell carries a nonzero volume low part
//! - the composite residual stays within the mass bound

use super::fixture::{DATUM, Result, batch, check_water, config, connection, pool};
use super::surveyed_fixture::{definition, held};
use super::surveyed_profiles::THREE_BANDS;
use hydr::channel::ChannelInitialState;
use hydr::coupling::{
    ChannelPoolEndpoint, ChannelPoolInitialState, ChannelPoolReach, ChannelPoolSetup,
};
use hydr::storage::LevelPoolState;

/// Leaving and entering pool stages with their entrance loss coefficients.
const STAGES: [(f64, f64); 2] = [(0.875, 0.0), (2.5, 0.5)];
/// Requested interval in seconds.
const SPAN_S: f64 = 0.125;
/// Requested intervals per run.
const INTERVALS: usize = 2;
/// End time in seconds.
const END_S: f64 = INTERVALS as f64 * SPAN_S;

fn run(endpoint: ChannelPoolEndpoint, area: f64, stage: f64, loss: f64) -> Result {
    let setup = ChannelPoolSetup {
        reaches: vec![ChannelPoolReach {
            id: "reach".into(),
            definition: definition(THREE_BANDS, 1.0, 4, 32.0),
        }],
        pools: vec![pool("pool", area, 0.0)?],
        connections: vec![connection("connection", "reach", endpoint, "pool", loss)],
        vertical_datum: DATUM.into(),
        nonoverlap_assertion: "Basin and reach occupy separate physical storage".into(),
        config: config(),
    };
    let initial = ChannelPoolInitialState {
        time_s: 0.0,
        reaches: vec![ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![32.0 * 5.125; 4],
            longitudinal_momentum_m4_per_s: vec![0.0; 4],
        }],
        pools: vec![LevelPoolState {
            volume_m3: area * stage,
        }],
    };
    let mut current = batch(setup.clone(), initial)?;
    let mut intervals = 0;
    while current.time() < END_S {
        let start = current.time();
        let end = (start + SPAN_S).min(END_S);
        let interval = held(&setup, 1.5, start, end);
        current = current.trial_to(end, &interval)?.into_next();
        assert!(current.time() > start && current.time() <= end);
        let reach = current.reach("reach").ok_or("Missing reach")?;
        let lows = reach
            .coupling_accounting()
            .cell_volume_low_m3
            .try_to_vec::<f64>()?;
        assert!(
            lows.iter().any(|low| *low != 0.0),
            "{endpoint:?} area{area} stage{stage}: {lows:?}"
        );
        check_water(&current)?;
        intervals += 1;
    }
    assert!(intervals >= INTERVALS);
    Ok(())
}

/// The leaving stage of `STAGES`, with its loss.
fn leaving(endpoint: ChannelPoolEndpoint, area: f64) -> Result {
    run(endpoint, area, STAGES[0].0, STAGES[0].1)
}

/// The entering stage of `STAGES`, with its loss.
fn entering(endpoint: ChannelPoolEndpoint, area: f64) -> Result {
    run(endpoint, area, STAGES[1].0, STAGES[1].1)
}

#[test]
fn surveyed_inventory_leaving_pool_area_1_right() -> Result {
    leaving(ChannelPoolEndpoint::Right, 1.0)
}

#[test]
fn surveyed_inventory_leaving_pool_area_4_right() -> Result {
    leaving(ChannelPoolEndpoint::Right, 4.0)
}

#[test]
fn surveyed_inventory_leaving_pool_area_16_right() -> Result {
    leaving(ChannelPoolEndpoint::Right, 16.0)
}

#[test]
fn surveyed_inventory_entering_pool_area_1_right() -> Result {
    entering(ChannelPoolEndpoint::Right, 1.0)
}

#[test]
fn surveyed_inventory_entering_pool_area_4_right() -> Result {
    entering(ChannelPoolEndpoint::Right, 4.0)
}

#[test]
fn surveyed_inventory_entering_pool_area_16_right() -> Result {
    entering(ChannelPoolEndpoint::Right, 16.0)
}

#[test]
fn surveyed_inventory_leaving_pool_area_4_left() -> Result {
    leaving(ChannelPoolEndpoint::Left, 4.0)
}

#[test]
fn surveyed_inventory_entering_pool_area_4_left() -> Result {
    entering(ChannelPoolEndpoint::Left, 4.0)
}
