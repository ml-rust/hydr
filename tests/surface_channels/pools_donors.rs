//! Competing withdrawals from one pool: shared donor factor, rain credit, and no incoming
//! credit.
//!
//! Frozen criterion, fixed before the first run (plan §5, "Competing donors"): one pool feeds a
//! surface cell, a channel cell, and another pool with demand above storage. Factor below one.
//! Pool volume `>= 0`. Rain on the pool funds withdrawals. Incoming water funds none.
//!
//! The donor holds `0.8 m³`. Three pumps on a flat curve send its water to a surface cell, a
//! reach cell, and another pool. A flat curve asks the same volume at every stage, so the demand
//! does not fall as the donor empties. Their law volumes over one interval exceed the storage and
//! the rain together. A weir would not do: its demand falls with the donor's stage, so the
//! coordinator shortens the interval until the demand fits the storage and no factor binds.
//!
//! - Factor: every withdrawal keeps the same share of its law volume, its cap factor times its
//!   donor factor, strictly below one, and every volume is positive. The withdrawals together stay within the donor's storage.
//! - Rain: the donor takes rain over its footprint. The withdrawals together exceed the original
//!   storage and stay within the storage plus that rain, and exceed the withdrawals without rain.
//! - Incoming: a pipe delivers more than the donor holds. The withdrawals together still stay
//!   within the original storage, and the donor ends above it.
//!
//! Every interval checks the mass bound, the pairing at the surface, the channel, and every pool,
//! and the composite balance.

use crate::pools_evidence::{build, check, pool_volume};
use crate::pools_fixture::laws::pump_on;
use crate::pools_fixture::{
    Batch, Case, PoolForcing, Result, channel, forcing, inflow_binding, inflow_input, manual, pond,
    pool, structure, surface,
};
use hydr::coupling::SurfaceChannelNode;
use hydr::forcing::WaterAmount;
use hydr::storage::LevelPoolSources;
use hydr::structures::{DirectionPolicy, StructureOperatingPoint};

/// Original storage of the donor, in m³.
const STORAGE_M3: f64 = 0.8;

/// Plan area of both pools, in m².
const POOL_AREA_M2: f64 = 8.0;

/// Plateau discharge of each pump, in m³/s.
const PLATEAU: f64 = 10.0;

/// Stage of the surface cell, the reach cell, and the sink, in m.
const RECEIVER_M: f64 = 0.1;

/// Requested duration of the interval, in s.
const DT: f64 = 0.25;

/// Rain rate on the donor, in m/s.
const RAIN_M_PER_S: f64 = 0.1;

/// Pipe inflow into the donor, in m³/s.
const FEED_M3_PER_S: f64 = 4.0;

/// The donor and its three withdrawals. `sources` declares pool sources, `feed` a pipe.
fn case(sources: bool, feed: bool) -> Result<Case> {
    let mut case = Case::tanks(RECEIVER_M, RECEIVER_M);
    case.ponds = vec![
        (pond("donor", 0.0, POOL_AREA_M2, 4.0)?, STORAGE_M3),
        (
            pond("sink", 0.0, POOL_AREA_M2, 4.0)?,
            POOL_AREA_M2 * RECEIVER_M,
        ),
    ];
    case.sources = sources;
    if feed {
        case.inflows = vec![inflow_binding("feed", "donor")];
    }
    let from_donor = |id: &str, to: SurfaceChannelNode| -> Result<_> {
        Ok(structure(
            id,
            pump_on(&[(-1.0, PLATEAU), (3.0, PLATEAU)], 0.0, None)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }),
            (pool("donor"), 0.0),
            (to, 0.0),
        ))
    };
    case.structures = vec![
        from_donor("to-channel", channel())?,
        from_donor("to-sink", pool("sink"))?,
        from_donor("to-surface", surface())?,
    ];
    Ok(case)
}

/// What one accepted interval of the donor case did.
struct Outcome {
    /// Factor and signed volume of every withdrawal. The factor is the cap factor times the
    /// donor factor: the share of its law volume the withdrawal keeps.
    withdrawals: Vec<(f64, f64)>,
    /// Native incoming volume of the donor.
    incoming: f64,
    /// Accepted duration.
    span: f64,
    donor_volume: f64,
}

/// One interval of `case` under `pools`, checked. `pipes` holds each pool's pipe inflow rate in
/// m³/s, in ascending pool identity. Rain is no native incoming volume, so it stays out.
fn interval(case: &Case, pools: &PoolForcing, pipes: [f64; 2]) -> Result<(Batch, Outcome)> {
    let batch = build(case)?;
    let held = forcing(&batch, DT, pools);
    let next = batch.trial_to(DT, &held)?.into_next();
    let step = next.last_step()?.ok_or("an accepted interval")?;
    let span = step.end_s - step.start_s;
    let external = [pipes[0] * span, pipes[1] * span];
    check(&next, &external)?;
    let withdrawals = step
        .structures
        .iter()
        .map(|t| (t.cap_factor * t.donor_factor, t.signed_volume_m3))
        .collect();
    let donor = step
        .pools
        .iter()
        .find(|p| p.pool_id == "donor")
        .ok_or("the donor")?;
    let outcome = Outcome {
        withdrawals,
        incoming: donor.incoming_m3,
        span,
        donor_volume: pool_volume(&next, "donor")?,
    };
    Ok((next, outcome))
}

/// Total withdrawal volume, in m³.
fn withdrawn(outcome: &Outcome) -> f64 {
    outcome.withdrawals.iter().map(|w| w.1).sum()
}

/// Source rates of the donor and the sink: rain on the donor only.
fn rain() -> PoolForcing {
    let rates = |rain_rate_m_per_s| LevelPoolSources {
        rain_rate_m_per_s,
        potential_evaporation_rate_m_per_s: 0.0,
        bottom_loss_rate_m3_per_s: 0.0,
    };
    PoolForcing {
        sources: Some(vec![rates(RAIN_M_PER_S), rates(0.0)]),
        inflows: Vec::new(),
    }
}

#[test]
fn competing_withdrawals_share_one_factor_below_one() -> Result {
    let (_, outcome) = interval(&case(false, false)?, &PoolForcing::default(), [0.0; 2])?;
    let factor = outcome.withdrawals[0].0;
    assert!(factor > 0.0 && factor < 1.0, "{:?}", outcome.withdrawals);
    for (donor_factor, volume) in &outcome.withdrawals {
        assert_eq!(donor_factor.to_bits(), factor.to_bits());
        assert!(*volume > 0.0, "{:?}", outcome.withdrawals);
    }
    let total = withdrawn(&outcome);
    assert!(total <= STORAGE_M3, "withdrawn={total}");
    assert!(outcome.donor_volume >= 0.0, "{}", outcome.donor_volume);
    println!("donors: factor={factor}, withdrawn={total}");
    Ok(())
}

#[test]
fn rain_on_the_pool_funds_withdrawals() -> Result {
    let (_, dry) = interval(
        &case(true, false)?,
        &PoolForcing {
            sources: Some(vec![
                LevelPoolSources {
                    rain_rate_m_per_s: 0.0,
                    potential_evaporation_rate_m_per_s: 0.0,
                    bottom_loss_rate_m3_per_s: 0.0,
                };
                2
            ]),
            inflows: Vec::new(),
        },
        [0.0; 2],
    )?;
    let (_, wet) = interval(&case(true, false)?, &rain(), [0.0; 2])?;
    let raining = RAIN_M_PER_S * POOL_AREA_M2 * wet.span;
    let (dry_total, wet_total) = (withdrawn(&dry), withdrawn(&wet));
    let factor = wet.withdrawals[0].0;
    assert!(factor > 0.0 && factor < 1.0, "{:?}", wet.withdrawals);
    for (donor_factor, _) in &wet.withdrawals {
        assert_eq!(donor_factor.to_bits(), factor.to_bits());
    }
    assert!(wet_total > STORAGE_M3, "withdrawn={wet_total}");
    assert!(
        wet_total <= STORAGE_M3 + raining,
        "withdrawn={wet_total}, rain={raining}"
    );
    assert!(wet_total > dry_total, "wet={wet_total}, dry={dry_total}");
    assert!(wet.donor_volume >= 0.0, "{}", wet.donor_volume);
    Ok(())
}

#[test]
fn incoming_water_funds_no_withdrawal() -> Result {
    let pools = PoolForcing {
        sources: None,
        inflows: vec![inflow_input(
            "feed",
            WaterAmount::Rate {
                rate_m3_per_s: FEED_M3_PER_S,
            },
        )],
    };
    let (_, outcome) = interval(&case(false, true)?, &pools, [FEED_M3_PER_S, 0.0])?;
    let total = withdrawn(&outcome);
    assert!(
        outcome.incoming > STORAGE_M3,
        "incoming={}",
        outcome.incoming
    );
    assert!(total > 0.0 && total <= STORAGE_M3, "withdrawn={total}");
    let factor = outcome.withdrawals[0].0;
    assert!(factor > 0.0 && factor < 1.0, "{:?}", outcome.withdrawals);
    assert!(
        outcome.donor_volume > STORAGE_M3,
        "{}",
        outcome.donor_volume
    );
    Ok(())
}
