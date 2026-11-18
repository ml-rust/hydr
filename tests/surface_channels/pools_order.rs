//! Declaration order of pools, structure links, pool ends, and bank links changes no bit.

use crate::pools_fixture::{
    Batch, Case, PoolForcing, Result, bank, channel, gate_law, inflow_binding, inflow_input,
    manual, orifice_law, pond, pool, pool_end, pump_law, step, structure, surface, weir_law,
};
use hydr::channel::ChannelEnd;
use hydr::forcing::WaterAmount;
use hydr::structures::{DirectionPolicy, StructureOperatingPoint};

/// Two ponds on both reach ends, joined to the tanks and to each other by four structures,
/// beside one bank link and a pipe into the first pond.
fn case() -> Result<Case> {
    let mut case = Case::tanks(0.6, 0.5);
    case.ponds = vec![
        (pond("east", 0.0, 8.0, 2.0)?, 0.45 * 8.0),
        (pond("west", 0.0, 6.0, 2.0)?, 0.7 * 6.0),
    ];
    case.inflows = vec![inflow_binding("feed", "east")];
    case.links = vec![bank("bank", 0.4)];
    case.pool_ends = vec![
        pool_end("east-end", ChannelEnd::Right, "east"),
        pool_end("west-end", ChannelEnd::Left, "west"),
    ];
    let passive = || manual(StructureOperatingPoint::Passive);
    case.structures = vec![
        structure(
            "culvert",
            orifice_law(0.0, 0.2, true)?,
            DirectionPolicy::Bidirectional,
            passive(),
            (surface(), 0.0),
            (channel(), 0.0),
        ),
        structure(
            "spill",
            weir_law(0.55, 1.0)?,
            DirectionPolicy::Bidirectional,
            passive(),
            (pool("west"), 0.0),
            (surface(), 0.0),
        ),
        structure(
            "gate",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            manual(StructureOperatingPoint::GateOpening { opening_m: 0.1 }),
            (pool("west"), 0.0),
            (pool("east"), 0.0),
        ),
        structure(
            "lift",
            pump_law(0.1, Some(1.0))?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }),
            (pool("east"), 0.0),
            (surface(), 0.0),
        ),
    ];
    Ok(case)
}

/// Text of every order-free result: receipts, ledgers, balance, and child states.
///
/// The setup and the original states hold declaration order, so they stay out.
fn results(batch: &Batch) -> Result<String> {
    let pools = batch.pools().ok_or("the pools")?;
    Ok(format!(
        "{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
        batch.last_step()?,
        batch.ledger()?,
        batch.link_ledgers()?,
        batch.structure_ledgers()?,
        batch.pool_end_ledgers()?,
        batch.pool_inflow_ledgers()?,
        pools.states()?,
        pools.elements(),
        batch
            .reach(crate::pools_fixture::REACH)
            .ok_or("the reach")?
            .state()?,
        batch.surface().state()?
    ))
}

/// `case` with every declaration list reversed. Pond states follow their ponds.
fn reversed(mut case: Case) -> Case {
    case.ponds.reverse();
    case.links.reverse();
    case.structures.reverse();
    case.pool_ends.reverse();
    case
}

/// `case` with every declaration list rotated by one.
fn rotated(mut case: Case) -> Case {
    case.ponds.rotate_left(1);
    case.structures.rotate_left(1);
    case.pool_ends.rotate_left(1);
    case
}

/// Frozen criteria, fixed before the first run:
/// - Over 8 intervals, the declared, reversed, and rotated orders give the same receipts,
///   ledgers, composite balance, and child states, bit for bit, after every interval.
/// - Every structure moves water at least once, so the order acts on live links.
#[test]
fn declaration_order_changes_no_bit() -> Result {
    let forcing = PoolForcing {
        sources: None,
        inflows: vec![inflow_input(
            "feed",
            WaterAmount::Rate {
                rate_m3_per_s: 0.05,
            },
        )],
    };
    let mut batches = [
        case()?.build()?,
        reversed(case()?).build()?,
        rotated(case()?).build()?,
    ];
    let mut moved = [false; 4];
    for _ in 0..8 {
        for batch in &mut batches {
            *batch = step(batch, 0.25, &forcing)?;
        }
        let expected = results(&batches[0])?;
        for batch in &batches[1..] {
            assert_eq!(results(batch)?, expected);
        }
        let step = batches[0].last_step()?.ok_or("an accepted interval")?;
        for (flag, transfer) in moved.iter_mut().zip(&step.structures) {
            *flag |= transfer.signed_volume_m3 != 0.0;
        }
    }
    assert_eq!(moved, [true; 4], "every structure moves water");
    Ok(())
}
