//! Pooled checkpoints with every operation owner: bitwise restart against the uninterrupted
//! run, decision replay, forged packages, and rejection of the earlier package format.

use crate::pools_fixture::{
    Batch, Case, PoolForcing, Result, channel, controlled_gate, controlled_pump, gate_law,
    inflow_binding, inflow_input, manual, operating, pond, pool, pool_end, pump_law, signal, state,
    step, structure, surface, trajectory, transfer, weir_law,
};
use hydr::channel::ChannelEnd;
use hydr::coupling::{
    SurfaceChannelError, SurfaceChannelOperating, SurfaceChannelOperationOwner,
    SurfaceChannelOperationReceipt, SurfaceChannelSnapshotData,
};
use hydr::forcing::WaterAmount;
use hydr::structures::{ControlCommand, DirectionPolicy, StructureOperatingPoint};

/// Requested duration of every interval, in s.
const DT: f64 = 0.5;
/// Decision times of both relays, in s.
const DECISIONS: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// Time of the checkpoint, in s. The interval before it ends at the decision at 2 s.
const CHECKPOINT: f64 = 2.0;
/// Intervals after the checkpoint.
const AFTER: usize = 6;

/// A pond fed by a pipe and by the reach's right end, draining onto the surface through a
/// manual spill weir and an actuated outlet gate. A prescribed sluice joins the surface to the
/// reach. A controlled pump lifts a sump onto the surface through a free outlet.
fn case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.5);
    case.ponds = vec![
        (pond("pond", 0.0, 8.0, 2.0)?, 0.45 * 8.0),
        (pond("sump", 0.0, 4.0, 2.0)?, 0.5 * 4.0),
    ];
    case.inflows = vec![inflow_binding("feed", "pond")];
    case.pool_ends = vec![pool_end("outfall", ChannelEnd::Right, "pond")];
    let actuated = SurfaceChannelOperationOwner::Actuated;
    case.structures = vec![
        structure(
            "spill",
            weir_law(0.9, 1.0)?,
            DirectionPolicy::Bidirectional,
            manual(StructureOperatingPoint::Passive),
            (pool("pond"), 0.0),
            (surface(), 0.0),
        ),
        structure(
            "outlet",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            actuated,
            (pool("pond"), 0.0),
            (surface(), 0.0),
        ),
        structure(
            "sluice",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            SurfaceChannelOperationOwner::Prescribed,
            (surface(), 0.0),
            (channel(), 0.0),
        ),
        structure(
            "lift",
            pump_law(0.1, Some(1.0))?,
            DirectionPolicy::AtoBOnly,
            SurfaceChannelOperationOwner::ControlledPump,
            (pool("sump"), 0.0),
            (surface(), 0.0),
        ),
    ];
    let schedule = (0.0, &DECISIONS[..], 20.0);
    case.operating = Some(SurfaceChannelOperating {
        gates: Some(controlled_gate(
            "outlet",
            [0.4, 0.5],
            ([0.0, 0.3], 0.1),
            schedule,
        )),
        pumps: Some(controlled_pump(
            "lift",
            [0.3, 0.45],
            [
                ControlCommand::PumpStoppedIsolated,
                ControlCommand::PumpRunning { speed_ratio: 1.0 },
            ],
            schedule,
        )),
        trajectories: vec![trajectory(
            "sluice",
            &[(0.0, 0.0), (1.0, 0.0), (3.0, 0.1), (20.0, 0.1)],
        )?],
        ..operating(vec![
            signal("outlet", pool("pond")),
            signal("lift", pool("sump")),
        ])
    });
    Ok(case)
}

fn pools() -> PoolForcing {
    PoolForcing {
        sources: None,
        inflows: vec![inflow_input(
            "feed",
            WaterAmount::Rate {
                rate_m3_per_s: 0.05,
            },
        )],
    }
}

/// `batch` advanced to `CHECKPOINT`, each request at most `DT` and none past it.
fn to_checkpoint(mut batch: Batch) -> Result<Batch> {
    while batch.time() < CHECKPOINT {
        batch = step(&batch, (CHECKPOINT - batch.time()).min(DT), &pools())?;
    }
    Ok(batch)
}

/// `batch` after `count` more intervals.
fn run(mut batch: Batch, count: usize) -> Result<Batch> {
    for _ in 0..count {
        batch = step(&batch, DT, &pools())?;
    }
    Ok(batch)
}

/// Input messages of import into a fresh batch and of resume, in that order.
fn rejections(data: &SurfaceChannelSnapshotData) -> Result<[String; 2]> {
    let message = |result: std::result::Result<Batch, SurfaceChannelError>| match result {
        Ok(_) => Err("the coordinator accepted a forged package".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    };
    let imported: Result<String> = message(case()?.build()?.import_snapshot(data));
    let resumed: Result<String> = message(Batch::resume(
        crate::runtime::client(),
        data,
        data.surface.setup.forcing.clone(),
    ));
    Ok([imported?, resumed?])
}

/// Fails unless import and resume both reject `data` with every fragment.
fn rejects(data: &SurfaceChannelSnapshotData, fragments: &[&str]) -> Result {
    for message in rejections(data)? {
        for fragment in fragments {
            assert!(message.contains(fragment), "{message:?} lacks {fragment:?}");
        }
    }
    Ok(())
}

/// Frozen criteria, fixed before the first run:
/// - The checkpoint interval ends at a decision time and carries an actuated receipt, a
///   prescribed receipt, a controlled-pump receipt, and a manual receipt.
/// - Before the checkpoint the pump runs and the pond fills through its pool end.
/// - Snapshot and restore, import into a fresh batch, and resume each match the checkpoint
///   state bit for bit.
/// - Each continues `AFTER` intervals and matches the uninterrupted run bit for bit, package,
///   every ledger, and every child state included.
#[test]
fn every_restart_path_matches_the_uninterrupted_run() -> Result {
    let checkpoint = to_checkpoint(case()?.build()?)?;
    let last = checkpoint.last_step()?.ok_or("an accepted interval")?;
    assert_eq!(last.end_s.to_bits(), CHECKPOINT.to_bits());
    assert!(DECISIONS.contains(&last.end_s));
    let receipts = [
        transfer(&checkpoint, "outlet")?.receipt,
        transfer(&checkpoint, "sluice")?.receipt,
        transfer(&checkpoint, "lift")?.receipt,
        transfer(&checkpoint, "spill")?.receipt,
    ];
    assert!(
        matches!(
            receipts,
            [
                SurfaceChannelOperationReceipt::ActuatedGate { .. },
                SurfaceChannelOperationReceipt::PrescribedGate { .. },
                SurfaceChannelOperationReceipt::ControlledPump { .. },
                SurfaceChannelOperationReceipt::Manual,
            ]
        ),
        "{receipts:?}"
    );
    let ledgers = checkpoint.structure_ledgers()?;
    let lift = ledgers
        .iter()
        .find(|l| l.structure_id == "lift")
        .ok_or("the pump ledger")?;
    assert!(lift.a_to_b_m3 > 0.0, "{lift:?}");
    let ends = checkpoint.pool_end_ledgers()?;
    assert!(
        ends.iter()
            .any(|e| e.reach_to_pool_m3 > 0.0 || e.pool_to_reach_m3 > 0.0)
    );
    let expected = state(&checkpoint)?;
    let data = checkpoint.snapshot_data()?;
    let restored = case()?.build()?.restore(&checkpoint.snapshot())?;
    let imported = case()?.build()?.import_snapshot(&data)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    let uninterrupted = state(&run(checkpoint, AFTER)?)?;
    for (path, batch) in [
        ("restore", restored),
        ("import", imported),
        ("resume", resumed),
    ] {
        assert_eq!(state(&batch)?, expected, "{path} at the checkpoint");
        assert_eq!(
            state(&run(batch, AFTER)?)?,
            uninterrupted,
            "{path} after the checkpoint"
        );
    }
    Ok(())
}

/// Frozen criteria, fixed before the first run: a package before the first interval imports
/// and resumes to the fresh batch state bit for bit.
#[test]
fn a_zero_history_package_restores_the_fresh_batch() -> Result {
    let fresh = case()?.build()?;
    let data = fresh.snapshot_data()?;
    let expected = state(&fresh)?;
    assert_eq!(state(&case()?.build()?.import_snapshot(&data)?)?, expected);
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    assert_eq!(state(&resumed)?, expected);
    Ok(())
}

/// The checkpoint package of the restart case.
fn package() -> Result<SurfaceChannelSnapshotData> {
    Ok(to_checkpoint(case()?.build()?)?.snapshot_data()?)
}

/// Frozen criteria, fixed before the first run: import and resume both reject, naming the
/// field:
/// - a pool package whose current volume moved (`pools`)
/// - a start-frame pool package whose current volume moved (`last_interval.previous.pools`)
/// - a structure ledger that moved (`counters.structures`)
/// - a pool-end ledger that moved (`counters.pool_ends`)
/// - a pool inflow ledger that moved (`counters.pool_inflows`)
/// - a controlled-gate package whose latch moved, and a controlled-pump package whose latch
///   moved (`operating`)
/// - a receipt structure transfer that moved (`last_interval.diagnostics.structures`)
#[test]
fn forged_pool_structure_and_control_packages_reject() -> Result {
    let data = package()?;
    let mut forged = data.clone();
    let pools = forged.pools.as_mut().ok_or("the pool package")?;
    pools.states[0].volume_m3 += 1e-6;
    rejections(&forged)?;
    let mut forged = data.clone();
    let last = forged.last_interval.as_mut().ok_or("the last interval")?;
    let previous = last.previous.pools.as_mut().ok_or("the start pools")?;
    previous.states[0].volume_m3 += 1e-6;
    rejections(&forged)?;
    let mut forged = data.clone();
    forged.counters.structures[0].a_to_b_m3 += 1e-6;
    rejects(&forged, &["counters"])?;
    let mut forged = data.clone();
    forged.counters.pool_ends[0].reach_to_pool_m3 += 1e-6;
    rejects(&forged, &["counters"])?;
    let mut forged = data.clone();
    forged.counters.pool_inflows[0].cumulative_volume_m3 += 1e-6;
    rejects(&forged, &["counters.pool_inflows"])?;
    let mut forged = data.clone();
    let operating = forged.operating.as_mut().ok_or("the operating package")?;
    let gate = operating.gate.as_mut().ok_or("the gate package")?;
    gate.controls.states[0].latch = flip(gate.controls.states[0].latch);
    rejections(&forged)?;
    let mut forged = data.clone();
    let operating = forged.operating.as_mut().ok_or("the operating package")?;
    let pump = operating.pump.as_mut().ok_or("the pump package")?;
    pump.controls.states[0].latch = flip(pump.controls.states[0].latch);
    rejections(&forged)?;
    let mut forged = data;
    let last = forged.last_interval.as_mut().ok_or("the last interval")?;
    last.diagnostics.structures[0].law_volume_m3 += 1e-6;
    rejects(
        &forged,
        &[
            "last_interval.diagnostics.structures[0]",
            "differs from the recalculated last interval",
        ],
    )
}

fn flip(latch: hydr::structures::ControlLatch) -> hydr::structures::ControlLatch {
    match latch {
        hydr::structures::ControlLatch::Low => hydr::structures::ControlLatch::High,
        hydr::structures::ControlLatch::High => hydr::structures::ControlLatch::Low,
    }
}

/// Frozen criteria, fixed before the first run: a package naming schema `4` or contract `6`
/// rejects by identity in import and resume, before any other check.
#[test]
fn a_checkpoint_4_package_rejects_by_identity() -> Result {
    let data = package()?;
    let mut earlier = data.clone();
    earlier.schema = "hydr-surface-channel-checkpoint-4".into();
    rejects(
        &earlier,
        &[
            "surface-channel checkpoint schema",
            "found \"hydr-surface-channel-checkpoint-4\"",
        ],
    )?;
    let mut earlier = data;
    earlier.contract = "hydr-surface-channel-numr-cpu-6".into();
    rejects(
        &earlier,
        &[
            "surface-channel checkpoint contract",
            "found \"hydr-surface-channel-numr-cpu-6\"",
        ],
    )
}
