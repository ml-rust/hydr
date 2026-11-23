//! Surface/channel coordinator pool curve epochs: smaller capacity accept/reject, attachment bed
//! below new origin rejection, dry invariance, coincident control decision alignment, and bitwise
//! restart with forged `pool_curve_epochs` rejection.

use crate::pools_evidence::{check, pool_stage};
use crate::pools_fixture::{
    Case, DATUM, PoolForcing, Result, controlled_gate, gate_law, manual, operating, pond, pool,
    signal, step, structure, transfer, weir_law,
};
use crate::runtime::client;
use hydr::coupling::{
    SurfaceChannelBatch, SurfaceChannelDefinition, SurfaceChannelError,
    SurfaceChannelOperationOwner,
};
use hydr::storage::{LevelPoolCurve, LevelPoolKnot, PoolCurveEpoch, PoolCurveSchedule};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint};

fn linear_curve(origin: f64, area: f64, depth: f64) -> Result<LevelPoolCurve> {
    Ok(LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: origin,
            },
            LevelPoolKnot {
                volume_m3: area * depth,
                stage_m: origin + depth,
            },
        ],
        DATUM.into(),
        "Synthetic epoch curve".into(),
    )?)
}

fn advance_to(
    mut batch: SurfaceChannelBatch<crate::runtime::Runtime>,
    target_s: f64,
) -> Result<SurfaceChannelBatch<crate::runtime::Runtime>> {
    let forcing = PoolForcing::default();
    while batch.time() < target_s {
        let remaining = target_s - batch.time();
        batch = step(&batch, remaining, &forcing)?;
        check(&batch, &[])?;
        let diag = batch.last_step()?.ok_or("missing step diagnostics")?;
        for receipt in &diag.geometry {
            assert_eq!(
                receipt.storage_before_m3.to_bits(),
                receipt.storage_after_m3.to_bits(),
                "pool curve epoch must conserve total storage bit-for-bit: {receipt:?}"
            );
        }
    }
    Ok(batch)
}

/// Smaller capacity curve epoch at t = 0.5 s: accepts when current volume fits below the new top
/// knot (raising stage while conserving volume bit-for-bit), and rejects without retry when
/// current volume exceeds the new top knot.
#[test]
fn smaller_capacity_curve_accepts_below_top_and_rejects_above() -> Result {
    // Pond starts with area=10, depth=2 (max_vol=20), initial vol=6 (stage=0.6), below crest 1.0.
    // At t = 0.5 s, curve shrinks to area=4, depth=2 (max_vol=8).
    // Volume 6 <= 8: accepts, stage jumps from 0.6 to 6/4 = 1.5 > crest 1.0, and weir spills!
    {
        let mut case = Case::tanks(0.0, 0.0);
        case.ponds = vec![
            (pond("basin", 0.0, 40.0, 3.0)?, 0.0),
            (pond("pond", 0.0, 10.0, 2.0)?, 6.0),
        ];
        case.structures = vec![structure(
            "spill",
            weir_law(1.0, 1.0)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("pond"), 0.0),
            (pool("basin"), 0.0),
        )];
        let (mut setup, initial) = case.parts()?;
        setup.config.maximum_step_s = 1.0;
        setup.pool_curve_schedules = vec![PoolCurveSchedule {
            schedule_id: "silt-basin".into(),
            pool_id: "pond".into(),
            epochs: vec![PoolCurveEpoch {
                time_s: 0.5,
                curve: linear_curve(0.0, 4.0, 2.0)?,
                evidence: "Siltation narrows active storage".into(),
            }],
        }];
        let batch0 = SurfaceChannelBatch::new(client(), setup, initial)?;
        let forcing = PoolForcing::default();

        // Requesting 1.0 s must split at t = 0.5 s.
        let batch1 = step(&batch0, 1.0, &forcing)?;
        check(&batch1, &[])?;
        assert_eq!(batch1.time(), 0.5);
        assert_eq!(
            transfer(&batch1, "spill")?.signed_volume_m3.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(pool_stage(&batch1, "pond")?, 0.6);

        // Next interval [0.5, 1.0] applies the smaller curve: stage jumps to 1.5 -> weir spills!
        let batch2 = step(&batch1, 0.5, &forcing)?;
        check(&batch2, &[])?;
        assert_eq!(batch2.time(), 1.0);
        assert!(
            transfer(&batch2, "spill")?.signed_volume_m3 > 0.0,
            "raised stage after curve shrink must spill over crest 1.0"
        );
        let d2 = batch2.last_step()?.ok_or("missing step diagnostics")?;
        assert_eq!(d2.geometry.len(), 1);
        assert_eq!(d2.geometry[0].element_id, "pond");
        assert_eq!(
            d2.geometry[0].storage_before_m3.to_bits(),
            d2.geometry[0].storage_after_m3.to_bits()
        );
    }

    // Same setup, but initial volume is 9.0 > new top knot 8.0: must reject at t = 0.5 s without retry.
    {
        let mut case = Case::tanks(0.0, 0.0);
        case.ponds = vec![
            (pond("basin", 0.0, 40.0, 3.0)?, 0.0),
            (pond("pond", 0.0, 10.0, 2.0)?, 9.0),
        ];
        case.structures = vec![structure(
            "spill",
            weir_law(1.0, 1.0)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("pond"), 0.0),
            (pool("basin"), 0.0),
        )];
        let (mut setup, initial) = case.parts()?;
        setup.config.maximum_step_s = 1.0;
        setup.pool_curve_schedules = vec![PoolCurveSchedule {
            schedule_id: "silt-overflow".into(),
            pool_id: "pond".into(),
            epochs: vec![PoolCurveEpoch {
                time_s: 0.5,
                curve: linear_curve(0.0, 4.0, 2.0)?,
                evidence: "Capacity drops below stored volume".into(),
            }],
        }];
        let batch0 = SurfaceChannelBatch::new(client(), setup, initial)?;
        let forcing = PoolForcing::default();
        let batch1 = step(&batch0, 1.0, &forcing)?;
        assert_eq!(batch1.time(), 0.5);
        match step(&batch1, 0.5, &forcing) {
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("pool-capacity-exceeded"),
                    "expected pool-capacity-exceeded, got {msg:?}"
                );
            }
            Ok(_) => return Err("expected pool-capacity-exceeded".into()),
        }
    }
    Ok(())
}

/// An epoch whose new curve origin rises above an attached structure bed rejects at admission.
#[test]
fn attachment_bed_below_new_origin_rejects_at_admission() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("basin", 0.0, 40.0, 3.0)?, 0.0),
        (pond("pond", 0.0, 10.0, 2.0)?, 4.0),
    ];
    case.structures = vec![structure(
        "spill",
        weir_law(1.0, 1.0)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("pond"), 0.2),
        (pool("basin"), 0.0),
    )];
    let (mut setup, _initial) = case.parts()?;
    setup.pool_curve_schedules = vec![PoolCurveSchedule {
        schedule_id: "raised-origin".into(),
        pool_id: "pond".into(),
        epochs: vec![PoolCurveEpoch {
            time_s: 0.5,
            curve: linear_curve(0.5, 10.0, 2.0)?,
            evidence: "Bed rises above attachment bed 0.2".into(),
        }],
    }];
    match SurfaceChannelDefinition::new(setup) {
        Err(SurfaceChannelError::Input(msg)) => {
            assert!(
                msg.contains("unobstructed-pool-connection"),
                "expected unobstructed-pool-connection, got {msg:?}"
            );
        }
        other => return Err(format!("expected admission error, got {other:?}").into()),
    }
    Ok(())
}

/// Dry pools across a curve epoch move exact zero water.
#[test]
fn dry_pools_move_zero_across_curve_epoch() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("a", 0.0, 10.0, 2.0)?, 0.0),
        (pond("b", 0.0, 10.0, 2.0)?, 0.0),
    ];
    case.structures = vec![structure(
        "weir",
        weir_law(0.8, 2.0)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("a"), 0.0),
        (pool("b"), 0.0),
    )];
    let (mut setup, initial) = case.parts()?;
    setup.pool_curve_schedules = vec![PoolCurveSchedule {
        schedule_id: "dry-shrink".into(),
        pool_id: "a".into(),
        epochs: vec![PoolCurveEpoch {
            time_s: 0.5,
            curve: linear_curve(0.0, 5.0, 2.0)?,
            evidence: "Dry epoch".into(),
        }],
    }];
    let batch = advance_to(SurfaceChannelBatch::new(client(), setup, initial)?, 1.0)?;
    assert_eq!(
        transfer(&batch, "weir")?.signed_volume_m3.to_bits(),
        0.0_f64.to_bits()
    );
    Ok(())
}

/// A pool curve epoch coincident with a control decision aligns cleanly at t = 0.5 s.
#[test]
fn coincident_pool_curve_and_control_decision_align() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 20.0, 3.0)?, 4.0),
        (pond("upper", 0.0, 20.0, 3.0)?, 20.0),
    ];
    case.structures = vec![structure(
        "gate",
        gate_law(0.4)?,
        DirectionPolicy::AtoBOnly,
        SurfaceChannelOperationOwner::Actuated,
        (pool("upper"), 0.0),
        (pool("lower"), 0.0),
    )];
    let mut op = operating(vec![signal("gate", pool("upper"))]);
    op.gates = Some(controlled_gate(
        "gate",
        [0.8, 1.2],
        ([0.1, 0.5], 0.5),
        (0.0, &[0.5], 2.0),
    ));
    case.operating = Some(op);
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = 1.0;
    setup.pool_curve_schedules = vec![PoolCurveSchedule {
        schedule_id: "coincident-curve".into(),
        pool_id: "upper".into(),
        epochs: vec![PoolCurveEpoch {
            time_s: 0.5,
            curve: linear_curve(0.0, 15.0, 3.0)?,
            evidence: "Coincident curve narrowing".into(),
        }],
    }];
    let batch0 = SurfaceChannelBatch::new(client(), setup, initial)?;
    let forcing = PoolForcing::default();
    let batch1 = step(&batch0, 1.0, &forcing)?;
    check(&batch1, &[])?;
    assert_eq!(batch1.time(), 0.5);
    let batch2 = step(&batch1, 0.5, &forcing)?;
    check(&batch2, &[])?;
    assert_eq!(batch2.time(), 1.0);
    Ok(())
}

/// Restart on and after a pool curve epoch (including when initial volume exceeded the shrunk
/// curve's top knot) matches the uninterrupted run bit-for-bit; forged `pool_curve_epochs` rejects.
#[test]
fn pool_curve_epoch_restart_and_forged_epoch_rejection() -> Result {
    // Pond starts at vol=15.0 on curve [0, 20] (stage=1.5), drains over weir (crest=0.5) below
    // vol=12.0 by t=0.5 s. At t=0.5 s, curve shrinks to max_vol=14.0 (< initial_volume_m3=15.0!).
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 40.0, 3.0)?, 0.0),
        (pond("upper", 0.0, 10.0, 2.0)?, 15.0),
    ];
    case.structures = vec![structure(
        "weir",
        weir_law(0.5, 2.0)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("upper"), 0.0),
        (pool("lower"), 0.0),
    )];
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = 0.5;
    setup.pool_curve_schedules = vec![PoolCurveSchedule {
        schedule_id: "shrink-upper".into(),
        pool_id: "upper".into(),
        epochs: vec![PoolCurveEpoch {
            time_s: 0.5,
            curve: linear_curve(0.0, 7.0, 2.0)?,
            evidence: "Shrink capacity to 14 m3 after initial drainage".into(),
        }],
    }];

    let fresh = || SurfaceChannelBatch::new(client(), setup.clone(), initial.clone());
    let whole = advance_to(fresh()?, 1.5)?;
    let whole_data = whole.snapshot_data()?;

    // Checkpoints on the event (0.5 s) and after the event (1.0 s).
    for checkpoint_t in [0.5, 1.0] {
        let mid = advance_to(fresh()?, checkpoint_t)?;
        let snap = mid.snapshot();
        let data = mid.snapshot_data()?;

        let restored = fresh()?.restore(&snap)?;
        let imported = fresh()?.import_snapshot(&data)?;
        let resumed =
            SurfaceChannelBatch::resume(client(), &data, data.surface.setup.forcing.clone())?;

        for (label, batch) in [
            ("restore", restored),
            ("import", imported),
            ("resume", resumed),
        ] {
            let continued = advance_to(batch, 1.5)?;
            assert_eq!(
                continued.snapshot_data()?,
                whole_data,
                "{label} from t={checkpoint_t} must match uninterrupted run bit-for-bit"
            );
        }

        // Forged pool_curve_epochs must reject on both import_snapshot and resume.
        let mut forged = data.clone();
        forged.pool_curve_epochs[0] += 1;
        for result in [
            fresh()?.import_snapshot(&forged),
            SurfaceChannelBatch::resume(client(), &forged, forged.surface.setup.forcing.clone()),
        ] {
            match result {
                Err(SurfaceChannelError::Input(msg)) => {
                    assert!(
                        msg.contains("pool_curve_epochs"),
                        "expected pool_curve_epochs in {msg:?}"
                    );
                }
                other => return Err(format!("expected Input error, got {other:?}").into()),
            }
        }
    }
    Ok(())
}
