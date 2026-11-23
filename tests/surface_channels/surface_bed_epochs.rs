//! Surface/channel coordinator surface bed epochs: lake at rest + bed step starting flow with
//! exact volume conservation, admission rejections, dry invariance, and bitwise restart with
//! forged `surface_bed_epochs` rejection.

use crate::fixture::Result;
use crate::runtime::client;
use crate::scenario::{Batch, ReachSpec, Spec, advance, step, weir};
use hydr::coupling::{SurfaceChannelBatch, SurfaceChannelDefinition, SurfaceChannelError};
use hydr::solver::{SurfaceBedEpoch, SurfaceBedSchedule, SurfaceBedUpdate};

/// Lake at rest on a 2x1 surface beside a reach, then a bed step at t = 0.5 s raises cell 0:
/// volume stays bit-identical across the event, flow starts across both the surface and the bank
/// weir, and composite water balance holds.
#[test]
fn lake_at_rest_bed_step_starts_flow_and_conserves_volume() -> Result {
    let mut spec = Spec::new(2, 1, 0.0, 1.0)
        .reach(ReachSpec::new("river", 0.0, &[0.5, 0.5]))
        .link(weir("bank", 0, "river", 0, 1.5, 2.0));
    spec.config.maximum_step_s = 1.0;
    let (mut setup, initial) = spec.parts()?;
    setup.surface_bed_schedules = vec![SurfaceBedSchedule {
        schedule_id: "uplift-cell-0".into(),
        epochs: vec![SurfaceBedEpoch {
            time_s: 0.5,
            updates: vec![SurfaceBedUpdate {
                cell: 0,
                bed_elevation_m: 0.6,
            }],
            evidence: "Coseismic uplift of cell 0".into(),
        }],
    }];
    let batch0 = SurfaceChannelBatch::new(client(), setup, initial)?;

    // Advance up to the 0.5 s bed event: no interval straddles 0.5 s and lake stays at rest.
    let mut batch1 = batch0;
    while batch1.time() < 0.5 {
        batch1 = step(&batch1, 1.0)?;
        assert!(batch1.time() <= 0.5, "interval straddled event time 0.5 s");
    }
    assert_eq!(batch1.time(), 0.5);
    let cells1 = batch1.surface().state()?;
    assert_eq!(cells1.cells()[0].h, 1.0);
    assert_eq!(cells1.cells()[1].h, 1.0);
    let d1 = batch1.last_step()?.ok_or("missing first step")?;
    assert_eq!(d1.links[0].signed_volume_m3.to_bits(), 0.0_f64.to_bits());

    // After t = 0.5 s, cell 0 stage jumps to 1.6 > crest 1.5 and > cell 1 stage 1.0.
    let batch2 = step(&batch1, 0.75)?;
    let d2 = batch2.last_step()?.ok_or("missing second step")?;
    assert_eq!(d2.geometry.len(), 1);
    assert_eq!(d2.geometry[0].element_id, "surface");
    assert_eq!(
        d2.geometry[0].storage_before_m3.to_bits(),
        d2.geometry[0].storage_after_m3.to_bits()
    );
    assert!(
        d2.links[0].signed_volume_m3 > 0.0,
        "lifted cell 0 must spill over bank crest 1.5"
    );
    let cells2 = batch2.surface().state()?;
    assert!(
        cells2.cells()[0].h < 1.0 && cells2.cells()[1].h > 1.0,
        "surface flow must start from lifted cell 0 into cell 1"
    );
    let ledger = batch2.ledger()?;
    assert!(
        ledger.residual_m3.abs() <= ledger.bound_m3,
        "composite balance violated: {ledger:?}"
    );
    Ok(())
}

/// Admission rejects a surface bed epoch that lifts a cell above its bank crest or targets an
/// out-of-range cell.
#[test]
fn bed_above_bank_crest_or_out_of_range_rejects_at_admission() -> Result {
    let spec = Spec::new(2, 1, 0.0, 0.5)
        .reach(ReachSpec::new("river", 0.0, &[0.5, 0.5]))
        .link(weir("bank", 0, "river", 0, 1.0, 2.0));
    let (mut setup, _) = spec.parts()?;
    setup.surface_bed_schedules = vec![SurfaceBedSchedule {
        schedule_id: "too-high".into(),
        epochs: vec![SurfaceBedEpoch {
            time_s: 0.5,
            updates: vec![SurfaceBedUpdate {
                cell: 0,
                bed_elevation_m: 1.2,
            }],
            evidence: "Bed above crest 1.0".into(),
        }],
    }];
    match SurfaceChannelDefinition::new(setup) {
        Err(SurfaceChannelError::Input(msg)) => {
            assert!(
                msg.contains("unobstructed-connection"),
                "expected unobstructed-connection, got {msg:?}"
            );
        }
        other => return Err(format!("expected admission error, got {other:?}").into()),
    }

    let (mut setup, _) = spec.parts()?;
    setup.surface_bed_schedules = vec![SurfaceBedSchedule {
        schedule_id: "bad-cell".into(),
        epochs: vec![SurfaceBedEpoch {
            time_s: 0.5,
            updates: vec![SurfaceBedUpdate {
                cell: 99,
                bed_elevation_m: 0.2,
            }],
            evidence: "Out-of-range cell".into(),
        }],
    }];
    assert!(matches!(
        SurfaceChannelDefinition::new(setup),
        Err(SurfaceChannelError::Input(_))
    ));
    Ok(())
}

/// A bed step on a dry surface moves exact zero water.
#[test]
fn dry_surface_bed_step_moves_zero_water() -> Result {
    let spec = Spec::new(2, 1, 0.0, 0.0)
        .reach(ReachSpec::new("river", 0.0, &[0.0, 0.0]))
        .link(weir("bank", 0, "river", 0, 1.0, 2.0));
    let (mut setup, initial) = spec.parts()?;
    setup.surface_bed_schedules = vec![SurfaceBedSchedule {
        schedule_id: "dry-step".into(),
        epochs: vec![SurfaceBedEpoch {
            time_s: 0.5,
            updates: vec![SurfaceBedUpdate {
                cell: 0,
                bed_elevation_m: 0.4,
            }],
            evidence: "Dry uplift".into(),
        }],
    }];
    let batch = advance(SurfaceChannelBatch::new(client(), setup, initial)?, 1.0)?;
    let cells = batch.surface().state()?;
    for cell in cells.cells() {
        assert_eq!(cell.h.to_bits(), 0.0_f64.to_bits());
    }
    let d = batch.last_step()?.ok_or("missing diagnostics")?;
    assert_eq!(d.links[0].signed_volume_m3.to_bits(), 0.0_f64.to_bits());
    Ok(())
}

/// Restart on and after a surface bed epoch matches the uninterrupted run bit-for-bit; forged
/// `surface_bed_epochs` rejects on import and resume.
#[test]
fn surface_bed_epoch_restart_and_forged_epoch_rejection() -> Result {
    let mut spec = Spec::new(2, 1, 0.0, 1.0)
        .reach(ReachSpec::new("river", 0.0, &[0.5, 0.5]))
        .link(weir("bank", 0, "river", 0, 1.5, 2.0));
    spec.config.maximum_step_s = 0.25;
    let (mut setup, initial) = spec.parts()?;
    setup.surface_bed_schedules = vec![SurfaceBedSchedule {
        schedule_id: "uplift-restart".into(),
        epochs: vec![SurfaceBedEpoch {
            time_s: 0.5,
            updates: vec![SurfaceBedUpdate {
                cell: 0,
                bed_elevation_m: 0.6,
            }],
            evidence: "Uplift at 0.5 s".into(),
        }],
    }];

    let fresh = || SurfaceChannelBatch::new(client(), setup.clone(), initial.clone());

    for checkpoint_t in [0.5, 0.75] {
        let mid = advance(fresh()?, checkpoint_t)?;
        let snap = mid.snapshot();
        let data = mid.snapshot_data()?;
        let whole = advance(mid, 1.0)?;
        let whole_data = whole.snapshot_data()?;

        let restored = fresh()?.restore(&snap)?;
        let imported = fresh()?.import_snapshot(&data)?;
        let resumed = Batch::resume(client(), &data, data.surface.setup.forcing.clone())?;

        for (label, batch) in [
            ("restore", restored),
            ("import", imported),
            ("resume", resumed),
        ] {
            let continued = advance(batch, 1.0)?;
            assert_eq!(
                continued.snapshot_data()?,
                whole_data,
                "{label} from t={checkpoint_t} must match uninterrupted run bit-for-bit"
            );
        }

        let mut forged = data.clone();
        forged.surface_bed_epochs[0] += 1;
        for result in [
            fresh()?.import_snapshot(&forged),
            Batch::resume(client(), &forged, forged.surface.setup.forcing.clone()),
        ] {
            match result {
                Err(SurfaceChannelError::Input(msg)) => {
                    assert!(
                        msg.contains("surface_bed_epochs"),
                        "expected surface_bed_epochs in {msg:?}"
                    );
                }
                other => return Err(format!("expected Input error, got {other:?}").into()),
            }
        }
    }
    Ok(())
}
