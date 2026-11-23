//! Surface/channel coordinator channel section epochs: narrowed section accept/reject against
//! bank limit, dry invariance, and bitwise restart with forged `channel_section_epochs` rejection.

use crate::fixture::Result;
use crate::runtime::client;
use crate::scenario::{Batch, ReachSpec, Spec, advance, step, weir};
use hydr::channel::{
    ChannelGeometry, ChannelSectionEpoch, ChannelSectionSchedule, RectangularChannelCell,
};
use hydr::coupling::{SurfaceChannelBatch, SurfaceChannelError};

fn rect_cells(width_m: f64, max_depth_m: f64) -> ChannelGeometry {
    ChannelGeometry::Rectangular {
        cells: (0..2)
            .map(|i| RectangularChannelCell {
                id: format!("cell-{i}"),
                length_m: 2.0,
                width_m,
                bed_elevation_m: 0.0,
                maximum_depth_m: max_depth_m,
                manning_n: 0.0,
            })
            .collect(),
    }
}

/// Narrowing a reach section at t = 0.5 s conserves wet area and volume bit-for-bit, raising
/// stage over the bank crest to spill onto the surface; narrowing past the bank limit rejects.
#[test]
fn narrowed_section_accepts_below_bank_and_rejects_above() -> Result {
    // Reach width starts at 1.0 (from ReachSpec default) with depth 0.8 -> area = 0.8 m2.
    // Narrowing to width 0.5 doubles depth to 1.6 > bank crest 1.2!
    {
        let mut spec = Spec::new(1, 1, 0.0, 0.2)
            .reach(ReachSpec::new("river", 0.0, &[0.8, 0.8]))
            .link(weir("bank", 0, "river", 0, 1.2, 2.0));
        spec.config.maximum_step_s = 0.5;
        let (mut setup, initial) = spec.parts()?;
        setup.channel_section_schedules = vec![ChannelSectionSchedule {
            schedule_id: "narrow-river".into(),
            reach_id: "river".into(),
            epochs: vec![ChannelSectionEpoch {
                time_s: 0.5,
                geometry: rect_cells(0.5, 4.0),
                evidence: "Channel constriction at 0.5 s".into(),
            }],
        }];
        let mut batch = SurfaceChannelBatch::new(client(), setup, initial)?;
        while batch.time() < 0.5 {
            batch = step(&batch, 1.0)?;
            assert!(batch.time() <= 0.5, "interval straddled event time 0.5 s");
        }
        assert_eq!(batch.time(), 0.5);
        let d1 = batch.last_step()?.ok_or("missing pre-event step")?;
        assert_eq!(d1.links[0].signed_volume_m3.to_bits(), 0.0_f64.to_bits());

        // Next interval applies narrowed width 0.5: reach stage jumps 0.8 -> 1.6 > crest 1.2!
        let batch2 = step(&batch, 0.75)?;
        let d2 = batch2.last_step()?.ok_or("missing post-event step")?;
        assert_eq!(d2.geometry.len(), 1);
        assert_eq!(d2.geometry[0].element_id, "river");
        assert_eq!(
            d2.geometry[0].storage_before_m3.to_bits(),
            d2.geometry[0].storage_after_m3.to_bits()
        );
        assert!(
            d2.links[0].signed_volume_m3 < 0.0,
            "narrowed reach must spill into surface (negative signed volume)"
        );
        let ledger = batch2.ledger()?;
        assert!(ledger.residual_m3.abs() <= ledger.bound_m3);
    }

    // Same reach, but new section has max_depth = 1.2 < recovered depth 1.6 -> must reject.
    {
        let mut spec = Spec::new(1, 1, 0.0, 0.2)
            .reach(ReachSpec::new("river", 0.0, &[0.8, 0.8]))
            .link(weir("bank", 0, "river", 0, 1.2, 2.0));
        spec.config.maximum_step_s = 0.5;
        let (mut setup, initial) = spec.parts()?;
        setup.channel_section_schedules = vec![ChannelSectionSchedule {
            schedule_id: "too-shallow".into(),
            reach_id: "river".into(),
            epochs: vec![ChannelSectionEpoch {
                time_s: 0.5,
                geometry: rect_cells(0.5, 1.2),
                evidence: "Bank limit below recovered depth".into(),
            }],
        }];
        let mut batch = SurfaceChannelBatch::new(client(), setup, initial)?;
        while batch.time() < 0.5 {
            batch = step(&batch, 1.0)?;
        }
        match step(&batch, 0.75) {
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("channel-section-bank-limit"),
                    "expected channel-section-bank-limit, got {msg:?}"
                );
            }
            Ok(_) => return Err("expected bank limit rejection".into()),
        }
    }
    Ok(())
}

/// A section epoch on a dry reach moves exact zero water.
#[test]
fn dry_reach_section_epoch_moves_zero_water() -> Result {
    let spec = Spec::new(1, 1, 0.0, 0.0)
        .reach(ReachSpec::new("river", 0.0, &[0.0, 0.0]))
        .link(weir("bank", 0, "river", 0, 1.0, 2.0));
    let (mut setup, initial) = spec.parts()?;
    setup.channel_section_schedules = vec![ChannelSectionSchedule {
        schedule_id: "dry-narrow".into(),
        reach_id: "river".into(),
        epochs: vec![ChannelSectionEpoch {
            time_s: 0.5,
            geometry: rect_cells(0.5, 4.0),
            evidence: "Dry constriction".into(),
        }],
    }];
    let batch = advance(SurfaceChannelBatch::new(client(), setup, initial)?, 1.0)?;
    let d = batch.last_step()?.ok_or("missing diagnostics")?;
    assert_eq!(d.links[0].signed_volume_m3.to_bits(), 0.0_f64.to_bits());
    Ok(())
}

/// Restart on and after a channel section epoch matches the uninterrupted run bit-for-bit; forged
/// `channel_section_epochs` rejects on import and resume.
#[test]
fn channel_section_epoch_restart_and_forged_epoch_rejection() -> Result {
    let mut spec = Spec::new(1, 1, 0.0, 0.2)
        .reach(ReachSpec::new("river", 0.0, &[0.8, 0.8]))
        .link(weir("bank", 0, "river", 0, 1.2, 2.0));
    spec.config.maximum_step_s = 0.25;
    let (mut setup, initial) = spec.parts()?;
    setup.channel_section_schedules = vec![ChannelSectionSchedule {
        schedule_id: "narrow-restart".into(),
        reach_id: "river".into(),
        epochs: vec![ChannelSectionEpoch {
            time_s: 0.5,
            geometry: rect_cells(0.5, 4.0),
            evidence: "Narrow at 0.5 s".into(),
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
        forged.channel_section_epochs[0] += 1;
        for result in [
            fresh()?.import_snapshot(&forged),
            Batch::resume(client(), &forged, forged.surface.setup.forcing.clone()),
        ] {
            match result {
                Err(SurfaceChannelError::Input(msg)) => {
                    assert!(
                        msg.contains("channel_section_epochs"),
                        "expected channel_section_epochs in {msg:?}"
                    );
                }
                other => return Err(format!("expected Input error, got {other:?}").into()),
            }
        }
    }
    Ok(())
}
