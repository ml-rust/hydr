//! Closed storages exchange through submerged banks until their heads level, then stay level.
//!
//! Walls close every storage and no source acts, so the exact level follows from total water
//! and plan area. The approach must be monotone and never cross. The levelling cap ends each
//! exchange in a finite number of intervals.

use crate::fixture::{Result, contains};
use crate::scenario::{Batch, ReachSpec, Spec, depths, forcing, last, volumes, weir};
use hydr::coupling::{SurfaceChannelError, SurfaceChannelWeirRegime};

const REACH: &str = "river";
const END: f64 = 3.0;
/// Most accepted intervals to `END`, rests at level included.
const MOST_INTERVALS: usize = 20;

/// Surface stages of every cell and the stage of channel cell 0.
fn heads(batch: &Batch, bed: f64) -> Result<(Vec<f64>, f64)> {
    let surface = depths(batch)?.iter().map(|h| bed + h).collect();
    // Every channel cell is 2 m long and 1 m wide on a bed at zero.
    Ok((surface, volumes(batch, REACH)?[0] / 2.0))
}

/// Advances `spec` to `END` and checks a monotone, conserved approach to `level`.
///
/// Returns the accepted interval count.
fn equilibrates(spec: &Spec, level: f64) -> Result<usize> {
    let config = &spec.config;
    let allowance = config.head_absolute_m + config.head_relative * level;
    let mut batch = spec.build()?;
    let (mut surface, mut channel) = heads(&batch, spec.bed)?;
    let mut steps = 0;
    let mut levelled_at = None;
    while batch.time() < END {
        batch = batch.trial_to(END, &forcing(&batch, END, &[]))?.into_next();
        steps += 1;
        let (next_surface, next_channel) = heads(&batch, spec.bed)?;
        for (before, after) in surface.iter().zip(&next_surface) {
            assert!(
                after - before <= allowance,
                "surface rises: {before} -> {after}"
            );
            assert!(
                *after + allowance >= next_channel,
                "heads cross: {after} < {next_channel}"
            );
        }
        assert!(
            next_channel + allowance >= channel,
            "channel falls: {channel} -> {next_channel}"
        );
        let step = last(&batch)?;
        assert!(
            step.mass_residual_m3.abs() <= step.mass_bound_m3,
            "{step:?}"
        );
        if step.links.iter().any(|link| link.level_factor < 1.0) {
            levelled_at.get_or_insert(steps);
        }
        if levelled_at.is_some_and(|at| at < steps) {
            for link in &step.links {
                assert_eq!(
                    link.signed_volume_m3, 0.0,
                    "level heads stay level: {link:?}"
                );
                assert_eq!(
                    link.regime,
                    SurfaceChannelWeirRegime::EqualHeads,
                    "{link:?}"
                );
            }
        }
        (surface, channel) = (next_surface, next_channel);
    }
    assert!(levelled_at.is_some(), "the cap must end the exchange");
    for head in surface.iter().chain([&channel]) {
        assert!(
            (head - level).abs() <= allowance,
            "head={head}, level={level}"
        );
    }
    let balance = batch.ledger()?;
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    assert_eq!(balance.external_in_m3, 0.0);
    assert_eq!(balance.external_out_m3, 0.0);
    Ok(steps)
}

/// A `nx` x 1 floodplain at stage 1.6 beside one reach cell at stage 1.3, behind 1.2 m crests.
fn closed(nx: usize) -> Spec {
    let mut spec = Spec::new(nx, 1, 1.0, 0.6).reach(ReachSpec::new(REACH, 0.0, &[1.3]));
    for cell in 0..nx {
        spec = spec.link(weir(&format!("bank-{cell}"), cell, REACH, 0, 1.2, 2.0));
    }
    spec
}

#[test]
fn submerged_exchange_equilibrates_to_level() -> Result {
    // Water 2.4 + 2.6 m³ over 4 + 2 m² of plan area levels at 1.5 m.
    let steps = equilibrates(&closed(1), 1.5)?;
    assert!(steps <= MOST_INTERVALS, "accepted intervals={steps}");
    Ok(())
}

#[test]
fn surveyed_exchange_equilibrates_to_level() -> Result {
    let mut spec = closed(1);
    spec.reaches[0] = ReachSpec::new(REACH, 0.0, &[1.3]).surveyed();
    // Certified area recovery resolves stage to this allowance.
    spec.config.head_absolute_m = 1e-10;
    spec.config.transfer_absolute_m3 = 1e-10;
    let steps = equilibrates(&spec, 1.5)?;
    assert!(steps <= MOST_INTERVALS, "accepted intervals={steps}");
    Ok(())
}

#[test]
fn levelled_interval_replays_and_rejects_a_forged_factor() -> Result {
    let spec = closed(1);
    let mut batch = spec.build()?;
    loop {
        if let Ok(step) = last(&batch)
            && step.links[0].level_factor < 1.0
        {
            break;
        }
        if batch.time() >= END {
            return Err("the cap never binds".into());
        }
        batch = batch.trial_to(END, &forcing(&batch, END, &[]))?.into_next();
    }
    let data = batch.snapshot_data()?;
    let restored = spec.build()?.import_snapshot(&data)?;
    assert_eq!(restored.snapshot_data()?, data);
    let mut forged = data.clone();
    let interval = forged
        .last_interval
        .as_mut()
        .ok_or("missing last interval")?;
    interval.diagnostics.links[0].level_factor = 1.0;
    match spec.build()?.import_snapshot(&forged) {
        Ok(_) => Err("the coordinator accepted a forged level factor".into()),
        Err(SurfaceChannelError::Input(message)) => contains(
            &message,
            &["last_interval.diagnostics.links[0] (\"bank-0\").level_factor"],
        ),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    }
}

#[test]
fn shared_channel_cell_equilibrates_to_level() -> Result {
    // Two floodplain cells feed one channel cell. Water 4.8 + 2.6 m³ over 8 + 2 m² levels at
    // 1.54 m. The cap holds for the summed inflow of both links.
    let steps = equilibrates(&closed(2), 1.54)?;
    assert!(steps <= MOST_INTERVALS, "accepted intervals={steps}");
    Ok(())
}
