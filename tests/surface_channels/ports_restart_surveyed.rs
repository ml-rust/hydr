//! Ported checkpoints on a surveyed rectangle: bitwise round trips, restart against the
//! uninterrupted run, and forged port data.
//!
//! Frozen criterion (plan section 6, "Coordinated restart"): the case of `ports_restart.rs`
//! with `alpha` a surveyed rectangle of the port width continues bit for bit after snapshot,
//! restore, import, resume, and continue. A forged port receipt or ledger rejects.

use crate::fixture::{Result, contains};
use crate::ports_fixture::{Batch, Reach, Scenario, step, west};
use crate::ports_restart::{HALF, run, state};
use hydr::coupling::{
    SurfaceChannelError, SurfaceChannelIntervalSnapshotData, SurfaceChannelPortRegime,
    SurfaceChannelSnapshotData,
};

/// Requested duration of every interval, in s.
const DT: f64 = 0.25;

/// A surveyed reach above its floodplain drains through a port and a bank link, then refills.
fn case() -> Scenario {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]).with_surveyed(true);
    case.alpha = Reach::rest(0.5, 1.75);
    case
}

/// The checkpoint after `HALF` intervals.
fn package() -> Result<SurfaceChannelSnapshotData> {
    Ok(run(case().build()?, HALF)?.snapshot_data()?)
}

fn last_interval(
    data: &mut SurfaceChannelSnapshotData,
) -> Result<&mut SurfaceChannelIntervalSnapshotData> {
    Ok(data
        .last_interval
        .as_mut()
        .ok_or("the package lacks its last interval")?)
}

/// Input messages of import into a fresh batch and of resume, in that order.
fn rejections(data: &SurfaceChannelSnapshotData) -> Result<[String; 2]> {
    let message = |result: std::result::Result<Batch, SurfaceChannelError>| match result {
        Ok(_) => Err("the coordinator accepted a forged package".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    };
    let imported: Result<String> = message(case().build()?.import_snapshot(data));
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
        contains(&message, fragments)?;
    }
    Ok(())
}

#[test]
fn surveyed_ported_snapshots_round_trip_and_restart_bit_for_bit() -> Result {
    let whole = run(case().build()?, 2 * HALF)?;
    let ledger = &whole.port_ledgers()?[0];
    assert!(ledger.surface_to_channel_m3 > 0.0 && ledger.channel_to_surface_m3 > 0.0);
    let middle = run(case().build()?, HALF)?;
    let data = middle.snapshot_data()?;
    let interval = &data
        .last_interval
        .as_ref()
        .ok_or("a last interval")?
        .diagnostics;
    assert_eq!(interval.ports.len(), 1);
    assert_eq!(data.counters.ports.ledgers.len(), 1);
    assert_eq!(data.setup.ports, middle.setup().ports);
    let expected = state(&middle)?;
    let restored = case().build()?.restore(&middle.snapshot())?;
    let imported = case().build()?.import_snapshot(&data)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    let continued = middle.continue_surface_forcing(middle.setup().surface.forcing.clone())?;
    for (name, batch) in [
        ("restored", restored),
        ("imported", imported),
        ("resumed", resumed),
        ("continued", continued),
    ] {
        assert_eq!(state(&batch)?, expected, "{name} at the checkpoint");
        assert_eq!(batch.snapshot_data()?, data, "{name} package");
        assert_eq!(
            state(&run(batch, HALF)?)?,
            state(&whole)?,
            "{name} after continuing"
        );
    }
    Ok(())
}

#[test]
fn surveyed_zero_history_package_restarts_the_whole_run() -> Result {
    let whole = run(case().build()?, 2 * HALF)?;
    let origin = case().build()?;
    let zero = origin.snapshot_data()?;
    assert!(zero.last_interval.is_none());
    let imported = case().build()?.import_snapshot(&zero)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &zero,
        zero.surface.setup.forcing.clone(),
    )?;
    for batch in [imported, resumed] {
        assert_eq!(state(&batch)?, state(&origin)?);
        assert_eq!(state(&run(batch, 2 * HALF)?)?, state(&whole)?);
    }
    Ok(())
}

#[test]
fn surveyed_reversal_flag_replays_from_the_start_frame_counters() -> Result {
    // Find a checkpoint whose last interval reverses the port.
    let mut batch = case().build()?;
    let reversed = loop {
        batch = step(&batch, DT)?;
        let data = batch.snapshot_data()?;
        let last = data.last_interval.as_ref().ok_or("a last interval")?;
        if last.diagnostics.ports[0].reversed {
            break data;
        }
        if batch.accepted_steps() > 200 {
            return Err("the port never reversed".into());
        }
    };
    let imported = case().build()?.import_snapshot(&reversed)?;
    assert_eq!(imported.snapshot_data()?, reversed);
    // The start frame counters hold the previous signed volume. Zeroing it erases the flip.
    let mut forged = reversed.clone();
    last_interval(&mut forged)?
        .previous
        .counters
        .ports
        .last_signed_volume_m3[0] = 0.0;
    rejects(
        &forged,
        &[
            "last_interval.diagnostics.ports[0] (\"p\")",
            "differs from the recalculated last interval",
        ],
    )
}

#[test]
fn surveyed_forged_port_receipts_reject() -> Result {
    let data = package()?;
    let path = "last_interval.diagnostics.ports[0] (\"p\")";
    let replay = "differs from the recalculated last interval";
    let forge = |edit: fn(&mut SurfaceChannelSnapshotData)| -> Result {
        let mut forged = data.clone();
        edit(&mut forged);
        rejects(&forged, &[path, replay])
    };
    forge(|d| {
        if let Some(last) = &mut d.last_interval {
            let port = &mut last.diagnostics.ports[0];
            port.signed_volume_m3 = port.signed_volume_m3.next_up();
        }
    })?;
    forge(|d| {
        if let Some(last) = &mut d.last_interval {
            last.diagnostics.ports[0].junction_reaction_m4_per_s[1] += 1e-9;
        }
    })?;
    forge(|d| {
        if let Some(last) = &mut d.last_interval {
            let port = &mut last.diagnostics.ports[0];
            port.regime = if port.regime == SurfaceChannelPortRegime::Dry {
                SurfaceChannelPortRegime::Subcritical
            } else {
                SurfaceChannelPortRegime::Dry
            };
        }
    })?;
    forge(|d| {
        if let Some(last) = &mut d.last_interval {
            let port = &mut last.diagnostics.ports[0];
            port.reversed = !port.reversed;
        }
    })?;
    let mut forged = data.clone();
    last_interval(&mut forged)?.diagnostics.port_paired_bound_m3 *= 2.0;
    rejects(
        &forged,
        &["last_interval.diagnostics.port_paired_bound_m3", replay],
    )?;
    let mut forged = data;
    last_interval(&mut forged)?.diagnostics.ports.clear();
    rejects(&forged, &["last_interval.diagnostics.ports", replay])
}

#[test]
fn surveyed_forged_port_ledgers_reject() -> Result {
    let data = package()?;
    let mut forged = data.clone();
    let ledger = &mut forged.counters.ports.ledgers[0];
    ledger.channel_to_surface_m3 = ledger.channel_to_surface_m3.next_up();
    rejects(
        &forged,
        &[
            "counters.ports.ledgers[0] (\"p\").channel_to_surface_m3",
            "differs from the recalculated last interval",
        ],
    )?;
    let mut forged = data;
    forged.counters.ports.surface_in_m3 *= 1.5;
    rejects(&forged, &["counters.ports.surface_in_m3"])
}

