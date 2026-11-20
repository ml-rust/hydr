//! Ported checkpoints: bitwise round trips, restart against the uninterrupted run, identity
//! rejection of earlier package formats, and forged port data.

use crate::fixture::{Result, contains};
use crate::ports_fixture::{Batch, Reach, Scenario, step, west};
use hydr::coupling::{
    SURFACE_CHANNEL_CONTRACT_VERSION, SURFACE_CHANNEL_SNAPSHOT_SCHEMA, SurfaceChannelError,
    SurfaceChannelIntervalSnapshotData, SurfaceChannelPortRegime, SurfaceChannelSnapshotData,
};
use hydr::solver::Solver;

/// Requested duration of every interval, in s.
const DT: f64 = 0.25;
/// Intervals before the checkpoint, and after it. The port runs both ways before the end.
pub(crate) const HALF: usize = 40;

/// A reach above its floodplain drains through a port and a bank link, then refills.
fn case() -> Scenario {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
    case.alpha = Reach::rest(0.5, 1.75);
    case
}

/// `batch` after `count` more intervals.
pub(crate) fn run(mut batch: Batch, count: usize) -> Result<Batch> {
    for _ in 0..count {
        batch = step(&batch, DT)?;
    }
    Ok(batch)
}

/// Debug text of the package, the ledgers, and the live surface state.
///
/// Debug text shows every bit of finite values, signed zero included.
pub(crate) fn state(batch: &Batch) -> Result<String> {
    Ok(format!(
        "{} {} {:?} {:?} {:?} {:?} {:?}",
        batch.time(),
        batch.accepted_steps(),
        batch.snapshot_data()?,
        batch.ledger()?,
        batch.port_ledgers()?,
        batch.link_ledgers()?,
        batch.surface().state()?
    ))
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
fn ported_snapshots_round_trip_and_restart_bit_for_bit() -> Result {
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
    // A zero-history package restarts the whole run.
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
fn a_reversal_flag_replays_from_the_start_frame_counters() -> Result {
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
fn earlier_package_identities_reject() -> Result {
    assert_eq!(
        SURFACE_CHANNEL_SNAPSHOT_SCHEMA,
        "hydr-surface-channel-checkpoint-5"
    );
    assert_eq!(
        SURFACE_CHANNEL_CONTRACT_VERSION,
        "hydr-surface-channel-numr-cpu-7"
    );
    let data = package()?;
    for found in [
        "hydr-surface-channel-checkpoint-3",
        "hydr-surface-channel-checkpoint-4",
    ] {
        let mut earlier = data.clone();
        earlier.schema = found.into();
        rejects(
            &earlier,
            &[
                "surface-channel checkpoint schema",
                &format!("found \"{found}\""),
                "capture a new package",
            ],
        )?;
    }
    for found in [
        "hydr-surface-channel-numr-cpu-4",
        "hydr-surface-channel-numr-cpu-5",
        "hydr-surface-channel-numr-cpu-6",
    ] {
        let mut earlier = data.clone();
        earlier.contract = found.into();
        rejects(
            &earlier,
            &[
                "surface-channel checkpoint contract",
                &format!("found \"{found}\""),
                "restore it with the hydr build that produced it",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn forged_port_receipts_reject() -> Result {
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
fn forged_port_ledgers_and_setup_ports_reject() -> Result {
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
    let mut forged = data.clone();
    forged.counters.ports.surface_in_m3 *= 1.5;
    rejects(&forged, &["counters.ports.surface_in_m3"])?;
    let mut forged = data.clone();
    last_interval(&mut forged)?.previous.counters.ports.ledgers[0].surface_to_channel_m3 = -1.0;
    rejects(
        &forged,
        &[
            "checkpoint last_interval.previous.counters.ports.ledgers[0] (\"p\")",
            "holds a negative or nonfinite value",
        ],
    )?;
    let mut forged = data.clone();
    last_interval(&mut forged)?.previous.counters.ports.ledgers[0].port_id = "q".into();
    rejects(
        &forged,
        &[
            "checkpoint last_interval.previous.counters.ports.ledgers ",
            "differs from the ports in ascending identity",
        ],
    )?;
    // Changed port evidence differs from the setup the destination holds.
    let mut forged = data.clone();
    forged.setup.ports[0].mapping_evidence = "Another junction".into();
    contains(
        &import_rejection(&forged)?,
        &[
            "checkpoint setup.ports[0] (\"p\")",
            "differs from the destination setup",
        ],
    )?;
    // Without its port the package's port water stays unexplained on resume.
    let mut forged = data;
    forged.setup.ports.clear();
    contains(
        &import_rejection(&forged)?,
        &[
            "checkpoint setup.ports",
            "differs from the destination setup",
        ],
    )?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &forged,
        forged.surface.setup.forcing.clone(),
    );
    match resumed {
        Err(SurfaceChannelError::Surface(error)) => {
            contains(&error.to_string(), &["source exchange volume"])
        }
        other => Err(format!("expected a surface rejection, got {other:?}").into()),
    }
}

/// Input message of import of `data` into a fresh batch.
fn import_rejection(data: &SurfaceChannelSnapshotData) -> Result<String> {
    match case().build()?.import_snapshot(data) {
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        other => Err(format!("expected an input rejection, got {other:?}").into()),
    }
}

#[test]
fn standalone_surface_restore_of_a_ported_package_rejects() -> Result {
    let batch = run(case().build()?, HALF)?;
    let data = batch.snapshot_data()?;
    assert!(data.surface.ledger.exchange_in_m3 + data.surface.ledger.exchange_out_m3 > 0.0);
    let coordinator = "restore this package through the surface-channel coordinator";
    let (setup, initial) = case().parts()?;
    let mut standalone = Solver::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.surface.clone(),
        initial.surface.clone(),
    )?;
    let messages = [
        standalone.restore_data(&data.surface).err(),
        Solver::<crate::runtime::Runtime>::resume(
            crate::runtime::client(),
            &data.surface,
            data.surface.setup.forcing.clone(),
        )
        .err(),
    ];
    for message in messages {
        let message = message.map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains(coordinator), "{message}");
    }
    assert_eq!(
        standalone.time(),
        0.0,
        "rejection leaves the engine unchanged"
    );
    Ok(())
}
