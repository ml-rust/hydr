//! Snapshot, restore, import, resume, and forecast continuation reach the uninterrupted bits.

use crate::fixture::Result;
use crate::scenario::{Batch, ReachSpec, Spec, advance, step, weir};
use hydr::coupling::SurfaceChannelError;
use hydr::solver::{ForcingInterval, ForcingSchedule, Solver};

/// Two reaches beside a spilling surface; one link reverses, one spills.
pub fn network() -> Spec {
    Spec::new(3, 2, 1.0, 0.4)
        .reach(ReachSpec::new("north", 0.0, &[0.5, 0.5, 0.5]))
        .reach(ReachSpec::new("south", 0.0, &[1.5, 1.5]))
        .link(weir("n-0", 0, "north", 0, 1.2, 2.0))
        .link(weir("n-1", 1, "north", 2, 1.2, 2.0))
        .link(weir("s-0", 3, "south", 0, 1.2, 2.0))
        .link(weir("s-1", 4, "south", 1, 1.2, 2.0))
}

/// A surface at 1.4 m spills freely into a reach at 1.15 m, below a 1.2 m crest.
///
/// The reach rises above the crest during the third 0.2 s interval. That interval's first
/// update changes the weir regime, so a second update must repeat it.
pub fn spill() -> Spec {
    Spec::new(1, 1, 1.0, 0.4)
        .reach(ReachSpec::new("river", 0.0, &[1.15, 1.15]))
        .link(weir("bank", 0, "river", 0, 1.2, 2.0))
}

/// Accepts one interval toward each target in turn.
pub fn run(mut batch: Batch, targets: &[f64]) -> Result<Batch> {
    for &target in targets {
        batch = step(&batch, target)?;
    }
    Ok(batch)
}

/// Debug text of the package, the composite ledger, and the live surface state.
///
/// Debug text shows every bit of finite values, signed zero included.
pub fn state(batch: &Batch) -> Result<String> {
    Ok(format!(
        "{} {} {:?} {:?} {:?}",
        batch.time(),
        batch.accepted_steps(),
        batch.snapshot_data()?,
        batch.ledger()?,
        batch.surface().state()?
    ))
}

/// Fails unless `setup().surface.forcing` is the window the surface engine steps with.
fn window_is_current(batch: &Batch) -> Result {
    if batch.setup().surface.forcing != *batch.surface().coupling_forcing() {
        return Err("setup().surface.forcing differs from the surface engine window".into());
    }
    Ok(())
}

const FIRST: [f64; 2] = [0.25, 0.5];
const SECOND: [f64; 2] = [0.75, 1.0];

#[test]
fn snapshot_restore_import_resume_are_bitwise() -> Result {
    let spec = network();
    let whole = run(run(spec.build()?, &FIRST)?, &SECOND)?;
    let exchanged: f64 = whole
        .link_ledgers()?
        .iter()
        .map(|l| l.surface_to_channel_m3 + l.channel_to_surface_m3)
        .sum();
    assert!(exchanged > 0.0, "the network must exchange water");
    let middle = run(spec.build()?, &FIRST)?;
    let data = middle.snapshot_data()?;
    let expected = state(&middle)?;
    let restored = spec.build()?.restore(&middle.snapshot())?;
    let imported = spec.build()?.import_snapshot(&data)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    window_is_current(&resumed)?;
    for (name, batch) in [
        ("restored", restored),
        ("imported", imported),
        ("resumed", resumed),
    ] {
        assert_eq!(state(&batch)?, expected, "{name} at the checkpoint");
        assert_eq!(batch.snapshot_data()?, data, "{name} package");
        assert_eq!(
            state(&run(batch, &SECOND)?)?,
            state(&whole)?,
            "{name} after continuing"
        );
    }
    let origin = spec.build()?;
    let zero = origin.snapshot_data()?;
    assert!(zero.last_interval.is_none());
    let imported = spec.build()?.import_snapshot(&zero)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &zero,
        zero.surface.setup.forcing.clone(),
    )?;
    for batch in [imported, resumed] {
        assert_eq!(state(&batch)?, state(&origin)?);
        assert_eq!(state(&run(run(batch, &FIRST)?, &SECOND)?)?, state(&whole)?);
    }
    Ok(())
}

/// Uniform rain over every surface cell for `[start, end)`.
fn rain(count: usize, start: f64, end: f64, rate: f64) -> ForcingInterval {
    ForcingInterval {
        start,
        end,
        rain: vec![rate; count],
        potential_evaporation: vec![0.0; count],
        withdrawal: vec![0.0; count],
        boundaries: Vec::new(),
    }
}

#[test]
fn continue_surface_forcing_matches_spliced_run() -> Result {
    let mut spec = network();
    spec.breaks = vec![0.0, 0.5, 2.0];
    let (setup, initial) = spec.parts()?;
    let domain = setup.surface.domain.clone();
    let boundaries = setup.surface.boundaries.clone();
    let count = domain.grid().cell_count();
    let forecast = vec![rain(count, 0.5, 1.0, 2e-3), rain(count, 1.0, 2.0, 4e-3)];
    let mut spliced = vec![rain(count, 0.0, 0.5, 0.0)];
    spliced.extend(forecast.iter().cloned());
    let mut spliced_setup = setup.clone();
    spliced_setup.surface.forcing = ForcingSchedule::new(&domain, &boundaries, spliced)?;
    let fresh = || {
        Batch::new(
            crate::runtime::client(),
            spliced_setup.clone(),
            initial.clone(),
        )
    };
    let boundary = 0.5;
    let cycle = advance(
        Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?,
        boundary,
    )?;
    assert_eq!(
        cycle.time(),
        boundary,
        "the cycle ends on the interval boundary"
    );
    let before = state(&cycle)?;
    let rewritten = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![rain(count, 0.0, 0.5, 1e-3), rain(count, 0.5, 2.0, 0.0)],
    )?;
    assert!(matches!(
        cycle.continue_surface_forcing(rewritten),
        Err(SurfaceChannelError::Surface(_))
    ));
    let continued =
        cycle.continue_surface_forcing(ForcingSchedule::new(&domain, &boundaries, forecast)?)?;
    assert_eq!(
        state(&cycle)?,
        before,
        "continuation leaves the batch unchanged"
    );
    let reference = advance(fresh()?, boundary)?;
    assert_eq!(state(&continued)?, state(&reference)?);
    window_is_current(&continued)?;
    assert!(cycle.setup().surface.forcing != continued.setup().surface.forcing);
    let checkpoint = continued.snapshot_data()?;
    let imported = fresh()?.import_snapshot(&checkpoint)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &checkpoint,
        checkpoint.surface.setup.forcing.clone(),
    )?;
    window_is_current(&resumed)?;
    // Steps toward 1.25 clip at the forecast boundary 1.0 in every run.
    let expected = state(&advance(reference, 1.25)?)?;
    for (name, batch) in [
        ("continued", continued),
        ("imported", imported),
        ("resumed", resumed),
    ] {
        let advanced = advance(batch, 1.25)?;
        window_is_current(&advanced)?;
        assert_eq!(state(&advanced)?, expected, "{name}");
    }
    Ok(())
}

#[test]
fn standalone_surface_restore_of_coupled_package_rejects() -> Result {
    let spec = network();
    let batch = run(spec.build()?, &FIRST)?;
    let data = batch.snapshot_data()?;
    assert!(data.surface.ledger.exchange_in_m3 + data.surface.ledger.exchange_out_m3 > 0.0);
    let coordinator = "restore this package through the surface-channel coordinator";
    let (setup, initial) = spec.parts()?;
    let mut standalone = Solver::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.surface.clone(),
        initial.surface.clone(),
    )?;
    let messages = [
        standalone.restore_data(&data.surface).err(),
        standalone.restore(&batch.surface().snapshot()).err(),
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
    assert_eq!(spec.build()?.import_snapshot(&data)?.snapshot_data()?, data);
    Ok(())
}
