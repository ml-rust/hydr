//! Common-interval lifecycle: determinism, rollback, shortening, and recomputation.

use crate::fixture::Result;
use crate::restoration::spill;
use crate::scenario::{Batch, ReachSpec, Spec, depths, forcing, last, step, volumes, weir};
use hydr::channel::ChannelLateralInput;
use hydr::coupling::SurfaceChannelError;
use hydr::forcing::WaterAmount;

/// Two reaches beside a spilling surface; one link reverses, one spills.
fn network() -> Spec {
    Spec::new(3, 2, 1.0, 0.4)
        .reach(ReachSpec::new("north", 0.0, &[0.5, 0.5, 0.5]))
        .reach(ReachSpec::new("south", 0.0, &[1.5, 1.5]))
        .link(weir("n-0", 0, "north", 0, 1.2, 2.0))
        .link(weir("n-1", 1, "north", 2, 1.2, 2.0))
        .link(weir("s-0", 3, "south", 0, 1.2, 2.0))
        .link(weir("s-1", 4, "south", 1, 1.2, 2.0))
}

/// Debug text of every accepted observable, which shows every bit of finite values.
fn observables(batch: &Batch) -> Result<String> {
    let mut text = format!(
        "{} {} {:?} {:?} {:?} {:?}",
        batch.time(),
        batch.accepted_steps(),
        depths(batch)?,
        batch.link_ledgers()?,
        batch.last_step()?,
        batch.ledger()?
    );
    for id in ["north", "south"] {
        text.push_str(&format!(" {id}={:?}", volumes(batch, id)?));
        let reach = batch.reach(id).ok_or("missing reach")?;
        text.push_str(&format!(" {:?}", reach.snapshot_data()?));
    }
    text.push_str(&format!(" {:?}", batch.surface().snapshot().data()?));
    Ok(text)
}

fn run(spec: &Spec, steps: usize) -> Result<Batch> {
    let mut batch = spec.build()?;
    for _ in 0..steps {
        batch = step(&batch, batch.time() + 0.5)?;
    }
    Ok(batch)
}

#[test]
fn link_order_and_reach_order_are_bitwise_invariant() -> Result {
    let spec = network();
    let mut permuted = network();
    permuted.links.reverse();
    permuted.links.swap(0, 2);
    permuted.reaches.reverse();
    let base = run(&spec, 3)?;
    let exchanged: f64 = base
        .link_ledgers()?
        .iter()
        .map(|l| l.surface_to_channel_m3 + l.channel_to_surface_m3)
        .sum();
    assert!(exchanged > 0.0, "the network must exchange water");
    assert_eq!(observables(&base)?, observables(&run(&permuted, 3)?)?);
    Ok(())
}

#[test]
fn rejected_trial_leaves_batch_bitwise_unchanged() -> Result {
    let batch = run(&network(), 1)?;
    let before = observables(&batch)?;
    let candidate = step(&batch, batch.time() + 0.5)?;
    assert!(candidate.accepted_steps() == 2);
    drop(candidate);
    let stale = forcing(&batch, batch.time() + 0.5, &[]);
    let rejected = batch.trial_to(batch.time() - 0.25, &stale);
    assert!(matches!(rejected, Err(SurfaceChannelError::Input(_))));
    assert_eq!(observables(&batch)?, before);
    let again = step(&batch, batch.time() + 0.5)?;
    let repeat = step(&batch, batch.time() + 0.5)?;
    assert_eq!(observables(&again)?, observables(&repeat)?);
    Ok(())
}

#[test]
fn nonconvergence_halves_common_interval() -> Result {
    // One update exhausts the third spill interval.
    let mut spec = spill();
    spec.config.iteration_limit = 1;
    let mut converged = spill();
    converged.config.iteration_limit = 64;
    let (mut limited, mut full) = (spec.build()?, converged.build()?);
    for _ in 0..2 {
        limited = step(&limited, limited.time() + 0.2)?;
        full = step(&full, full.time() + 0.2)?;
        assert_eq!(last(&limited)?, last(&full)?);
    }
    let start = full.time();
    let requested = start + 0.2;
    let reference = last(&step(&full, requested)?)?;
    assert_eq!((reference.retries, reference.end_s), (0, requested));
    assert!(reference.iterations > 1, "{reference:?}");
    let halved = last(&step(&limited, requested)?)?;
    assert!(halved.retries > 0, "{halved:?}");
    let mut end = requested;
    for _ in 0..halved.retries {
        end = start + 0.5 * (end - start);
    }
    assert_eq!(halved.end_s.to_bits(), end.to_bits(), "{halved:?}");
    assert!(halved.iterations <= 1);
    Ok(())
}

#[test]
fn integrated_lateral_requires_recompute() -> Result {
    let spec = Spec::new(2, 1, 1.0, 0.0)
        .reach(ReachSpec::new("river", 0.0, &[0.5, 0.5]).binding(
            "tributary",
            1,
            hydr::channel::ChannelLateralDirection::Inflow,
        ))
        .link(weir("bank", 0, "river", 0, 1.2, 2.0));
    let batch = spec.build()?;
    let inflow = ChannelLateralInput::Inflow {
        boundary_id: "tributary".into(),
        amount: WaterAmount::IntervalVolume { volume_m3: 0.5 },
        longitudinal_velocity_m_per_s: 0.0,
    };
    let requested = 2.0;
    let interval = forcing(&batch, requested, &[("river", inflow.clone())]);
    match batch.trial_to(requested, &interval) {
        Err(SurfaceChannelError::RecomputeInterval {
            start_s,
            requested_end_s,
            suggested_end_s,
            ..
        }) => {
            assert_eq!((start_s, requested_end_s), (0.0, requested));
            assert_eq!(suggested_end_s, batch.setup().config.maximum_step_s);
        }
        Err(error) => return Err(format!("wrong error: {error}").into()),
        Ok(_) => return Err("an integrated span past maximum_step_s must recompute".into()),
    }
    let longer = forcing(&batch, 3.0, &[("river", inflow.clone())]);
    assert!(matches!(
        batch.trial_to(0.5, &longer),
        Err(SurfaceChannelError::Input(_))
    ));
    let exact = forcing(&batch, 0.25, &[("river", inflow)]);
    let next = batch.trial_to(0.25, &exact)?.into_next();
    assert_eq!(next.time(), 0.25);
    let total: f64 = volumes(&next, "river")?.iter().sum();
    assert!((total - 2.5).abs() <= 1e-12, "{total}");
    Ok(())
}

#[test]
fn surface_interval_end_clips_common_step() -> Result {
    let mut spec = network();
    spec.breaks = vec![0.0, 0.1, 100.0];
    let batch = step(&spec.build()?, 0.9)?;
    let first = last(&batch)?;
    assert_eq!((first.end_s, first.retries), (0.1, 0));
    let next = step(&batch, 0.2)?;
    assert_eq!((last(&next)?.start_s, next.time()), (0.1, 0.2));
    Ok(())
}

#[test]
fn head_crossing_levels_without_shortening() -> Result {
    // A long, efficient bank between a high floodplain and a channel just above the crest.
    // The weir volume would lift the channel above the floodplain. The cap levels both instead.
    let spec = Spec::new(1, 1, 1.0, 0.5)
        .reach(ReachSpec::new("river", 0.0, &[1.25, 1.25]))
        .link(weir("bank", 0, "river", 0, 1.2, 400.0));
    let batch = step(&spec.build()?, 0.2)?;
    let diagnostics = last(&batch)?;
    assert_eq!((diagnostics.retries, diagnostics.end_s), (0, 0.2));
    let link = &diagnostics.links[0];
    assert!(
        link.level_factor < 1.0 && link.signed_volume_m3 > 0.0,
        "{link:?}"
    );
    assert!(link.signed_volume_m3 < link.weir_volume_m3, "{link:?}");
    let surface = 1.0 + depths(&batch)?[0];
    let channel = volumes(&batch, "river")?[0] / 2.0;
    // Storage 6 m³ over 4 + 2 m² of plan area levels at 17/12 m.
    let level = 17.0 / 12.0;
    let budget = spec.config.head_absolute_m + spec.config.head_relative * level;
    assert!(
        surface + budget >= channel,
        "surface={surface}, channel={channel}"
    );
    assert!((surface - level).abs() <= 1e-12, "surface={surface}");
    assert!((channel - level).abs() <= 1e-12, "channel={channel}");
    Ok(())
}
