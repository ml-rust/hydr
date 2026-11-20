//! Coordinated restart of a surveyed channel/pool setup.
//!
//! Frozen criterion (plan section 6, "Coordinated restart"): the trusted restore and the cold
//! import of a package taken after three intervals continue bit for bit as the uninterrupted
//! run does, and a forged shared transfer packet rejects. The cases are the frozen
//! rectangular surveyed case and the first three-band case.

use super::fixture::{Batch, Result, batch, forcing};
use super::reference::CASES;
use super::surveyed_fixture::{rest_depth, single, surveyed_case};
use super::surveyed_reference::{SurveyedCase, THREE_BANDS_CASES};
use hydr::channel::ChannelBoundary;
use hydr::coupling::{ChannelPoolEndpoint, ChannelPoolSnapshotData};

/// The forcing of an interval: the unconnected end holds the case at rest, with signed zeros.
fn held(
    batch: &Batch,
    case: &SurveyedCase,
    end_s: f64,
) -> hydr::coupling::ChannelPoolForcingInterval {
    let mut interval = forcing(batch.setup(), batch.time(), end_s);
    interval.reaches[0].left = Some(ChannelBoundary::Exterior {
        depth_m: rest_depth(case),
        velocity_m_per_s: -0.0,
    });
    interval
}

/// The initial batch and the batch after three intervals of one case.
fn history(case: &SurveyedCase) -> Result<(Batch, Batch)> {
    let (mut setup, initial) = single(case, ChannelPoolEndpoint::Right)?;
    setup.connections[0].entrance_loss_coefficient = -0.0;
    let origin = batch(setup, initial)?;
    let mut current = origin.clone();
    for index in 0..3 {
        let end = (index + 1) as f64 / 8.0;
        let interval = held(&current, case, end);
        current = current.trial_to(end, &interval)?.into_next();
    }
    Ok((origin, current))
}

fn rejects_unchanged(destination: &Batch, data: &ChannelPoolSnapshotData) -> Result {
    let before = destination.snapshot_data()?;
    assert!(
        destination.import_snapshot(data).is_err(),
        "Altered coordinated replay enters accepted state"
    );
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

fn change(value: &mut f64) {
    *value = f64::from_bits(value.to_bits() + 1);
}

/// Restore and import of the three-interval package continue as the uninterrupted run does.
fn replays(case: &SurveyedCase) -> Result {
    let (origin, current) = history(case)?;
    let before = origin.snapshot_data()?;
    let saved = current.snapshot_data()?;
    let checkpoint = current.snapshot();
    assert_eq!(checkpoint.data()?, saved);
    let trusted = origin.restore(&checkpoint)?;
    let cold = origin.import_snapshot(&saved)?;
    for restored in [&trusted, &cold] {
        assert_eq!(restored.snapshot_data()?, saved);
        assert_eq!(restored.accepted_steps(), 3);
        let last = restored
            .last_step()?
            .ok_or("Missing restored common receipt")?;
        let Some(ChannelBoundary::Exterior {
            velocity_m_per_s, ..
        }) = &last.forcing_interval.reaches[0].left
        else {
            return Err("Restored forcing changes policy".into());
        };
        assert_eq!(velocity_m_per_s.to_bits(), (-0.0_f64).to_bits());
    }
    let end = 0.5;
    let interval = held(&current, case, end);
    let expected = current
        .trial_to(end, &interval)?
        .into_next()
        .snapshot_data()?;
    for restarted in [trusted, cold] {
        assert_eq!(
            restarted
                .trial_to(end, &interval)?
                .into_next()
                .snapshot_data()?,
            expected
        );
    }
    assert_eq!(current.snapshot_data()?, saved);
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

/// A forged shared transfer packet rejects and leaves the destination unchanged.
fn forgeries_reject(case: &SurveyedCase) -> Result {
    let (origin, current) = history(case)?;
    let saved = current.snapshot_data()?;
    for alteration in 0..8 {
        let mut altered = saved.clone();
        let retained = &mut altered
            .last_interval
            .as_mut()
            .ok_or("Missing retained shared interval")?
            .interval;
        match alteration {
            0 => change(&mut retained.midpoint_pool_stages_m[0]),
            1 => change(&mut retained.input_packet[0]),
            2 => change(&mut retained.output_packet[0]),
            3 => change(&mut retained.connection_volume_m3[0]),
            4 => change(&mut retained.connection_momentum_m4_per_s[0]),
            5 => change(&mut retained.cut_available_lower_m3[0]),
            6 => change(&mut retained.cut_outgoing_upper_m3[0]),
            _ => {
                retained.output_packet.pop().ok_or("Empty output packet")?;
            }
        }
        rejects_unchanged(&origin, &altered)?;
        assert_eq!(current.snapshot_data()?, saved);
    }
    Ok(())
}

#[test]
fn surveyed_rectangle_trusted_and_cold_replay_continue_bit_for_bit() -> Result {
    replays(&surveyed_case(&CASES[0]))
}

#[test]
fn surveyed_three_bands_trusted_and_cold_replay_continue_bit_for_bit() -> Result {
    replays(&THREE_BANDS_CASES[0])
}

#[test]
fn surveyed_rectangle_forged_shared_transfer_packets_reject() -> Result {
    forgeries_reject(&surveyed_case(&CASES[0]))
}

#[test]
fn surveyed_three_bands_forged_shared_transfer_packets_reject() -> Result {
    forgeries_reject(&THREE_BANDS_CASES[0])
}
