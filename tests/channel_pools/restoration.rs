use super::fixture::{Batch, Result, batch, check_water, connection, forcing, single};
use super::reference::CASES;
use hydr::channel::ChannelBoundary;
use hydr::coupling::{ChannelPoolEndpoint, ChannelPoolSnapshotData};

pub(super) fn history() -> Result<(Batch, Batch)> {
    let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    setup.connections[0].entrance_loss_coefficient = -0.0;
    let origin = batch(setup.clone(), initial)?;
    let mut current = origin.clone();
    for index in 0..3 {
        let end = (index + 1) as f64 / 8.0;
        let mut held = forcing(&setup, current.time(), end);
        held.reaches[0].left = Some(ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: -0.0,
        });
        current = current.trial_to(end, &held)?.into_next();
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

#[test]
fn complete_trusted_and_cold_replay_preserve_original_forcing_and_future_children() -> Result {
    let (origin, current) = history()?;
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
    let mut held = forcing(current.setup(), current.time(), end);
    held.reaches[0].left = Some(ChannelBoundary::Exterior {
        depth_m: 1.0,
        velocity_m_per_s: -0.0,
    });
    let expected = current.trial_to(end, &held)?.into_next().snapshot_data()?;
    for restarted in [trusted, cold] {
        assert_eq!(
            restarted
                .trial_to(end, &held)?
                .into_next()
                .snapshot_data()?,
            expected
        );
    }
    assert_eq!(current.snapshot_data()?, saved);
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn replay_requires_the_actual_used_head_and_conservative_shared_transfer_packet() -> Result {
    let (origin, current) = history()?;
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
fn replay_requires_original_interval_support_retry_history_and_external_operand_bits() -> Result {
    let (origin, current) = history()?;
    let saved = current.snapshot_data()?;
    for alteration in 0..7 {
        let mut altered = saved.clone();
        if alteration == 0 {
            altered.last_interval = None;
        } else {
            let last = altered
                .last_interval
                .as_mut()
                .ok_or("Missing retained requested interval")?;
            match alteration {
                1 => last.requested_end_s += 0.125,
                2 => last.retries += 1,
                3 => last.previous.pools.time += 0.01,
                4 => change(&mut last.previous.counters.connection_in_m3[0]),
                5 => last.forcing.start_s += 0.01,
                _ => {
                    let Some(ChannelBoundary::Exterior {
                        velocity_m_per_s, ..
                    }) = &mut last.forcing.reaches[0].left
                    else {
                        return Err("Retained external forcing changes policy".into());
                    };
                    *velocity_m_per_s = 0.0;
                }
            }
        }
        rejects_unchanged(&origin, &altered)?;
        assert_eq!(current.snapshot_data()?, saved);
    }
    Ok(())
}

#[test]
fn original_initial_and_setup_identity_preserve_signed_zero_and_control_bits() -> Result {
    let (origin, current) = history()?;
    let saved = current.snapshot_data()?;
    for alteration in 0..5 {
        let mut altered = saved.clone();
        match alteration {
            0 => altered.initial.time_s = -0.0,
            1 => altered.initial.reaches[0].longitudinal_momentum_m4_per_s[0] = -0.0,
            2 => altered.setup.connections[0].entrance_loss_coefficient = 0.0,
            3 => change(&mut altered.setup.config.head_absolute_m),
            _ => change(&mut altered.initial.pools[0].volume_m3),
        }
        rejects_unchanged(&origin, &altered)?;
    }
    let mut incompatible = saved.setup.clone();
    incompatible.connections[0].entrance_loss_coefficient = 0.0;
    let destination = batch(incompatible, saved.initial.clone())?;
    let destination_before = destination.snapshot_data()?;
    assert!(destination.restore(&current.snapshot()).is_err());
    assert_eq!(destination.snapshot_data()?, destination_before);
    assert_eq!(current.snapshot_data()?, saved);
    Ok(())
}

#[test]
fn zero_history_and_previous_frame_counters_require_original_positive_zero_bits() -> Result {
    let (setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    let origin = batch(setup.clone(), initial)?;
    let original = origin.snapshot_data()?;
    for charge in [false, true] {
        let mut altered = original.clone();
        let values = if charge {
            &mut altered.counters.historical_physical_charge_m3
        } else {
            &mut altered.counters.connection_in_m3
        };
        assert_eq!(values[0].to_bits(), 0.0_f64.to_bits());
        values[0] = -0.0;
        rejects_unchanged(&origin, &altered)?;
    }
    let first = origin
        .trial_to(0.125, &forcing(&setup, 0.0, 0.125))?
        .into_next();
    let saved = first.snapshot_data()?;
    for charge in [false, true] {
        let mut altered = saved.clone();
        let counters = &mut altered
            .last_interval
            .as_mut()
            .ok_or("Missing first-interval frame")?
            .previous
            .counters;
        let values = if charge {
            &mut counters.historical_physical_charge_m3
        } else {
            &mut counters.connection_in_m3
        };
        assert_eq!(values[0].to_bits(), 0.0_f64.to_bits());
        values[0] = -0.0;
        rejects_unchanged(&origin, &altered)?;
        assert_eq!(first.snapshot_data()?, saved);
    }
    Ok(())
}

#[test]
fn grouped_connection_and_pool_histories_remain_paired_across_shared_endpoint_steps() -> Result {
    let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    setup.connections.push(connection(
        "left",
        "reach",
        ChannelPoolEndpoint::Left,
        "pool",
        1.0,
    ));
    let origin = batch(setup.clone(), initial)?;
    let mut current = origin.clone();
    for index in 0..4 {
        let end = (index + 1) as f64 / 16.0;
        current = current
            .trial_to(end, &forcing(&setup, current.time(), end))?
            .into_next();
        let saved = current.snapshot_data()?;
        let pool = &saved.pools.element_ledgers[0];
        for (connections, bounds, actual, actual_bound, assembly) in [
            (
                &saved.counters.connection_in_m3,
                &saved.counters.connection_in_bound_m3,
                pool.cumulative_incoming_volume_m3,
                pool.incoming_roundoff_bound_m3,
                saved.counters.pool_incoming_assembly_bound_m3[0],
            ),
            (
                &saved.counters.connection_out_m3,
                &saved.counters.connection_out_bound_m3,
                pool.cumulative_outgoing_volume_m3,
                pool.outgoing_roundoff_bound_m3,
                saved.counters.pool_outgoing_assembly_bound_m3[0],
            ),
        ] {
            let total: f64 = connections.iter().sum();
            let allowance = bounds.iter().sum::<f64>()
                + actual_bound
                + assembly
                + 8.0 * f64::EPSILON * total.abs();
            assert!((total - actual).abs() <= allowance);
        }
        check_water(&current)?;
    }
    let saved = current.snapshot_data()?;
    let cold = origin.import_snapshot(&saved)?;
    assert_eq!(cold.snapshot_data()?, saved);
    let mut altered = saved.clone();
    altered.counters.connection_in_m3[0] += 0.01;
    rejects_unchanged(&origin, &altered)?;
    let mut altered = saved.clone();
    altered.pools.element_ledgers[0].cumulative_incoming_volume_m3 += 0.01;
    rejects_unchanged(&origin, &altered)?;
    assert_eq!(current.snapshot_data()?, saved);
    Ok(())
}

#[test]
fn parent_external_histories_require_each_reachs_actual_boundary_throughput() -> Result {
    let (setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    let origin = batch(setup.clone(), initial)?;
    let mut current = origin.clone();
    for index in 0..2 {
        let end = (index + 1) as f64 / 8.0;
        let mut held = forcing(&setup, current.time(), end);
        held.reaches[0].left = Some(ChannelBoundary::Wall);
        current = current.trial_to(end, &held)?.into_next();
    }
    let saved = current.snapshot_data()?;
    assert_eq!(saved.counters.external_in_m3[0], 0.0);
    assert_eq!(saved.counters.external_out_m3[0], 0.0);
    assert_eq!(saved.counters.external_in_bound_m3[0], 0.0);
    assert_eq!(saved.counters.external_out_bound_m3[0], 0.0);
    let mut altered = saved.clone();
    altered.counters.external_in_m3[0] = 1.0;
    altered.counters.external_out_m3[0] = 1.0;
    let previous = &mut altered
        .last_interval
        .as_mut()
        .ok_or("Missing external-history replay frame")?
        .previous;
    assert_eq!(previous.counters.external_in_m3[0], 0.0);
    assert_eq!(previous.counters.external_out_m3[0], 0.0);
    assert_eq!(previous.counters.external_in_bound_m3[0], 0.0);
    assert_eq!(previous.counters.external_out_bound_m3[0], 0.0);
    previous.counters.external_in_m3[0] = 1.0;
    previous.counters.external_out_m3[0] = 1.0;
    assert_eq!(altered.reaches, saved.reaches);
    assert_eq!(
        altered
            .last_interval
            .as_ref()
            .ok_or("Missing altered replay frame")?
            .previous
            .reaches,
        saved
            .last_interval
            .as_ref()
            .ok_or("Missing original replay frame")?
            .previous
            .reaches
    );
    rejects_unchanged(&origin, &altered)?;
    assert_eq!(current.snapshot_data()?, saved);
    Ok(())
}

#[test]
fn claimed_parent_counter_bounds_cannot_mask_unrecorded_external_throughput() -> Result {
    let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    setup.config.mass_absolute_m3 = 1.0;
    let origin = batch(setup.clone(), initial)?;
    let mut current = origin.clone();
    for index in 0..3 {
        let end = (index + 1) as f64 / 8.0;
        current = current
            .trial_to(end, &forcing(&setup, current.time(), end))?
            .into_next();
    }
    let saved = current.snapshot_data()?;
    let original_previous = &saved
        .last_interval
        .as_ref()
        .ok_or("Missing claimed-bound replay frame")?
        .previous;
    for reach in [&saved.reaches[0], &original_previous.reaches[0]] {
        assert!(reach.ledger.boundary_in_m3 > 0.0);
        assert!(reach.ledger.boundary_out_m3 > 0.0);
    }
    let mut altered = saved.clone();
    let previous = &mut altered
        .last_interval
        .as_mut()
        .ok_or("Missing altered claimed-bound frame")?
        .previous;
    for counters in [&mut altered.counters, &mut previous.counters] {
        counters.external_in_m3[0] += 0.01;
        counters.external_out_m3[0] += 0.01;
        counters.external_in_bound_m3[0] += 0.01;
        counters.external_out_bound_m3[0] += 0.01;
        assert!(
            counters.external_in_bound_m3[0] + counters.external_out_bound_m3[0]
                < setup.config.mass_absolute_m3
        );
    }
    assert_eq!(altered.reaches, saved.reaches);
    assert_eq!(altered.pools, saved.pools);
    let altered_previous = &altered
        .last_interval
        .as_ref()
        .ok_or("Missing retained claimed-bound frame")?
        .previous;
    assert_eq!(altered_previous.reaches, original_previous.reaches);
    assert_eq!(altered_previous.pools, original_previous.pools);
    rejects_unchanged(&origin, &altered)?;
    assert_eq!(current.snapshot_data()?, saved);
    Ok(())
}
