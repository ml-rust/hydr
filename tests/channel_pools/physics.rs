use super::fixture::{
    Result, batch, check_water, connection, forcing, near, pool, side_index, single,
};
use super::reference::CASES;
use hydr::channel::{ChannelBoundary, ChannelReach};
use hydr::coupling::{ChannelPoolEndpoint, ChannelPoolReach};
use hydr::storage::LevelPoolState;

#[test]
fn nonlinear_finite_pool_feedback_matches_all_frozen_full_span_references() -> Result {
    for case in CASES {
        for endpoint in [ChannelPoolEndpoint::Left, ChannelPoolEndpoint::Right] {
            let (setup, initial) = single(case, endpoint)?;
            let origin = batch(setup.clone(), initial.clone())?;
            let before = origin.snapshot_data()?;
            let held = forcing(&setup, 0.0, 0.125);
            let next = origin.trial_to(0.125, &held)?.into_next();
            assert_eq!(next.time().to_bits(), 0.125_f64.to_bits());
            let receipt = next.last_step()?.ok_or("Missing finite-pool receipt")?;
            assert_eq!(receipt.retries, 0, "{} area{}", case.name, case.pool_area);
            assert!(receipt.iterations <= 64);
            let child = next.reach("reach").ok_or("Missing reach")?;
            let child_receipt = child.last_step()?.ok_or("Missing actual channel receipt")?;
            let trace = child_receipt.boundary_traces[side_index(endpoint)]
                .as_ref()
                .ok_or("Missing actual reservoir trace")?;
            assert!(trace.reservoir.is_some());
            let transfer = &receipt.transfers[0];
            let reference = case.expected;
            near(trace.depth_m, reference.depth);
            near(trace.outward_velocity_m_per_s, reference.velocity);
            near(trace.outward_discharge_m3_per_s, reference.discharge);
            near(trace.momentum_flux_m4_per_s2, reference.momentum_flux);
            near(transfer.signed_volume_m3, reference.transfer);
            near(transfer.midpoint_pool_elevation_m, reference.head);
            near(next.pools().states()?[0].volume_m3, reference.pool_volume);
            let state = child.state()?;
            near(state[0].volume_m3, reference.channel_volume);
            let normal = if endpoint == ChannelPoolEndpoint::Left {
                -1.0
            } else {
                1.0
            };
            near(
                state[0].longitudinal_momentum_m4_per_s,
                normal * reference.right_momentum,
            );
            near(
                transfer.longitudinal_momentum_impulse_m4_per_s,
                -normal * 0.125 * reference.momentum_flux,
            );
            assert_eq!(
                transfer.signed_volume_m3.to_bits(),
                (normal * child_receipt.face_volume_m3[side_index(endpoint)]).to_bits()
            );
            let pool_receipt = next
                .pools()
                .last_step()?
                .ok_or("Missing actual pool receipt")?;
            let paired = &pool_receipt.receipts[0];
            assert_eq!(
                paired.incoming_volume_m3.to_bits(),
                transfer.signed_volume_m3.max(0.0).to_bits()
            );
            assert_eq!(
                paired.outgoing_volume_m3.to_bits(),
                (-transfer.signed_volume_m3).max(0.0).to_bits()
            );
            let held_reach = ChannelReach::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                setup.reaches[0].definition.clone(),
                initial.reaches[0].clone(),
            )?;
            assert_ne!(
                transfer.midpoint_pool_elevation_m.to_bits(),
                case.elevation.to_bits()
            );
            let fixed_policy = ChannelBoundary::QuiescentReservoir {
                elevation_m: case.elevation,
                entrance_loss_coefficient: case.loss,
            };
            let fixed_forcing = hydr::channel::ChannelForcingInterval {
                start_s: 0.0,
                end_s: 0.125,
                left: if endpoint == ChannelPoolEndpoint::Left {
                    fixed_policy.clone()
                } else {
                    held.reaches[0]
                        .left
                        .clone()
                        .ok_or("Missing external left")?
                },
                right: if endpoint == ChannelPoolEndpoint::Right {
                    fixed_policy
                } else {
                    held.reaches[0]
                        .right
                        .clone()
                        .ok_or("Missing external right")?
                },
                lateral_inputs: Vec::new(),
            };
            let fixed_next = held_reach.step(0.125, &fixed_forcing)?.next;
            let fixed_receipt = fixed_next.last_step()?.ok_or("Missing held-head receipt")?;
            let fixed_trace = fixed_receipt.boundary_traces[side_index(endpoint)]
                .as_ref()
                .ok_or("Missing held-head trace")?;
            assert!(
                (fixed_trace.outward_discharge_m3_per_s - trace.outward_discharge_m3_per_s).abs()
                    > 1e-7
            );
            check_water(&next)?;
            assert_eq!(origin.snapshot_data()?, before);
        }
    }
    Ok(())
}

#[test]
fn finite_pools_preserve_bounded_lake_rest_at_both_endpoint_orientations() -> Result {
    for mut case in [CASES[0], CASES[3]] {
        case.elevation = 1.0;
        for endpoint in [ChannelPoolEndpoint::Left, ChannelPoolEndpoint::Right] {
            for loss in [0.0, 1.0] {
                case.loss = loss;
                let (setup, initial) = single(case, endpoint)?;
                let origin = batch(setup.clone(), initial)?;
                let before = origin.snapshot_data()?;
                let storage = origin.ledger()?.storage_m3;
                let mut current = origin.clone();
                for index in 0..16 {
                    let end = (index + 1) as f64 / 8.0;
                    current = current
                        .trial_to(end, &forcing(&setup, current.time(), end))?
                        .into_next();
                    assert_eq!(current.time(), end);
                    let state = current.reach("reach").ok_or("Missing reach")?.state()?;
                    assert!((state[0].depth_m - 1.0).abs() <= 1e-12);
                    assert!(state[0].discharge_m3_per_s.abs() <= 1e-11);
                    assert!(state[0].longitudinal_momentum_m4_per_s.abs() <= 1e-11);
                    assert!((current.ledger()?.storage_m3 - storage).abs() <= 1e-11);
                    let step = current.last_step()?.ok_or("Missing finite rest receipt")?;
                    assert!(step.transfers[0].signed_volume_m3.abs() <= 1e-11);
                    check_water(&current)?;
                }
                assert_eq!(origin.snapshot_data()?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn shared_pool_both_endpoints_and_multiple_reaches_preserve_declaration_order_independence()
-> Result {
    let (mut setup, mut initial) = single(CASES[0], ChannelPoolEndpoint::Left)?;
    setup.connections.push(connection(
        "second-end",
        "reach",
        ChannelPoolEndpoint::Right,
        "pool",
        1.0,
    ));
    setup.reaches.push(ChannelPoolReach {
        id: "second".into(),
        definition: setup.reaches[0].definition.clone(),
    });
    initial.reaches.push(initial.reaches[0].clone());
    setup.pools.push(pool("second-pool", 4.0, 0.0)?);
    initial.pools.push(LevelPoolState { volume_m3: 4.4 });
    setup.connections.push(connection(
        "second-reach",
        "second",
        ChannelPoolEndpoint::Right,
        "second-pool",
        1.0,
    ));
    let mut records = Vec::new();
    for reverse in [false, true] {
        let mut layout = setup.clone();
        let mut start = initial.clone();
        if reverse {
            layout.reaches.reverse();
            start.reaches.reverse();
            layout.pools.reverse();
            start.pools.reverse();
            layout.connections.reverse();
        }
        let origin = batch(layout.clone(), start)?;
        let mut held = forcing(&layout, 0.0, 0.125);
        held.reaches.reverse();
        let next = origin.trial_to(0.125, &held)?.into_next();
        let step = next
            .last_step()?
            .ok_or("Missing multiple-component receipt")?;
        let mut transfers = step.transfers;
        transfers.sort_by(|a, b| a.connection_id.cmp(&b.connection_id));
        let pool_states = next.pools().states()?;
        let mut pools: Vec<_> = layout
            .pools
            .iter()
            .zip(pool_states)
            .map(|(p, s)| (p.id.clone(), s.volume_m3))
            .collect();
        pools.sort_by(|a, b| a.0.cmp(&b.0));
        records.push((
            next.reach("reach").ok_or("Missing reach")?.state()?,
            next.reach("second")
                .ok_or("Missing second reach")?
                .state()?,
            transfers,
            pools,
        ));
        check_water(&next)?;
    }
    for (a, b) in records[0].0.iter().zip(&records[1].0) {
        near(a.volume_m3, b.volume_m3);
        near(
            a.longitudinal_momentum_m4_per_s,
            b.longitudinal_momentum_m4_per_s,
        );
    }
    for (a, b) in records[0].1.iter().zip(&records[1].1) {
        near(a.volume_m3, b.volume_m3);
        near(
            a.longitudinal_momentum_m4_per_s,
            b.longitudinal_momentum_m4_per_s,
        );
    }
    for (a, b) in records[0].2.iter().zip(&records[1].2) {
        assert_eq!(a.connection_id, b.connection_id);
        near(a.signed_volume_m3, b.signed_volume_m3);
        near(a.midpoint_pool_elevation_m, b.midpoint_pool_elevation_m);
    }
    for (a, b) in records[0].3.iter().zip(&records[1].3) {
        assert_eq!(a.0, b.0);
        near(a.1, b.1);
    }
    Ok(())
}

#[test]
fn finite_storage_exchange_reverses_without_a_policy_switch_or_transfer_scaling() -> Result {
    let mut case = CASES[0];
    case.coefficient = 1.0;
    case.elevation = 0.95;
    case.pool_area = 1.0;
    case.loss = 0.0;
    let (mut setup, mut initial) = single(case, ChannelPoolEndpoint::Right)?;
    setup.reaches[0].definition = super::fixture::definition(false, 1.0, 1.0, 10, 0.1, 0.0);
    initial.reaches[0] = hydr::channel::ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![0.1; 10],
        longitudinal_momentum_m4_per_s: vec![0.0; 10],
    };
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let mut current = origin.clone();
    let mut positive = false;
    let mut negative = false;
    let mut steps = 0;
    // Stop at the first reversal. Later steps repeat the same exchange.
    while current.time() < 4.0 && !(positive && negative) {
        let mut held = forcing(&setup, current.time(), 4.0);
        held.reaches[0].left = Some(ChannelBoundary::Wall);
        let start = current.time();
        current = current.trial_to(4.0, &held)?.into_next();
        assert!(current.time() > start && current.time() <= 4.0);
        let step = current
            .last_step()?
            .ok_or("Missing reversal transfer receipt")?;
        let transfer = step.transfers[0].signed_volume_m3;
        positive |= transfer > 1e-6;
        negative |= transfer < -1e-6;
        let child = current
            .reach("reach")
            .ok_or("Missing reversal reach")?
            .last_step()?
            .ok_or("Missing reversal child receipt")?;
        assert_eq!(transfer.to_bits(), child.face_volume_m3[10].to_bits());
        check_water(&current)?;
        steps += 1;
        assert!(steps < 100_000, "Finite storage reversal stalls");
    }
    assert!(positive && negative, "Finite-pool flow never reverses");
    let counters = current.connection_ledgers()?;
    assert!(counters[0].channel_to_pool_volume_m3 > 0.0);
    assert!(counters[0].pool_to_channel_volume_m3 > 0.0);
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn competing_pool_exports_reserve_original_water_without_incoming_credit() -> Result {
    let mut case = CASES[0];
    case.coefficient = 1.0;
    let (mut setup, mut initial) = single(case, ChannelPoolEndpoint::Right)?;
    setup.pools[0].curve = hydr::storage::LevelPoolCurve::new(
        vec![
            hydr::storage::LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: 0.0,
            },
            hydr::storage::LevelPoolKnot {
                volume_m3: 0.01,
                stage_m: 2.0,
            },
            hydr::storage::LevelPoolKnot {
                volume_m3: 32.0,
                stage_m: 4.0,
            },
        ],
        super::fixture::DATUM.into(),
        "Small original donor with broad upper storage".into(),
    )?;
    initial.pools[0].volume_m3 = 0.01;
    setup.reaches.push(ChannelPoolReach {
        id: "second-export".into(),
        definition: setup.reaches[0].definition.clone(),
    });
    initial.reaches.push(initial.reaches[0].clone());
    setup.connections.push(connection(
        "second-export",
        "second-export",
        ChannelPoolEndpoint::Right,
        "pool",
        0.0,
    ));
    setup.reaches.push(ChannelPoolReach {
        id: "incoming".into(),
        definition: super::fixture::definition(false, 1.0, 4.0, 1, 128.0, 0.0),
    });
    initial.reaches.push(hydr::channel::ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![1024.0],
        longitudinal_momentum_m4_per_s: vec![512.0],
    });
    setup.connections.push(connection(
        "incoming",
        "incoming",
        ChannelPoolEndpoint::Right,
        "pool",
        0.0,
    ));
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let mut held = forcing(&setup, 0.0, 0.125);
    held.reaches[2].left = Some(ChannelBoundary::Exterior {
        depth_m: 2.0,
        velocity_m_per_s: 0.5,
    });
    let next = origin.trial_to(0.125, &held)?.into_next();
    assert!(next.time() > 0.0 && next.time() < 0.125);
    let step = next.last_step()?.ok_or("Missing aggregate donor receipt")?;
    let outgoing: f64 = step
        .transfers
        .iter()
        .map(|t| (-t.signed_volume_m3).max(0.0))
        .sum();
    let incoming: f64 = step
        .transfers
        .iter()
        .map(|t| t.signed_volume_m3.max(0.0))
        .sum();
    assert!(outgoing > 0.0 && incoming > outgoing);
    assert!(outgoing <= 0.01);
    assert!(next.pools().states()?[0].volume_m3 > 0.01);
    let cut = step
        .donor_cuts
        .iter()
        .find(|c| c.bed_elevation_m == 0.0)
        .ok_or("Missing original donor cut")?;
    assert!(cut.available_lower_m3 <= 0.01);
    assert!(cut.outgoing_upper_m3 >= outgoing);
    assert!(cut.outgoing_upper_m3 <= cut.available_lower_m3);
    check_water(&next)?;
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}
