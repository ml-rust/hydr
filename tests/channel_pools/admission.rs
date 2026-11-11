use super::fixture::{Result, batch, forcing, single};
use super::reference::CASES;
use hydr::channel::ChannelBoundary;
use hydr::coupling::{ChannelPoolBatch, ChannelPoolEndpoint, ChannelPoolError};

#[test]
fn topology_time_gravity_datum_and_endpoint_ownership_require_consistent_declarations() -> Result {
    for mode in 0..17 {
        let (mut setup, mut initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
        match mode {
            0 => setup.reaches.clear(),
            1 => setup.pools.clear(),
            2 => setup.connections.clear(),
            3 => setup.connections.push(setup.connections[0].clone()),
            4 => {
                let mut other = setup.connections[0].clone();
                other.id = "duplicate-owner".into();
                setup.connections.push(other);
            }
            5 => setup.connections[0].reach_id = "foreign".into(),
            6 => setup.connections[0].pool_id = "foreign".into(),
            7 => setup.reaches[0].definition.vertical_datum = "foreign".into(),
            8 => setup.vertical_datum = "foreign".into(),
            9 => initial.reaches[0].time_s = 0.125,
            10 => initial.time_s = -0.0,
            11 => setup.connections[0].entrance_loss_coefficient = -1.0,
            12 => setup.connections[0].entrance_loss_coefficient = f64::NAN,
            13 => setup.connections[0].physical_connection_assertion.clear(),
            14 => setup.connections[0].quiescent_reservoir_assertion.clear(),
            15 => setup.nonoverlap_assertion.clear(),
            _ => {
                let mut second = setup.reaches[0].clone();
                second.id = "second".into();
                second.definition.config.gravity = 2.0;
                setup.reaches.push(second);
                initial.reaches.push(initial.reaches[0].clone());
            }
        }
        assert!(
            ChannelPoolBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                setup,
                initial
            )
            .is_err(),
            "Accepted incompatible topology {mode}"
        );
    }
    Ok(())
}

#[test]
fn foreign_missing_duplicate_and_owned_forcing_records_reject_atomically() -> Result {
    let (setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    for mode in 0..8 {
        let mut held = forcing(&setup, 0.0, 0.125);
        match mode {
            0 => held.reaches.clear(),
            1 => held.reaches[0].reach_id = "foreign".into(),
            2 => held.reaches.push(held.reaches[0].clone()),
            3 => held.reaches[0].right = Some(ChannelBoundary::Wall),
            4 => held.reaches[0].left = None,
            5 => held.start_s = -0.0,
            6 => held.end_s = f64::INFINITY,
            _ => {
                held.reaches[0].right = Some(ChannelBoundary::QuiescentReservoir {
                    elevation_m: 1.0,
                    entrance_loss_coefficient: 0.0,
                })
            }
        }
        assert!(
            matches!(
                origin.trial_to(0.125, &held),
                Err(ChannelPoolError::Input(_))
            ),
            "Accepted malformed forcing {mode}"
        );
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn bounded_iteration_and_invalid_requests_preserve_every_child_and_counter() -> Result {
    let (mut setup, initial) = single(CASES[12], ChannelPoolEndpoint::Right)?;
    setup.config.iteration_limit = 1;
    setup.config.retry_limit = 0;
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let error = origin
        .trial_to(0.125, &forcing(&setup, 0.0, 0.125))
        .err()
        .ok_or("Unconverged finite-pool interval enters accepted state")?;
    assert!(matches!(
        error,
        ChannelPoolError::Unresolved { retries: 0, .. }
    ));
    assert_eq!(origin.snapshot_data()?, before);
    for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(origin.trial_to(end, &forcing(&setup, 0.0, 0.125)).is_err());
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}
