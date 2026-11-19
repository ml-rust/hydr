use super::fixture::{Result, batch, forcing, single};
use super::reference::CASES;
use hydr::channel::{
    ChannelBoundary, ChannelGeometry, ChannelSurveyedConfig, CrossSectionDefinition,
    CrossSectionPoint, SurveyedChannelCell,
};
use hydr::coupling::{ChannelPoolBatch, ChannelPoolEndpoint, ChannelPoolError};

/// A surveyed rectangle of the case width enters the coordinator and advances one interval.
/// The connection volume equals the reach end face volume bit for bit.
#[test]
fn surveyed_reach_attachment_admits_and_pairs_the_end_face_volume() -> Result {
    let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    setup.reaches[0].definition.geometry = ChannelGeometry::PrismaticSurveyed {
        profile: CrossSectionDefinition {
            id: "surveyed-rectangle".into(),
            vertical_datum: setup.vertical_datum.clone(),
            points: [(0.0, 4.0), (0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: 4.0,
            profile_provenance: "Explicit analytical rectangular survey".into(),
        },
        cells: vec![SurveyedChannelCell {
            id: "cell-0".into(),
            length_m: 128.0,
            manning_n: 0.0,
        }],
        numerical: ChannelSurveyedConfig::default(),
    };
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let next = origin
        .trial_to(0.125, &forcing(&setup, 0.0, 0.125))?
        .into_next();
    assert_eq!(next.time().to_bits(), 0.125_f64.to_bits());
    let receipt = next.last_step()?.ok_or("Missing finite-pool receipt")?;
    let child = next.reach("reach").ok_or("Missing reach")?;
    let child_receipt = child.last_step()?.ok_or("Missing channel receipt")?;
    assert_eq!(
        receipt.transfers[0].signed_volume_m3.to_bits(),
        child_receipt.face_volume_m3[1].to_bits()
    );
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

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
