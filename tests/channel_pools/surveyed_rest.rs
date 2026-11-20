//! Surveyed channel/pool lake at rest and flow reversal.
//!
//! - Lake: depth `1 m` within `1e-12`, discharge and momentum `<= 1e-11`, storage change
//!   `<= 1e-11`, 16 intervals, both ends, `k = 0` and `1`.
//! - Reversal: both transfer directions beyond `1e-6 m³`.

use super::fixture::{Result, batch, check_water, forcing};
use super::reference::CASES;
use super::surveyed_fixture::{definition, held, single, surveyed_case};
use super::surveyed_profiles::RECTANGLE_1;
use super::surveyed_reference::{SurveyedCase, THREE_BANDS_CASES};
use hydr::channel::{ChannelBoundary, ChannelInitialState};
use hydr::coupling::ChannelPoolEndpoint;

/// The surveyed rectangle, the surveyed triangle, and three bands, each at depth 1 m.
fn lake_cases() -> [SurveyedCase; 3] {
    let mut bands = THREE_BANDS_CASES[0];
    // The three-band section holds 3 m² at depth 1 m, a knot depth.
    bands.interior_area_m2 = 3.0;
    [surveyed_case(&CASES[0]), surveyed_case(&CASES[3]), bands]
}

#[test]
fn surveyed_finite_pools_preserve_bounded_lake_rest_at_both_endpoint_orientations() -> Result {
    for mut case in lake_cases() {
        case.elevation = 1.0;
        for endpoint in [ChannelPoolEndpoint::Left, ChannelPoolEndpoint::Right] {
            for loss in [0.0, 1.0] {
                case.loss = loss;
                let (setup, initial) = single(&case, endpoint)?;
                let origin = batch(setup.clone(), initial)?;
                let before = origin.snapshot_data()?;
                let storage = origin.ledger()?.storage_m3;
                let mut current = origin.clone();
                for index in 0..16 {
                    let end = (index + 1) as f64 / 8.0;
                    let interval = held(&setup, 1.0, current.time(), end);
                    current = current.trial_to(end, &interval)?.into_next();
                    assert_eq!(current.time(), end);
                    let state = current.reach("reach").ok_or("Missing reach")?.state()?;
                    let label = format!(
                        "{} {endpoint:?} k={loss} interval {index}",
                        case.profile.name
                    );
                    assert!((state[0].depth_m - 1.0).abs() <= 1e-12, "{label}");
                    assert!(state[0].discharge_m3_per_s.abs() <= 1e-11, "{label}");
                    assert!(
                        state[0].longitudinal_momentum_m4_per_s.abs() <= 1e-11,
                        "{label}"
                    );
                    assert!(
                        (current.ledger()?.storage_m3 - storage).abs() <= 1e-11,
                        "{label}"
                    );
                    let step = current.last_step()?.ok_or("Missing finite rest receipt")?;
                    assert!(step.transfers[0].signed_volume_m3.abs() <= 1e-11, "{label}");
                    check_water(&current)?;
                }
                assert_eq!(origin.snapshot_data()?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn surveyed_finite_storage_exchange_reverses_without_a_policy_switch() -> Result {
    let mut case = surveyed_case(&CASES[0]);
    case.profile = RECTANGLE_1;
    case.gravity = 1.0;
    case.interior_area_m2 = 1.0;
    case.elevation = 0.95;
    case.pool_area = 1.0;
    case.loss = 0.0;
    let (mut setup, mut initial) = single(&case, ChannelPoolEndpoint::Right)?;
    setup.reaches[0].definition = definition(RECTANGLE_1, 1.0, 10, 0.1);
    initial.reaches[0] = ChannelInitialState {
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
        let mut interval = forcing(&setup, current.time(), 4.0);
        interval.reaches[0].left = Some(ChannelBoundary::Wall);
        let start = current.time();
        current = current.trial_to(4.0, &interval)?.into_next();
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
        assert!(steps < 100_000, "Surveyed storage reversal stalls");
    }
    assert!(
        positive && negative,
        "Surveyed finite-pool flow never reverses"
    );
    let counters = current.connection_ledgers()?;
    assert!(counters[0].channel_to_pool_volume_m3 > 0.0);
    assert!(counters[0].pool_to_channel_volume_m3 > 0.0);
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}
