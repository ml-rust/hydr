//! Frozen channel/pool references on surveyed reaches.
//!
//! - References: all 18 frozen `CASES` on the surveyed rectangle (`b = coefficient`) and
//!   triangle (`s = coefficient`) profiles, both ends. Six three-band cases follow.
//! - Tolerance: trace, transfer, head, pool volume, reach volume, and momentum within
//!   `1e-10` absolute plus `1e-10` relative. Pool-end pairing is exact every interval.

use super::fixture::{Result, batch, check_water, near, side_index};
use super::surveyed_fixture::{held, rectangle_and_triangle_cases, rest_depth, single};
use super::surveyed_profiles::TRIANGLE_1;
use super::surveyed_reference::{SurveyedCase, THREE_BANDS_CASES};
use hydr::coupling::ChannelPoolEndpoint;

/// Runs one interval of a case at one end and compares every frozen quantity.
pub(super) fn check_case(case: &SurveyedCase, endpoint: ChannelPoolEndpoint) -> Result {
    let label = format!(
        "{} {} area{} {endpoint:?}",
        case.name, case.profile.name, case.pool_area
    );
    let (setup, initial) = single(case, endpoint)?;
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let interval = held(&setup, rest_depth(case), 0.0, case.dt);
    let next = origin.trial_to(case.dt, &interval)?.into_next();
    assert_eq!(next.time().to_bits(), case.dt.to_bits(), "{label}");
    let receipt = next.last_step()?.ok_or("Missing finite-pool receipt")?;
    assert_eq!(receipt.retries, 0, "{label}");
    assert!(receipt.iterations <= 64, "{label}");
    let child = next.reach("reach").ok_or("Missing reach")?;
    let child_receipt = child.last_step()?.ok_or("Missing actual channel receipt")?;
    let trace = child_receipt.boundary_traces[side_index(endpoint)]
        .as_ref()
        .ok_or("Missing actual reservoir trace")?;
    assert!(trace.reservoir.is_some(), "{label}");
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
        -normal * case.dt * reference.momentum_flux,
    );
    assert_eq!(
        transfer.signed_volume_m3.to_bits(),
        (normal * child_receipt.face_volume_m3[side_index(endpoint)]).to_bits(),
        "{label}"
    );
    let pool_receipt = next
        .pools()
        .last_step()?
        .ok_or("Missing actual pool receipt")?;
    let paired = &pool_receipt.receipts[0];
    assert_eq!(
        paired.incoming_volume_m3.to_bits(),
        transfer.signed_volume_m3.max(0.0).to_bits(),
        "{label}"
    );
    assert_eq!(
        paired.outgoing_volume_m3.to_bits(),
        (-transfer.signed_volume_m3).max(0.0).to_bits(),
        "{label}"
    );
    check_water(&next)?;
    assert_eq!(origin.snapshot_data()?, before, "{label}");
    Ok(())
}

/// Runs the rectangular cases or the triangular cases at one end.
fn check_profile(triangle: bool, endpoint: ChannelPoolEndpoint) -> Result {
    for case in rectangle_and_triangle_cases() {
        if (case.profile.name == TRIANGLE_1.name) == triangle {
            check_case(&case, endpoint)?;
        }
    }
    Ok(())
}

/// Runs the six three-band cases at one end.
fn check_bands(endpoint: ChannelPoolEndpoint) -> Result {
    for case in THREE_BANDS_CASES {
        check_case(&case, endpoint)?;
    }
    Ok(())
}

#[test]
fn surveyed_channel_pool_rectangle_left_matches_frozen_references() -> Result {
    check_profile(false, ChannelPoolEndpoint::Left)
}

#[test]
fn surveyed_channel_pool_rectangle_right_matches_frozen_references() -> Result {
    check_profile(false, ChannelPoolEndpoint::Right)
}

#[test]
fn surveyed_channel_pool_triangle_left_matches_frozen_references() -> Result {
    check_profile(true, ChannelPoolEndpoint::Left)
}

#[test]
fn surveyed_channel_pool_triangle_right_matches_frozen_references() -> Result {
    check_profile(true, ChannelPoolEndpoint::Right)
}

#[test]
fn surveyed_channel_pool_three_bands_left_match_frozen_references() -> Result {
    check_bands(ChannelPoolEndpoint::Left)
}

#[test]
fn surveyed_channel_pool_three_bands_right_match_frozen_references() -> Result {
    check_bands(ChannelPoolEndpoint::Right)
}
