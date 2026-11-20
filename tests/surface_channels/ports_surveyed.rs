//! Reach-end ports on a surveyed rectangle of the face width.
//!
//! Frozen criterion (plan section 6): each rectangular port test holds with `alpha` declared
//! as one surveyed rectangle of the same width, bed, and start state. Every test here runs the
//! rectangular test body with the surveyed reach, so its criteria are those of the original.
//!
//! - Rest: `ports_rest.rs`.
//! - Uniform flow: `ports_uniform.rs`.
//! - Limits: `ports_limits.rs`.
//! - Donors: `ports_donors.rs`.
//! - Step: `ports_step.rs`.
//! - Order: `ports_order.rs`.
//! - Receipt identity: the rectangular and the surveyed-rectangle reach with identical start
//!   states give bit-identical port receipts for the first attempt.
//!
//! Dam break and seiche cases are in `ports_surveyed_dam_break.rs`.

use crate::fixture::Result;
use crate::ports_donors::donor_factor;
use crate::ports_fixture::{CELLS, Reach, Scenario, still, transfer, west};
use crate::ports_limits::{
    draining_end_cell, east_case, port_speed_binds, speed_covers_every_face_state,
};
use crate::ports_order::declaration_order;
use crate::ports_rest::{four_face_mirror, lake_at_rest, water_below_the_higher_bed};
use crate::ports_step::{steady_down, steady_up};
use crate::ports_uniform::{holds, uniform};
use hydr::coupling::SurfaceChannelPortRegime;

#[test]
fn surveyed_lake_at_rest_stays_at_rest_on_flat_and_stepped_beds() -> Result {
    lake_at_rest(true)
}

#[test]
fn surveyed_water_below_the_higher_bed_moves_nothing() -> Result {
    water_below_the_higher_bed(true)
}

#[test]
fn surveyed_four_face_port_keeps_mirror_cells_equal() -> Result {
    four_face_mirror(true)
}

#[test]
fn surveyed_subcritical_uniform_flow_crosses_the_port_unchanged_both_ways() -> Result {
    holds(0.5, true, SurfaceChannelPortRegime::Subcritical, true)?;
    holds(0.5, false, SurfaceChannelPortRegime::Subcritical, true)
}

#[test]
fn surveyed_supercritical_uniform_flow_crosses_the_port_unchanged_both_ways() -> Result {
    holds(
        2.0,
        true,
        SurfaceChannelPortRegime::SupercriticalToSurface,
        true,
    )?;
    holds(
        2.0,
        false,
        SurfaceChannelPortRegime::SupercriticalToReach,
        true,
    )
}

#[test]
fn surveyed_port_speed_covers_every_face_state() -> Result {
    speed_covers_every_face_state(true)
}

#[test]
fn surveyed_port_speed_binds_the_common_interval() -> Result {
    port_speed_binds(true)
}

#[test]
fn surveyed_draining_end_cell_takes_the_hll_flux_of_its_start_states() -> Result {
    draining_end_cell(true)
}

#[test]
fn surveyed_donor_factor_shares_the_end_cell_water_net_of_port_outflow() -> Result {
    donor_factor(true)
}

#[test]
fn surveyed_steady_flow_up_a_bed_step_matches_the_pure_2d_twin() -> Result {
    steady_up(true)
}

#[test]
fn surveyed_steady_flow_down_a_bed_step_matches_the_pure_2d_twin() -> Result {
    steady_down(true)
}

#[test]
fn surveyed_declaration_order_changes_no_accepted_bit() -> Result {
    declaration_order(true)
}

/// Debug text of the port receipts and the retry count of the first interval of `case`.
///
/// Debug text shows every bit of finite values, signed zero included.
fn first_attempt(case: &Scenario, dt: f64) -> Result<String> {
    let batch = case.step(&case.build()?, dt)?;
    let diagnostics = batch.last_step()?.ok_or("an accepted interval")?;
    if diagnostics.retries != 0 {
        return Err(format!(
            "the interval of {dt} s needs {} retries, so it has no first attempt",
            diagnostics.retries
        )
        .into());
    }
    Ok(format!("{:?} {:?}", diagnostics.ports, transfer(&batch)?))
}

/// Fails unless `build(false)` and `build(true)` give identical port receipts.
fn identical(name: &str, dt: f64, build: impl Fn(bool) -> Scenario) -> Result {
    let rectangular =
        first_attempt(&build(false), dt).map_err(|error| format!("{name} rectangular: {error}"))?;
    let surveyed =
        first_attempt(&build(true), dt).map_err(|error| format!("{name} surveyed: {error}"))?;
    assert_eq!(rectangular, surveyed, "{name}");
    Ok(())
}

#[test]
fn surveyed_port_receipts_equal_the_rectangular_receipts_on_the_first_attempt() -> Result {
    identical("seiche", 0.01, |surveyed| {
        let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]).with_surveyed(surveyed);
        case.alpha = Reach::rest(0.5, 1.75);
        case
    })?;
    identical("rest", 0.01, |surveyed| {
        Scenario::rest(1.0, 1.6, 1.0, vec![west("p")]).with_surveyed(surveyed)
    })?;
    for (froude, toward) in [(0.5, true), (0.5, false), (2.0, true), (2.0, false)] {
        identical(&format!("uniform Fr {froude} {toward}"), 0.01, |surveyed| {
            uniform(froude, toward, surveyed)
        })?;
    }
    identical("shock", 0.01, |surveyed| {
        east_case(
            Reach::uniform(1.0, 1.0, 10.0),
            vec![still(1.0); CELLS],
            surveyed,
        )
    })?;
    identical("dry floodplain", 0.01, |surveyed| {
        east_case(
            Reach::uniform(1.0, 0.5, 0.0),
            vec![still(0.0); CELLS],
            surveyed,
        )
    })
}
