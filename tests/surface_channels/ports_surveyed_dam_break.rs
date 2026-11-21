//! Dam breaks and seiches through a reach-end port on a surveyed rectangle.
//!
//! Frozen criterion: every frozen criterion of `ports_dam_break.rs` and
//! `ports_seiche.rs` holds with the reach declared as a surveyed rectangle of the strip width.
//! The L1 depth difference between the surveyed and the rectangular run is at most the
//! rectangular run's own L1 depth error against the analytic solution, at every spacing.
//!
//! The seiche has no analytic depth field in `ports_seiche.rs`. Its criteria are the period,
//! the pairing, and the reversal flag.

use crate::analytic::{ritter, stoker};
use crate::fixture::Result;
use crate::ports_dam_break::{
    OFFSET_M, Reference, Run, SIDE_M, SPACINGS, crossing, dry_floodplain, l1, run, sonic,
    stoker_crossing,
};
use crate::ports_line::Order;
use crate::ports_seiche::{pairing_through_inflow_and_outflow, reversal_flag, tilted_basin_period};

/// Fails unless the surveyed and rectangular depths differ by no more than the rectangular
/// run's own L1 error, for a dam at `dam` metres, at every spacing. `surveyed` holds the
/// surveyed run of each spacing in [`SPACINGS`] order.
fn within_rectangular_error(
    label: &str,
    order: Order,
    dam: f64,
    exact: Reference,
    surveyed: &[Run],
) -> Result {
    assert_eq!(surveyed.len(), SPACINGS.len());
    for (dx, surveyed) in SPACINGS.into_iter().zip(surveyed) {
        let rectangular = run(order, dx, dam, exact, false)?;
        let own = l1(&rectangular.line, &rectangular.coupled, dam, exact);
        let difference: f64 = rectangular
            .coupled
            .iter()
            .zip(&surveyed.coupled)
            .map(|(a, b)| (a - b).abs() * dx)
            .sum();
        assert!(
            difference <= own,
            "{label}: dx={dx}: surveyed against rectangular {difference}, rectangular against exact {own}"
        );
    }
    Ok(())
}

/// The frozen crossing criteria on the surveyed reach, then the rectangular-run difference.
fn surveyed_crossing(label: &str, order: Order, exact: Reference) -> Result {
    let runs = crossing(label, order, exact, true)?;
    within_rectangular_error(label, order, SIDE_M - OFFSET_M, exact, &runs)
}

#[test]
fn surveyed_ritter_from_the_reach_onto_a_dry_strip_refines_and_matches_the_rectangular_run()
-> Result {
    surveyed_crossing("Surveyed Ritter reach to strip", Order::ReachFirst, ritter)
}

#[test]
fn surveyed_ritter_from_the_strip_into_a_dry_reach_refines_and_matches_the_rectangular_run()
-> Result {
    surveyed_crossing(
        "Surveyed Ritter strip to reach",
        Order::SurfaceFirst,
        ritter,
    )
}

#[test]
fn surveyed_stoker_from_the_reach_crosses_the_port_and_matches_the_rectangular_run() -> Result {
    let runs = stoker_crossing("Surveyed Stoker reach to strip", Order::ReachFirst, true)?;
    within_rectangular_error(
        "Surveyed Stoker reach to strip",
        Order::ReachFirst,
        SIDE_M - OFFSET_M,
        stoker,
        &runs,
    )
}

#[test]
fn surveyed_stoker_from_the_strip_crosses_the_port_and_matches_the_rectangular_run() -> Result {
    let runs = stoker_crossing("Surveyed Stoker strip to reach", Order::SurfaceFirst, true)?;
    within_rectangular_error(
        "Surveyed Stoker strip to reach",
        Order::SurfaceFirst,
        SIDE_M - OFFSET_M,
        stoker,
        &runs,
    )
}

#[test]
fn surveyed_ritter_dam_at_the_reach_end_passes_the_sonic_discharge() -> Result {
    let runs = sonic("Surveyed sonic reach to strip", Order::ReachFirst, true)?;
    within_rectangular_error(
        "Surveyed sonic reach to strip",
        Order::ReachFirst,
        SIDE_M,
        ritter,
        &runs,
    )
}

#[test]
fn surveyed_ritter_dam_at_the_strip_edge_passes_the_sonic_discharge() -> Result {
    let runs = sonic("Surveyed sonic strip to reach", Order::SurfaceFirst, true)?;
    within_rectangular_error(
        "Surveyed sonic strip to reach",
        Order::SurfaceFirst,
        SIDE_M,
        ritter,
        &runs,
    )
}

#[test]
fn surveyed_dam_break_onto_a_dry_floodplain_runs_and_balances() -> Result {
    dry_floodplain(true)
}

#[test]
fn surveyed_tilted_basin_through_the_port_rings_at_its_period() -> Result {
    tilted_basin_period(true)
}

#[test]
fn surveyed_pairing_and_composite_balance_hold_through_inflow_and_outflow() -> Result {
    pairing_through_inflow_and_outflow(true)
}

#[test]
fn surveyed_reversal_sets_the_reversed_flag() -> Result {
    reversal_flag(true)
}
