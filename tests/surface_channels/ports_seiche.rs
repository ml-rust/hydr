//! Seiches through a reach-end port: reversal, pairing, and the period of a closed basin.
//!
//! The refined case is a closed [`Line`] of a reach and a surface strip, [`SIDE_M`] each, on a
//! flat bed under [`DEPTH_M`] of still water. The stage starts tilted linearly, `a` above the
//! mean at the west wall and `a` below at the east wall, with `a/h = 0.01`. The fundamental
//! period of the basin of length `L` is `2L/sqrt(g h)`. The port sits at its centre.
//!
//! Frozen measurements:
//!
//! - Run end: `1.25` exact periods.
//! - Port rate: the signed port volume of an interval over its duration, placed at the
//!   interval midpoint.
//! - Sign change: two consecutive intervals with rates of strictly opposite sign. Its time is
//!   the linear zero of the rate between the two midpoints.
//! - Period: `2 (t_last - t_first) / (N - 1)` over all `N` sign-change times.

use crate::fixture::Result;
use crate::ports_fixture::{Batch, GRAVITY, Reach, Scenario, balanced, last, step, transfer, west};
use crate::ports_line::{Interval, Line, Order, advance};
use hydr::coupling::SurfaceChannelPortTransfer;

/// Length of each side of the line, in m.
const SIDE_M: f64 = 8.0;

/// Mean still-water depth, in m.
const DEPTH_M: f64 = 1.0;

/// Frozen tilt amplitude over depth.
const TILT: f64 = 0.01;

/// Spacings, coarse to fine, in m.
const SPACINGS: [f64; 2] = [2.0, 1.0];

/// Run length, in exact periods. The port changes sign near each half period.
const PERIODS: f64 = 1.25;

/// Least count of port sign changes.
const SIGN_CHANGES: usize = 2;

/// Relative bound on the period at the finest spacing. The 1 m run measures about `3.8e-3`.
const PERIOD: f64 = 0.006;

/// Accepted intervals of the bank-and-port seiche. The port first reverses at interval 38.
const SEICHE_INTERVALS: usize = 44;

/// Times of every port sign change, in s.
fn sign_changes(intervals: &[Interval]) -> Vec<f64> {
    let rates: Vec<(f64, f64)> = intervals
        .iter()
        .map(|i| {
            (
                0.5 * (i.start_s + i.end_s),
                i.port.signed_volume_m3 / (i.end_s - i.start_s),
            )
        })
        .collect();
    rates
        .windows(2)
        .filter(|w| (w[0].1 > 0.0 && w[1].1 < 0.0) || (w[0].1 < 0.0 && w[1].1 > 0.0))
        .map(|w| {
            let ((t0, r0), (t1, r1)) = (w[0], w[1]);
            t0 + r0 / (r0 - r1) * (t1 - t0)
        })
        .collect()
}

#[test]
fn a_tilted_basin_through_the_port_rings_at_its_period() -> Result {
    tilted_basin_period(false)
}

/// The period of the tilted basin on the rectangular reach, or on its surveyed rectangle.
pub(crate) fn tilted_basin_period(surveyed: bool) -> Result {
    let length = 2.0 * SIDE_M;
    let exact = 2.0 * length / (GRAVITY * DEPTH_M).sqrt();
    let amplitude = TILT * DEPTH_M;
    let mut errors = Vec::with_capacity(SPACINGS.len());
    for dx in SPACINGS {
        let n = (SIDE_M / dx).round() as usize;
        let line = Line::new(Order::ReachFirst, n, dx, |x| {
            DEPTH_M + amplitude * (1.0 - 2.0 * x / length)
        })
        .with_surveyed(surveyed);
        // Every interval checks the pairing, the step mass bound, and the composite balance.
        let (_, intervals) = advance(line.batch()?, PERIODS * exact)?;
        let flags = intervals.iter().filter(|i| i.port.reversed).count();
        let times = sign_changes(&intervals);
        assert_eq!(
            flags,
            times.len(),
            "dx={dx}: reversal flags against sign changes"
        );
        assert!(
            times.len() >= SIGN_CHANGES,
            "dx={dx}: sign changes at {times:?}"
        );
        let span = times[times.len() - 1] - times[0];
        let period = 2.0 * span / (times.len() - 1) as f64;
        let error = (period - exact).abs() / exact;
        println!(
            "Seiche: dx={dx}, sign changes={}, period={period:.12e}, exact={exact:.12e}, relative={error:.6e}",
            times.len()
        );
        errors.push(error);
    }
    assert!(
        errors.windows(2).all(|w| w[1] < w[0]),
        "relative period errors={errors:?}"
    );
    let finest = *errors.last().ok_or("a spacing")?;
    assert!(finest <= PERIOD, "relative period errors={errors:?}");
    Ok(())
}

/// The 4 x 2 case: the reach starts above the surface. Water runs out through the port and
/// the bank link, then back.
///
/// Every interval checks the step mass bound and the lifetime composite balance.
/// Returns the port receipts of every interval and the last batch.
fn seiche(intervals: usize, surveyed: bool) -> Result<(Vec<SurfaceChannelPortTransfer>, Batch)> {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]).with_surveyed(surveyed);
    case.alpha = Reach::rest(0.5, 1.75);
    let mut batch = case.build()?;
    let mut run = Vec::with_capacity(intervals);
    for _ in 0..intervals {
        batch = step(&batch, 0.25)?;
        let diagnostics = last(&batch)?;
        assert!(diagnostics.mass_residual_m3.abs() <= diagnostics.mass_bound_m3);
        balanced(&batch)?;
        run.push(transfer(&batch)?);
    }
    Ok((run, batch))
}

#[test]
fn pairing_and_composite_balance_hold_through_inflow_and_outflow() -> Result {
    pairing_through_inflow_and_outflow(false)
}

/// Pairing through inflow and outflow on the rectangular reach, or on its surveyed rectangle.
pub(crate) fn pairing_through_inflow_and_outflow(surveyed: bool) -> Result {
    let (run, last) = seiche(SEICHE_INTERVALS, surveyed)?;
    let (mut inflow, mut outflow) = (false, false);
    for transfer in &run {
        assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
        inflow |= transfer.signed_volume_m3 > 0.0;
        outflow |= transfer.signed_volume_m3 < 0.0;
    }
    assert!(inflow && outflow, "the seiche must run both ways");
    let ledger = &last.port_ledgers()?[0];
    assert!(ledger.surface_to_channel_m3 > 0.0 && ledger.channel_to_surface_m3 > 0.0);
    Ok(())
}

#[test]
fn a_reversal_sets_the_reversed_flag() -> Result {
    reversal_flag(false)
}

/// The reversal flag on the rectangular reach, or on its surveyed rectangle.
pub(crate) fn reversal_flag(surveyed: bool) -> Result {
    let (run, _) = seiche(SEICHE_INTERVALS, surveyed)?;
    let mut previous: Option<f64> = None;
    let mut reversals = 0;
    for transfer in &run {
        let signed = transfer.signed_volume_m3;
        let flipped =
            previous.is_some_and(|p| (p > 0.0 && signed < 0.0) || (p < 0.0 && signed > 0.0));
        assert_eq!(transfer.reversed, flipped);
        reversals += usize::from(flipped);
        previous = Some(signed);
    }
    assert!(reversals >= 1, "the seiche must reverse the port");
    Ok(())
}
