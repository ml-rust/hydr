//! Dam breaks through a reach-end port against exact Riemann solutions, with refinement.
//!
//! Each refined case is a [`Line`] of [`SIDE_M`] per side on a flat bed, run to [`END_S`] at
//! every spacing of [`SPACINGS`]. Water starts `1 m` deep west of the dam. The other side is
//! dry (Ritter) or `0.25 m` deep (Stoker). The exact solutions come from `analytic.rs`, shifted
//! to the dam. Both waves stay inside the line until [`END_S`].
//!
//! Frozen measurements:
//!
//! - L1 depth error: `Σ |h_i - h(x_i, END_S)| dx` over every composite cell, reach and strip,
//!   with the exact depth sampled at the cell centre `x_i`. The twin uses the same cells.
//! - Observed order: `log2(E_k / E_k+1)` for each of the three halvings.
//! - Port discharge per width: the signed port volume of every interval starting at or after
//!   `END_S / 2`, over the time from the first such start to `END_S`, over the port width,
//!   positive eastward.
//! - Shock position: scanning west from the east end, the first adjacent centre pair with
//!   `h_i >= h_m > h_i+1`, `h_m` midway between the star depth and `0.25 m`, linearly
//!   interpolated between the two centres.

use crate::analytic::{ritter, stoker};
use crate::fixture::Result;
use crate::ports_fixture::{
    ALPHA, GRAVITY, Reach, Scenario, balanced, reach_cells, step, transfer, west,
};
use crate::ports_line::{Interval, Line, Order, WIDTH, advance, twin_depths};
use hydr::coupling::SurfaceChannelPortRegime;

type Reference = fn(f64, f64, f64) -> (f64, f64);

/// Length of each side of the line, in m.
const SIDE_M: f64 = 8.0;

/// Frozen spacings: three halvings, in m.
const SPACINGS: [f64; 4] = [0.5, 0.25, 0.125, 0.0625];

/// Frozen final time, in s.
const END_S: f64 = 1.0;

/// Distance of the dam west of the port in the crossing cases, in m.
const OFFSET_M: f64 = 1.0;

/// Frozen bound on the coupled L1 error against the twin's at the same spacing.
const TWIN_RATIO: f64 = 1.25;

/// Frozen floor of the observed L1 order of each halving.
const ORDER: f64 = 0.5;

/// Frozen relative bound on the port discharge per width at the finest spacing.
const SONIC: f64 = 0.05;

/// Frozen bound on the shock position error, in cells.
const SHOCK_CELLS: f64 = 2.0;

/// Stoker depth east of the dam, in m.
const STOKER_RIGHT_M: f64 = 0.25;

/// One refined dam break at spacing `dx`, run to [`END_S`].
struct Run {
    line: Line,
    coupled: Vec<f64>,
    twin: Vec<f64>,
    intervals: Vec<Interval>,
}

fn run(order: Order, dx: f64, dam: f64, exact: Reference) -> Result<Run> {
    let n = (SIDE_M / dx).round() as usize;
    let line = Line::new(order, n, dx, |x| exact(x - dam, 0.0, GRAVITY).0);
    let (batch, intervals) = advance(line.batch()?, END_S)?;
    let coupled = line.depths_of(&batch)?;
    let twin = twin_depths(line.twin()?, END_S)?;
    assert!(coupled.iter().all(|&h| h >= 0.0));
    Ok(Run {
        line,
        coupled,
        twin,
        intervals,
    })
}

/// L1 depth error of `depths` at [`END_S`], in m².
fn l1(line: &Line, depths: &[f64], dam: f64, exact: Reference) -> f64 {
    line.centres()
        .iter()
        .zip(depths)
        .map(|(&x, &h)| (h - exact(x - dam, END_S, GRAVITY).0).abs() * line.dx)
        .sum()
}

/// Observed order of each halving.
fn orders(errors: &[f64]) -> Vec<f64> {
    errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
}

/// Runs every spacing, checks the twin ratio and the order, and returns the runs.
fn crossing(label: &str, order: Order, exact: Reference) -> Result<Vec<Run>> {
    let dam = SIDE_M - OFFSET_M;
    let mut runs = Vec::with_capacity(SPACINGS.len());
    let mut errors = Vec::with_capacity(SPACINGS.len());
    for dx in SPACINGS {
        let run = run(order, dx, dam, exact)?;
        let coupled = l1(&run.line, &run.coupled, dam, exact);
        let twin = l1(&run.line, &run.twin, dam, exact);
        println!("{label}: dx={dx}, coupled L1={coupled:.12e}, twin L1={twin:.12e}");
        assert!(
            coupled <= TWIN_RATIO * twin,
            "{label}: dx={dx}: coupled {coupled} against twin {twin}"
        );
        errors.push(coupled);
        runs.push(run);
    }
    let observed = orders(&errors);
    println!("{label}: orders={observed:?}");
    assert!(
        observed.iter().all(|&p| p >= ORDER),
        "{label}: errors={errors:?}, orders={observed:?}"
    );
    Ok(runs)
}

#[test]
fn ritter_from_the_reach_onto_a_dry_strip_matches_the_twin_and_refines() -> Result {
    crossing("Ritter reach to strip", Order::ReachFirst, ritter).map(drop)
}

#[test]
fn ritter_from_the_strip_into_a_dry_reach_matches_the_twin_and_refines() -> Result {
    crossing("Ritter strip to reach", Order::SurfaceFirst, ritter).map(drop)
}

/// Port discharge per width over the second half of the run, positive eastward, in m²/s.
fn port_discharge(order: Order, intervals: &[Interval]) -> Result<f64> {
    let window: Vec<&Interval> = intervals
        .iter()
        .filter(|interval| interval.start_s >= 0.5 * END_S)
        .collect();
    let first = window.first().ok_or("an interval in the window")?.start_s;
    let volume: f64 = window.iter().map(|i| i.port.signed_volume_m3).sum();
    // Positive signed volume runs surface to channel.
    let eastward = match order {
        Order::ReachFirst => -volume,
        Order::SurfaceFirst => volume,
    };
    Ok(eastward / ((END_S - first) * WIDTH))
}

/// Ritter's dam at the port face: the sonic point sits on the port.
fn sonic(label: &str, order: Order) -> Result {
    let exact = 8.0 / 27.0 * GRAVITY.sqrt();
    let mut errors = Vec::with_capacity(SPACINGS.len());
    for dx in SPACINGS {
        let run = run(order, dx, SIDE_M, ritter)?;
        let q = port_discharge(order, &run.intervals)?;
        let error = (q - exact).abs() / exact;
        println!("{label}: dx={dx}, q={q:.12e}, exact={exact:.12e}, relative={error:.6e}");
        errors.push(error);
    }
    assert!(
        errors.windows(2).all(|w| w[1] < w[0]),
        "{label}: relative errors={errors:?}"
    );
    let finest = *errors.last().ok_or("a spacing")?;
    assert!(finest <= SONIC, "{label}: relative errors={errors:?}");
    Ok(())
}

#[test]
fn a_ritter_dam_at_the_reach_end_passes_the_sonic_discharge() -> Result {
    sonic("Sonic reach to strip", Order::ReachFirst)
}

#[test]
fn a_ritter_dam_at_the_strip_edge_passes_the_sonic_discharge() -> Result {
    sonic("Sonic strip to reach", Order::SurfaceFirst)
}

/// Numerical shock position of `run`, in m.
fn shock(run: &Run, mid: f64) -> Result<f64> {
    let centres = run.line.centres();
    (0..centres.len() - 1)
        .rev()
        .find(|&i| run.coupled[i] >= mid && mid > run.coupled[i + 1])
        .map(|i| {
            let (a, b) = (run.coupled[i], run.coupled[i + 1]);
            centres[i] + (a - mid) / (a - b) * (centres[i + 1] - centres[i])
        })
        .ok_or_else(|| "no shock in the depths".into())
}

/// Stoker with the shock crossing the port from the west side.
fn stoker_crossing(label: &str, order: Order) -> Result {
    let runs = crossing(label, order, stoker)?;
    let (star_h, star_q) = stoker(0.0, 1.0, GRAVITY);
    let speed = star_q / (star_h - STOKER_RIGHT_M);
    let exact = SIDE_M - OFFSET_M + speed * END_S;
    assert!(exact > SIDE_M, "{label}: the shock must cross the port");
    let mid = 0.5 * (star_h + STOKER_RIGHT_M);
    for run in &runs {
        let position = shock(run, mid)?;
        let dx = run.line.dx;
        println!("{label}: dx={dx}, shock={position:.12e}, exact={exact:.12e}");
        assert!(
            (position - exact).abs() <= SHOCK_CELLS * dx,
            "{label}: dx={dx}: shock {position} against {exact}"
        );
    }
    Ok(())
}

#[test]
fn stoker_from_the_reach_crosses_the_port_and_refines() -> Result {
    stoker_crossing("Stoker reach to strip", Order::ReachFirst)
}

#[test]
fn stoker_from_the_strip_crosses_the_port_and_refines() -> Result {
    stoker_crossing("Stoker strip to reach", Order::SurfaceFirst)
}

#[test]
fn a_dam_break_onto_a_dry_floodplain_runs_and_balances() -> Result {
    // A still reach releases onto the dry 4 x 2 floodplain through its port.
    let mut case = Scenario::rest(1.0, 0.0, 1.0, vec![west("p")]);
    case.link = false;
    case.alpha = Reach::uniform(1.0, 0.5, 0.0);
    let mut batch = case.build()?;
    let held: f64 = reach_cells(&batch, ALPHA)?
        .iter()
        .map(|c| c.volume_m3)
        .sum();
    let mut released = 0.0;
    for index in 0..20 {
        batch = step(&batch, 0.1)?;
        let transfer = transfer(&batch)?;
        if index == 0 {
            assert_eq!(transfer.regime, SurfaceChannelPortRegime::WetDryFront);
        }
        assert!(transfer.signed_volume_m3 <= 0.0);
        released -= transfer.signed_volume_m3;
        assert!(
            reach_cells(&batch, ALPHA)?
                .iter()
                .all(|c| c.volume_m3 >= 0.0)
        );
        assert!(batch.surface().state()?.cells().iter().all(|c| c.h >= 0.0));
        balanced(&batch)?;
    }
    assert!(released > 0.0 && released < held);
    Ok(())
}
