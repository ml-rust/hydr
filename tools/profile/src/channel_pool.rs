//! Profiles channel/pool linear-mode oscillation intervals, surveyed and rectangular.
//!
//! Usage: `channel-pool-profile <surveyed-mode|rect-mode> <cells> <intervals>`.
//!
//! Each interval is a `trial_to` toward the half period, accepted with `into_next`, as in
//! `tests/channel_pools/surveyed_mode.rs` and `linear_mode.rs`. Prints one row per accepted
//! interval: duration, retries, fixed-point iterations, allocation calls, and requested bytes,
//! then a mean row. Run it under `perf stat -e instructions:u` with `0` intervals and with `N`
//! intervals: the difference over `N` is the instruction count per accepted interval.
//!
//! `surveyed-mode` is the surveyed `RECTANGLE_1` reach at `cfl 0.1`. `rect-mode` is the
//! analytical rectangular reach at `cfl 0.4`.

mod alloc;

#[allow(dead_code)]
#[path = "../../../tests/channel_pools/fixture.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../../../tests/channel_pools/reference.rs"]
mod reference;
#[allow(dead_code)]
#[path = "../../../tests/support/runtime.rs"]
mod runtime;
#[allow(dead_code)]
#[path = "../../../tests/channel_pools/surveyed_fixture.rs"]
mod surveyed_fixture;
#[allow(dead_code)]
#[path = "../../../tests/support/surveyed_profiles.rs"]
mod surveyed_profiles;
#[allow(dead_code)]
#[path = "../../../tests/channel_pools/surveyed_reference.rs"]
mod surveyed_reference;

use alloc::{Counting, Totals};
use fixture::{Batch, Result, batch, forcing};
use hydr::channel::{ChannelBoundary, ChannelInitialState};
use hydr::coupling::{ChannelPoolEndpoint, ChannelPoolInitialState, ChannelPoolSetup};
use reference::CASES;
use surveyed_fixture::{definition, single, surveyed_case};
use surveyed_profiles::RECTANGLE_1;

#[global_allocator]
static GLOBAL: Counting = Counting;

const AMPLITUDE: f64 = 1e-5;
const K: f64 = 0.20287578381104342;
const HALF: f64 = 15.48530137296151;

/// Cell average of the cosine mode over `[left, right]`.
fn average_cos(left: f64, right: f64) -> f64 {
    ((K * right).sin() - (K * left).sin()) / (K * (right - left))
}

/// Initial cosine volumes of `count` cells of length `dx`.
fn initial_reach(count: usize, dx: f64) -> ChannelInitialState {
    ChannelInitialState {
        time_s: 0.0,
        volume_m3: (0..count)
            .map(|i| dx * (1.0 + AMPLITUDE * average_cos(i as f64 * dx, (i + 1) as f64 * dx)))
            .collect(),
        longitudinal_momentum_m4_per_s: vec![0.0; count],
    }
}

type Parts = (ChannelPoolSetup, ChannelPoolInitialState);

/// The setup of `measure` in `tests/channel_pools/surveyed_mode.rs`.
fn surveyed_mode(count: usize) -> Result<Parts> {
    let mut case = surveyed_case(&CASES[0]);
    case.profile = RECTANGLE_1;
    case.gravity = 1.0;
    case.interior_area_m2 = 1.0;
    case.elevation = 1.0 + AMPLITUDE * (10.0 * K).cos();
    case.pool_area = 10.0;
    case.loss = 0.0;
    let (mut setup, mut initial) = single(&case, ChannelPoolEndpoint::Right)?;
    let dx = 10.0 / count as f64;
    setup.reaches[0].definition = definition(RECTANGLE_1, 1.0, count, dx);
    setup.reaches[0].definition.config.cfl = 0.1;
    initial.reaches[0] = initial_reach(count, dx);
    Ok((setup, initial))
}

/// The setup of `measure` in `tests/channel_pools/linear_mode.rs`.
fn rect_mode(count: usize) -> Result<Parts> {
    let mut case = CASES[0];
    case.coefficient = 1.0;
    case.elevation = 1.0 + AMPLITUDE * (10.0 * K).cos();
    case.pool_area = 10.0;
    case.loss = 0.0;
    let (mut setup, mut initial) = fixture::single(case, ChannelPoolEndpoint::Right)?;
    let dx = 10.0 / count as f64;
    setup.reaches[0].definition = fixture::definition(false, 1.0, 1.0, count, dx, 0.0);
    setup.reaches[0].definition.config.cfl = 0.4;
    initial.reaches[0] = initial_reach(count, dx);
    Ok((setup, initial))
}

/// Receipt of one accepted interval.
struct Row {
    dt_s: f64,
    retries: usize,
    iterations: usize,
    allocations: Totals,
}

fn main() -> Result {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: channel-pool-profile <surveyed-mode|rect-mode> <cells> <intervals>";
    let name = args.get(1).ok_or(usage)?;
    let count: usize = args.get(2).ok_or(usage)?.parse()?;
    let intervals: usize = args.get(3).ok_or(usage)?.parse()?;
    let (setup, initial) = match name.as_str() {
        "surveyed-mode" => surveyed_mode(count)?,
        "rect-mode" => rect_mode(count)?,
        other => return Err(format!("case={other:?}: {usage}").into()),
    };
    let mut current: Batch = batch(setup.clone(), initial)?;
    let mut rows = Vec::with_capacity(intervals);
    for index in 0..intervals {
        let start = current.time();
        if start >= HALF {
            return Err(format!("interval={index}: time reached the half period").into());
        }
        let mut held = forcing(&setup, start, HALF);
        held.reaches[0].left = Some(ChannelBoundary::Wall);
        let before = Totals::now();
        current = current.trial_to(HALF, &held)?.into_next();
        let allocations = Totals::now().since(before);
        let step = current.last_step()?.ok_or("missing interval receipt")?;
        rows.push(Row {
            dt_s: step.end_s - step.start_s,
            retries: step.retries,
            iterations: step.iterations,
            allocations,
        });
    }
    println!("interval dt_s retries iterations alloc_calls alloc_bytes");
    for (index, row) in rows.iter().enumerate() {
        println!(
            "{index} {:e} {} {} {} {}",
            row.dt_s, row.retries, row.iterations, row.allocations.calls, row.allocations.bytes
        );
    }
    if !rows.is_empty() {
        let count = rows.len() as f64;
        let mean = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / count;
        println!(
            "mean dt_s={:e} retries={} iterations={} alloc_calls={} alloc_bytes={}",
            mean(&|r| r.dt_s),
            mean(&|r| r.retries as f64),
            mean(&|r| r.iterations as f64),
            mean(&|r| r.allocations.calls as f64),
            mean(&|r| r.allocations.bytes as f64),
        );
    }
    Ok(())
}
