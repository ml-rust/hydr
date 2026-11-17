//! Profiles standalone channel intervals on fixed reach fixtures.
//!
//! Usage: `channel-profile <rect|triangular|surveyed> <intervals>`.
//!
//! Prints one row per accepted interval: duration, allocation calls, and requested bytes.
//! Run it under `perf stat -e instructions:u` with `0` intervals and with `N` intervals:
//! the difference over `N` is the instruction count per accepted interval.

mod alloc;

#[allow(dead_code)]
#[path = "../../../tests/support/runtime.rs"]
mod runtime;
#[allow(dead_code)]
#[path = "../../../tests/channel_restart/standalone.rs"]
mod standalone;

use alloc::{Counting, Totals};
use standalone::{Kind, Result, reach, step};

#[global_allocator]
static GLOBAL: Counting = Counting;

fn main() -> Result {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: channel-profile <rect|triangular|surveyed> <intervals>";
    let name = args.get(1).ok_or(usage)?;
    let intervals: usize = args.get(2).ok_or(usage)?.parse()?;
    let kind = Kind::parse(name).ok_or_else(|| format!("case={name:?}: {usage}"))?;
    let mut current = reach(kind)?;
    println!("interval dt_s alloc_calls alloc_bytes root_solves root_iterations residual_points");
    for index in 0..intervals {
        let start = current.time();
        let before = Totals::now();
        current = step(kind, &current)?;
        let allocations = Totals::now().since(before);
        let data = current.snapshot_data()?;
        // Root work of the accepted-start face solve: zero solves for analytical fluxes.
        let riemann = data
            .last_interval
            .and_then(|interval| interval.diagnostics.riemann);
        let (solves, iterations, points) = match riemann {
            Some(riemann) => match riemann.surveyed_work {
                Some(work) => (work.root_solves, work.root_iterations, work.residual_points),
                None => (1, riemann.root_iterations, 0),
            },
            None => (0, 0, 0),
        };
        println!(
            "{index} {:e} {} {} {solves} {iterations} {points}",
            current.time() - start,
            allocations.calls,
            allocations.bytes
        );
    }
    Ok(())
}
