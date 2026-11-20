//! Profiles standalone channel intervals on fixed reach fixtures.
//!
//! Usage: `channel-profile <rect|triangular|surveyed|trace> <intervals> [midpoint|slack-secant]`.
//! The optional probe sets the root probe of the surveyed and trace reaches. Default: `slack-secant`.
//!
//! Prints one row per accepted interval: duration, allocation calls, and requested bytes.
//! Run it under `perf stat -e instructions:u` with `0` intervals and with `N` intervals:
//! the difference over `N` is the instruction count per accepted interval.
//!
//! `trace` steps the frozen three-band scalar endpoint trace rows, one row per interval.

mod alloc;
mod probe;

#[allow(dead_code)]
#[path = "../../../tests/channel_surveyed/endpoints_reference.rs"]
mod endpoints_reference;
#[allow(dead_code)]
#[path = "../../../tests/support/runtime.rs"]
mod runtime;
#[allow(dead_code)]
#[path = "../../../tests/channel_restart/standalone.rs"]
mod standalone;
#[allow(dead_code)]
#[path = "../../../tests/support/surveyed_profiles.rs"]
mod surveyed_profiles;
mod trace;

use alloc::{Counting, Totals};
use standalone::{Kind, Result, reach_with, step};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Hash of `text`: equal receipts and states give equal digests within one build.
fn digest(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

fn main() -> Result {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: channel-profile <rect|triangular|surveyed|trace> <intervals> [midpoint|slack-secant]";
    let name = args.get(1).ok_or(usage)?;
    let intervals: usize = args.get(2).ok_or(usage)?.parse()?;
    let probe = probe::parse(args.get(3))?;
    if name == "trace" {
        return trace::run(intervals, probe);
    }
    let kind = Kind::parse(name).ok_or_else(|| format!("case={name:?}: {usage}"))?;
    let mut current = reach_with(kind, probe)?;
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
