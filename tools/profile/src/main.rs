//! Profiles surface/channel common intervals on fixed coupled cases.
//!
//! Usage: `surface-channel-profile <rect|surveyed|flood> <intervals> [cpu|cuda]`.
//!
//! Prints one row per accepted interval: duration, retries, fixed-point iterations, allocation
//! calls, and requested bytes. Run it under `perf stat -e instructions:u` with `0` intervals and
//! with `N` intervals: the difference over `N` is the instruction count per accepted interval.

mod alloc;

#[allow(dead_code)]
#[path = "../../../tests/surface_channels/fixture.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../../../tests/support/runtime.rs"]
mod runtime;
#[allow(dead_code)]
#[path = "../../../tests/surface_channels/scenario.rs"]
mod scenario;

use alloc::{Counting, Totals};
use fixture::Result;
use hydr::channel::{ChannelBoundary, ChannelLateralDirection, ChannelLateralInput};
use hydr::coupling::{
    SurfaceChannelBatch, SurfaceChannelForcingInterval, SurfaceChannelInitialState,
    SurfaceChannelReachForcing, SurfaceChannelSetup,
};
use hydr::forcing::WaterAmount;
use hydr::solver::{ForcingInterval, ForcingSchedule};
use scenario::{ReachSpec, Spec, weir};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Tributary inflow binding of the surveyed case.
const TRIBUTARY: &str = "tributary";

type Parts = (SurfaceChannelSetup, SurfaceChannelInitialState);

/// One fixed coupled case and its caller forcing.
struct Case {
    parts: fn() -> Result<Parts>,
    /// Caller inflow rate on every inflow binding, in m³/s.
    inflow: f64,
}

/// Two rectangular reaches beside a spilling 3 x 2 surface; one link reverses, one spills.
fn rectangular() -> Result<Parts> {
    Spec::new(3, 2, 1.0, 0.4)
        .reach(ReachSpec::new("north", 0.0, &[0.5, 0.5, 0.5]))
        .reach(ReachSpec::new("south", 0.0, &[1.5, 1.5]))
        .link(weir("n-0", 0, "north", 0, 1.2, 2.0))
        .link(weir("n-1", 1, "north", 2, 1.2, 2.0))
        .link(weir("s-0", 3, "south", 0, 1.2, 2.0))
        .link(weir("s-1", 4, "south", 1, 1.2, 2.0))
        .parts()
}

/// A tributary floods a surveyed reach over a 1.05 m bank crest onto a dry surface cell.
fn surveyed() -> Result<Parts> {
    let mut spec = Spec::new(1, 1, 1.0, 0.0)
        .reach(
            ReachSpec::new("river", 0.0, &[1.0, 1.0])
                .surveyed()
                .binding(TRIBUTARY, 1, ChannelLateralDirection::Inflow),
        )
        .link(weir("bank", 0, "river", 0, 1.05, 2.0));
    spec.config.head_absolute_m = 1e-10;
    spec.config.transfer_absolute_m3 = 1e-10;
    spec.parts()
}

/// Surface stage 2 m spills into a reach at 1.3 m, then evaporation reverses the link at 0.5 s.
fn flood() -> Result<Parts> {
    let spec = Spec::new(1, 1, 1.0, 1.0)
        .reach(ReachSpec::new("river", 0.0, &[1.3, 1.3]))
        .link(weir("bank", 0, "river", 0, 1.2, 2.0));
    let (mut setup, initial) = spec.parts()?;
    let surface = &mut setup.surface;
    let interval = |start, end, evaporation| ForcingInterval {
        start,
        end,
        rain: vec![0.0],
        potential_evaporation: vec![evaporation],
        withdrawal: vec![0.0],
        boundaries: Vec::new(),
    };
    surface.forcing = ForcingSchedule::new(
        &surface.domain,
        &surface.boundaries,
        vec![interval(0.0, 0.5, 0.0), interval(0.5, 100.0, 2.0)],
    )?;
    Ok((setup, initial))
}

/// Walls, `inflow` on every inflow binding, and zero withdrawals from `start` to `end`.
fn forcing(
    setup: &SurfaceChannelSetup,
    start: f64,
    end: f64,
    inflow: f64,
) -> SurfaceChannelForcingInterval {
    let rate = |rate_m3_per_s| WaterAmount::Rate { rate_m3_per_s };
    let reaches = setup
        .reaches
        .iter()
        .map(|reach| SurfaceChannelReachForcing {
            reach_id: reach.id.clone(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: reach
                .definition
                .lateral_bindings
                .iter()
                .map(|binding| match binding.direction {
                    ChannelLateralDirection::Inflow => ChannelLateralInput::Inflow {
                        boundary_id: binding.boundary_id.clone(),
                        amount: rate(inflow),
                        longitudinal_velocity_m_per_s: 0.0,
                    },
                    ChannelLateralDirection::Withdrawal => ChannelLateralInput::Withdrawal {
                        boundary_id: binding.boundary_id.clone(),
                        amount: rate(0.0),
                    },
                })
                .collect(),
        })
        .collect();
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches,
        pool_sources: None,
        pool_inflows: Vec::new(),
    }
}

/// Receipt of one accepted interval.
struct Row {
    dt_s: f64,
    retries: usize,
    iterations: usize,
    allocations: Totals,
}

/// Accepts `$intervals` common intervals of `$case` on `$client`, one second requested each.
macro_rules! profile {
    ($runtime:ty, $client:expr, $case:expr, $intervals:expr) => {{
        let case: &Case = $case;
        let (setup, initial) = (case.parts)()?;
        let mut batch = SurfaceChannelBatch::<$runtime>::new($client, setup, initial)?;
        let mut rows = Vec::with_capacity($intervals);
        for _ in 0..$intervals {
            let start = batch.time();
            let end = start + 1.0;
            let interval = forcing(batch.setup(), start, end, case.inflow);
            let before = Totals::now();
            batch = batch.trial_to(end, &interval)?.into_next();
            let allocations = Totals::now().since(before);
            let step = batch.last_step()?.ok_or("missing interval receipt")?;
            rows.push(Row {
                dt_s: step.end_s - step.start_s,
                retries: step.retries,
                iterations: step.iterations,
                allocations,
            });
        }
        rows
    }};
}

fn main() -> Result {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: surface-channel-profile <rect|surveyed|flood> <intervals> [cpu|cuda]";
    let name = args.get(1).ok_or(usage)?;
    let intervals: usize = args.get(2).ok_or(usage)?.parse()?;
    let device = args.get(3).map_or("cpu", String::as_str);
    let case = match name.as_str() {
        "rect" => Case {
            parts: rectangular,
            inflow: 0.0,
        },
        "surveyed" => Case {
            parts: surveyed,
            inflow: 4.0,
        },
        "flood" => Case {
            parts: flood,
            inflow: 0.0,
        },
        other => return Err(format!("case={other:?}: {usage}").into()),
    };
    let rows = match device {
        "cpu" => profile!(runtime::Runtime, runtime::client(), &case, intervals),
        #[cfg(feature = "cuda")]
        "cuda" => {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            profile!(
                CudaRuntime,
                CudaClient::new(CudaDevice::new(0))?,
                &case,
                intervals
            )
        }
        other => return Err(format!("device={other:?}: {usage}").into()),
    };
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
            "mean retries={} iterations={} alloc_calls={} alloc_bytes={}",
            mean(&|r| r.retries as f64),
            mean(&|r| r.iterations as f64),
            mean(&|r| r.allocations.calls as f64),
            mean(&|r| r.allocations.bytes as f64),
        );
    }
    Ok(())
}
