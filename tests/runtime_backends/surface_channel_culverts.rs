//! Culvert cases of the surface/channel coordinator on configured runtime clients.
//!
//! Frozen criterion, fixed before the first run: each case runs on the CPU client and on the
//! CUDA client over the same intervals. After every accepted interval, the text of every child
//! state is equal between the two runs. Equal text means equal bits, signed zero included. The
//! text holds the time, the accepted count, the full checkpoint package, every ledger, the live
//! pool and surface states, and the last interval receipt.
//!
//! The cases are the tide through a two-way and a one-end culvert, the sweep from inlet to
//! outlet control, the partly full outlet that a basin drowns, the culvert beside a road weir,
//! and the pipe-arch drain. Each culvert must carry water.

use super::Result;
use crate::culverts_fixture::{
    drowning_case, pipe_arch_case, reversal_case, roadway_case, sweep_case, tide,
};
use crate::pools_fixture::Case;

/// One culvert case: its builder, its tide, its longest interval, and its end, in s.
///
/// A child step limit can shorten an interval, so a run counts time, not intervals.
pub struct Run {
    pub case: fn() -> crate::pools_fixture::Result<Case>,
    /// Tide rate on the reach cell at a time, in m³/s.
    pub tide: fn(f64) -> f64,
    pub dt: f64,
    pub end_s: f64,
}

fn still(_: f64) -> f64 {
    0.0
}

/// A flood for one interval, then an ebb that drops the reach below the lagoon.
fn short_tide(t: f64) -> f64 {
    if t < 0.5 { 0.1 } else { -0.1 }
}

/// The five culvert cases with the interval counts of the parity runs.
///
/// Parity needs every law path once, not a long run: the tide both ways, inlet control into a
/// dry basin, a partly full outlet whose one interval runs cold and warm root searches, a
/// drowned barrel under a road, and a pipe-arch barrel draining into a dry basin.
pub fn runs() -> [Run; 5] {
    [
        Run {
            case: reversal_case,
            tide: short_tide,
            dt: 0.5,
            end_s: 2.0,
        },
        Run {
            case: sweep_case,
            tide: still,
            dt: 0.5,
            end_s: 1.0,
        },
        Run {
            case: drowning_case,
            tide: still,
            dt: 0.5,
            end_s: 0.5,
        },
        Run {
            case: roadway_case,
            tide: still,
            dt: 0.5,
            end_s: 1.0,
        },
        Run {
            case: pipe_arch_case,
            tide: still,
            dt: 0.5,
            end_s: 1.0,
        },
    ]
}

/// Runs `run` on one client and returns the state text after every accepted interval.
///
/// Fails unless every structure carries water over the run.
macro_rules! culvert_execution {
    ($runtime:ty, $client:expr, $run:expr) => {{
        use hydr::coupling::SurfaceChannelBatch;

        let run: &Run = $run;
        let (setup, initial) = (run.case)()?.parts()?;
        let mut batch = SurfaceChannelBatch::<$runtime>::new($client, setup, initial)?;
        let mut states = Vec::new();
        while batch.time() < run.end_s {
            let start = batch.time();
            let end = (start + run.dt).min(run.end_s);
            let held = tide(batch.setup(), (start, end), (run.tide)(start));
            batch = batch.trial_to(end, &held)?.into_next();
            let pools = batch.pools().map(|pools| pools.states()).transpose()?;
            states.push(format!(
                "{} {} {:?} {:?} {:?} {:?} {:?} {:?}",
                batch.time(),
                batch.accepted_steps(),
                batch.snapshot_data()?,
                batch.ledger()?,
                batch.structure_ledgers()?,
                pools,
                batch.surface().state()?,
                batch.last_step()?,
            ));
        }
        for ledger in batch.structure_ledgers()? {
            assert!(
                ledger.a_to_b_m3 > 0.0 || ledger.b_to_a_m3 > 0.0,
                "structure {} carried no water: {ledger:?}",
                ledger.structure_id
            );
        }
        states
    }};
}

#[test]
fn selected_cpu_client_executes_surface_channel_culverts() -> Result {
    for run in runs() {
        let states = culvert_execution!(super::runtime::Runtime, super::runtime::client(), &run);
        assert!(!states.is_empty());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_surface_channel_culverts() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    for (index, run) in runs().iter().enumerate() {
        let cpu = culvert_execution!(super::runtime::Runtime, super::runtime::client(), run);
        let cuda = culvert_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?, run);
        assert_eq!(
            cpu.len(),
            cuda.len(),
            "case {index}: accepted interval count"
        );
        for (interval, (cpu, cuda)) in cpu.iter().zip(&cuda).enumerate() {
            super::same_bits(cpu, cuda)
                .map_err(|error| format!("case {index}, interval {}: {error}", interval + 1))?;
        }
    }
    Ok(())
}
