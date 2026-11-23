//! Culvert cases of the coupled level pools on configured runtime clients.
//!
//! Frozen criterion, fixed before the first run: each case runs on the CPU client and on the
//! CUDA client over the same intervals. After every accepted interval, the text of the package,
//! the external ledgers, and the live pool states is equal between the two runs. Equal text
//! means equal bits, signed zero included.
//!
//! The cases are the rising sea through a two-way and a one-end culvert, and the culvert beside
//! a road weir. Each structure must carry water.

#[allow(dead_code)]
#[path = "../coupled_level_pools/culverts_fixture.rs"]
mod culverts_fixture;
use crate::coupled_pool_fixture as fixture;

use super::Result;
use culverts_fixture::{blocked_culvert, reversal, roadway};
use hydr::coupling::{
    CoupledLevelPoolForcingInterval, CoupledLevelPoolInitialState, CoupledLevelPoolSetup,
    CoupledPoolInflow, CoupledPoolInflowValue,
};

type Parts = (CoupledLevelPoolSetup, CoupledLevelPoolInitialState);

/// Requested duration of every interval, in s.
const DT: f64 = 0.5;

/// Accepted intervals of every run.
const INTERVALS: usize = 8;

/// Forcing of the interval from `start` to `end`: 0.2 m³/s into every declared boundary.
fn forcing(setup: &CoupledLevelPoolSetup, start: f64, end: f64) -> CoupledLevelPoolForcingInterval {
    CoupledLevelPoolForcingInterval {
        start_s: start,
        end_s: end,
        sources: None,
        inflows: setup
            .external_inflows
            .iter()
            .map(|binding| CoupledPoolInflow {
                boundary_id: binding.boundary_id.clone(),
                input_id: binding.input_id.clone(),
                value: CoupledPoolInflowValue::Rate { rate_m3_per_s: 0.2 },
            })
            .collect(),
    }
}

/// Runs `parts` on one client and returns the state text after every accepted interval.
///
/// Fails unless every structure carries water over the run.
macro_rules! coupled_culvert_execution {
    ($runtime:ty, $client:expr, $parts:expr) => {{
        use hydr::coupling::CoupledLevelPoolBatch;

        let (setup, initial): Parts = $parts;
        let mut batch = CoupledLevelPoolBatch::<$runtime>::new($client, setup, initial)?;
        let mut states = Vec::new();
        let mut moved = vec![false; batch.setup().structures.len()];
        for _ in 0..INTERVALS {
            let start = batch.time();
            let end = start + DT;
            let held = forcing(batch.setup(), start, end);
            batch = batch.trial_with_forcing_to(end, &held)?.into_next();
            let step = batch.last_step()?.ok_or("an accepted interval")?;
            for (flag, transfer) in moved.iter_mut().zip(&step.transfers) {
                *flag |= transfer.signed_volume_m3 != 0.0;
            }
            states.push(format!(
                "{} {:?} {:?} {:?}",
                batch.time(),
                batch.snapshot_data()?,
                batch.external_inflows()?,
                batch.pools().states()?,
            ));
        }
        assert!(
            moved.iter().all(|flag| *flag),
            "a structure carried no water"
        );
        states
    }};
}

#[test]
fn selected_cpu_client_executes_coupled_level_pool_culverts() -> Result {
    for parts in [reversal()?, roadway()?, blocked_culvert()?] {
        let states =
            coupled_culvert_execution!(super::runtime::Runtime, super::runtime::client(), parts);
        assert_eq!(states.len(), INTERVALS);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_coupled_level_pool_culverts() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    for (index, build) in [reversal, roadway, blocked_culvert].into_iter().enumerate() {
        let cpu =
            coupled_culvert_execution!(super::runtime::Runtime, super::runtime::client(), build()?);
        let cuda =
            coupled_culvert_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?, build()?);
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
