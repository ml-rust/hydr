//! Native runoff of the surface/channel coordinator on configured runtime clients.
//!
//! Frozen criterion, fixed before the first run: the runoff case runs into the pond and into
//! the reach cell, on the CPU client and on the CUDA client, over the same intervals. After
//! every accepted interval, the text of every child state is equal between the two runs. Equal
//! text means equal bits, signed zero included. No difference is allowed, in any float. The
//! text holds:
//! - the time and the accepted interval count
//! - the full checkpoint package: surface, reach, pool, and runoff packages, counters, and the
//!   last interval with its runoff transfers
//! - the coordinator ledger and the runoff lane ledgers
//! - the live pool, surface, and reach states
//!
//! Each run must also deliver positive runoff in every interval, and a package taken after the
//! first interval must resume on the same client and repeat the remaining intervals bit for bit.

use super::{Result, runtime};
use crate::pools_fixture::REACH;
use crate::runoff_fixture::{Receiver, forcing, parts};

/// Requested duration of every interval, in s.
const DT: f64 = 0.25;
/// Number of accepted intervals of one run.
const INTERVALS: usize = 3;
/// Rain on the subcatchment, in m/s.
const RAIN: f64 = 2e-5;

/// Runs the runoff case into `$receiver` on one client and returns the state text after every
/// accepted interval.
macro_rules! runoff_execution {
    ($runtime:ty, $client:expr, $receiver:expr) => {{
        use hydr::coupling::SurfaceChannelBatch;

        let client = $client;
        let (setup, initial) = parts($receiver)?;
        let mut batch = SurfaceChannelBatch::<$runtime>::new(client.clone(), setup, initial)?;
        let mut states = Vec::new();
        let mut package = None;
        for _ in 0..INTERVALS {
            let end = batch.time() + DT;
            batch = batch
                .trial_to(end, &forcing(batch.time(), end, RAIN))?
                .into_next();
            let step = batch.last_step()?.ok_or("an accepted interval")?;
            assert!(
                step.runoff.iter().all(|transfer| transfer.volume_m3 > 0.0),
                "{:?}",
                step.runoff
            );
            let pools = batch.pools().map(|pools| pools.states()).transpose()?;
            let reach = batch.reach(REACH).ok_or("missing reach")?.state()?;
            states.push(format!(
                "{} {} {:?} {:?} {:?} {:?} {:?} {:?}",
                batch.time(),
                batch.accepted_steps(),
                batch.snapshot_data()?,
                batch.ledger()?,
                batch.runoff_ledgers()?,
                pools,
                batch.surface().state()?,
                reach,
            ));
            if package.is_none() {
                package = Some(batch.snapshot_data()?);
            }
        }
        let data = package.ok_or("the first-interval package")?;
        let mut resumed = SurfaceChannelBatch::<$runtime>::resume(
            client,
            &data,
            data.surface.setup.forcing.clone(),
        )?;
        for _ in 1..INTERVALS {
            let end = resumed.time() + DT;
            resumed = resumed
                .trial_to(end, &forcing(resumed.time(), end, RAIN))?
                .into_next();
        }
        assert_eq!(resumed.snapshot_data()?, batch.snapshot_data()?);
        states
    }};
}

#[test]
fn selected_cpu_client_executes_surface_channel_runoff_into_pools_and_reaches() -> Result {
    for receiver in [Receiver::Pool, Receiver::Reach] {
        let states = runoff_execution!(runtime::Runtime, runtime::client(), receiver);
        assert_eq!(states.len(), INTERVALS);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_surface_channel_runoff_into_pools_and_reaches() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    for receiver in [Receiver::Pool, Receiver::Reach] {
        let cpu = runoff_execution!(runtime::Runtime, runtime::client(), receiver);
        let cuda = runoff_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?, receiver);
        assert_eq!(cpu.len(), cuda.len(), "accepted interval count");
        for (interval, (cpu, cuda)) in cpu.iter().zip(&cuda).enumerate() {
            super::same_bits(cpu, cuda)
                .map_err(|error| format!("{receiver:?} interval {}: {error}", interval + 1))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_surface_channel_runoff_f64_without_fallback() -> Result {
    use hydr::coupling::{SurfaceChannelBatch, SurfaceChannelError};
    use numr::runtime::wgpu::WgpuRuntime;
    let (setup, initial) = parts(Receiver::Pool)?;
    let error = super::f64_rejection(
        SurfaceChannelBatch::<WgpuRuntime>::new(super::wgpu_client()?, setup, initial),
        "surface/channel runoff",
    )?;
    assert!(
        matches!(
            error,
            SurfaceChannelError::Surface(_)
                | SurfaceChannelError::Input(_)
                | SurfaceChannelError::Runoff(_)
        ),
        "Unexpected WGPU surface/channel runoff rejection: {error}"
    );
    Ok(())
}
