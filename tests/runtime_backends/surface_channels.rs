//! Surface/channel flood, recession, and restart on configured runtime clients.

#[allow(dead_code)]
#[path = "../surface_channels/fixture.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../surface_channels/scenario.rs"]
mod scenario;

use super::{Result, runtime};
use hydr::channel::ChannelBoundary;
use hydr::coupling::{
    SurfaceChannelForcingInterval, SurfaceChannelInitialState, SurfaceChannelReachForcing,
    SurfaceChannelSetup,
};
use hydr::solver::{ForcingInterval, ForcingSchedule};
use scenario::{ReachSpec, Spec, weir};

/// End of the flood, in seconds. Surface evaporation drains the surface afterwards.
const FLOOD_END: f64 = 0.5;

/// Potential evaporation during the recession, in m/s.
const RECESSION_EVAPORATION: f64 = 2.0;

/// Requested common endpoints, in seconds. The checkpoint falls at `FLOOD_END`.
const TARGETS: [f64; 4] = [0.25, FLOOD_END, 0.75, 1.0];

/// Surface stage 2 m spills over a 1.2 m crest into a reach at stage 1.3 m.
///
/// The flood raises the reach further above the crest. The recession then lowers the surface
/// below the reach, so the reach spills back through the same link.
fn case() -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
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
        vec![
            interval(0.0, FLOOD_END, 0.0),
            interval(FLOOD_END, 100.0, RECESSION_EVAPORATION),
        ],
    )?;
    Ok((setup, initial))
}

/// Walls at both reach ends and no caller lateral inputs from `start` to `end`.
fn forcing(setup: &SurfaceChannelSetup, start: f64, end: f64) -> SurfaceChannelForcingInterval {
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: setup
            .reaches
            .iter()
            .map(|reach| SurfaceChannelReachForcing {
                reach_id: reach.id.clone(),
                left: ChannelBoundary::Wall,
                right: ChannelBoundary::Wall,
                lateral_inputs: Vec::new(),
            })
            .collect(),
    }
}

/// Runs the flood and recession on one client and checks restart against the whole run.
///
/// Returns the final per-cell states and the final checkpoint package for comparison.
macro_rules! surface_channel_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::coupling::SurfaceChannelBatch;

        let client = $client;
        let (setup, initial) = case()?;
        let build =
            || SurfaceChannelBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone());
        // Accepts intervals toward each target in turn and records each link direction.
        let advance = |mut batch: SurfaceChannelBatch<$runtime>,
                       targets: &[f64]|
         -> Result<(SurfaceChannelBatch<$runtime>, [bool; 2])> {
            let mut directions = [false; 2];
            for &target in targets {
                while batch.time() < target {
                    let held = forcing(batch.setup(), batch.time(), target);
                    batch = batch.trial_to(target, &held)?.into_next();
                    let step = batch.last_step()?.ok_or("missing interval receipt")?;
                    let signed = step.links[0].signed_volume_m3;
                    let flood = step.end_s <= FLOOD_END;
                    directions[0] |= flood && signed > 0.0;
                    directions[1] |= !flood && signed < 0.0;
                }
            }
            Ok((batch, directions))
        };
        let origin = build()?;
        let original = origin.snapshot_data()?;
        let first = origin.trial_to(TARGETS[0], &forcing(&setup, 0.0, TARGETS[0]))?;
        assert!(first.time() > 0.0);
        drop(first);
        assert_eq!(
            origin.snapshot_data()?,
            original,
            "a trial changed the batch"
        );
        let split = TARGETS
            .iter()
            .position(|&t| t == FLOOD_END)
            .ok_or("no flood target")?
            + 1;
        let (flooded, [spilled, _]) = advance(origin, &TARGETS[..split])?;
        assert!(spilled, "the surface must spill into the reach");
        let trusted = flooded.snapshot();
        let data = flooded.snapshot_data()?;
        let (whole, [_, reversed]) = advance(flooded, &TARGETS[split..])?;
        assert!(reversed, "the reach must spill back onto the surface");
        let expected = whole.snapshot_data()?;
        let ledger = whole.ledger()?;
        assert!(ledger.residual_m3.abs() <= ledger.bound_m3, "{ledger:?}");
        let restored = build()?.restore(&trusted)?;
        let imported = build()?.import_snapshot(&data)?;
        let resumed = SurfaceChannelBatch::<$runtime>::resume(
            client.clone(),
            &data,
            data.surface.setup.forcing.clone(),
        )?;
        for (name, batch) in [
            ("restored", restored),
            ("imported", imported),
            ("resumed", resumed),
        ] {
            assert_eq!(batch.snapshot_data()?, data, "{name} at the checkpoint");
            let (continued, _) = advance(batch, &TARGETS[split..])?;
            assert_eq!(
                continued.snapshot_data()?,
                expected,
                "{name} after continuing"
            );
        }
        let surface = whole.surface().state()?;
        let reach = whole.reach("river").ok_or("missing reach")?.state()?;
        vec![((surface, reach), expected)]
    }};
}

#[test]
fn selected_cpu_client_executes_surface_channel_overflow_reversal_and_restart() -> Result {
    surface_channel_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_surface_channel_overflow_reversal_and_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = surface_channel_execution!(runtime::Runtime, runtime::client());
    let cuda = surface_channel_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    // Accepted cell states match bit for bit. Lifetime counters, ledgers, bounds and iteration
    // residuals come from device reductions and differ in their last bits.
    let config = case()?.0.config;
    super::same_states_bounded_data(&cpu, &cuda, config.mass_absolute_m3, config.mass_relative)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_surface_channel_f64_without_fallback() -> Result {
    use hydr::coupling::{SurfaceChannelBatch, SurfaceChannelError};
    use numr::runtime::wgpu::WgpuRuntime;
    let (setup, initial) = case()?;
    let error = super::f64_rejection(
        SurfaceChannelBatch::<WgpuRuntime>::new(super::wgpu_client()?, setup, initial),
        "surface/channel",
    )?;
    assert!(
        matches!(error, SurfaceChannelError::Surface(_)),
        "Unexpected WGPU surface/channel rejection: {error}"
    );
    Ok(())
}
