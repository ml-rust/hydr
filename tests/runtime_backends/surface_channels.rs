//! Surface/channel flood, recession, and restart on configured runtime clients.

#[allow(dead_code)]
#[path = "../surface_channels/fixture.rs"]
pub(crate) mod fixture;
#[allow(dead_code)]
#[path = "../surface_channels/ports_fixture.rs"]
pub(crate) mod ports_fixture;
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
use ports_fixture::{ALPHA, BETA, Reach, Scenario, west};
use scenario::{ReachSpec, Spec, weir};

/// Requested duration of every port interval, in s. The CFL limit shortens it.
const PORT_DT: f64 = 0.25;

/// Port intervals before the checkpoint, and after it. The reach refills after about 38
/// intervals and drains again after about 68.
const PORT_HALF: usize = 40;

/// Start and end of the storm rain in the CPU/CUDA parity port run, in s.
const STORM: [f64; 2] = [0.3, 0.8];

/// Storm rain rate, in m/s.
const STORM_RAIN: f64 = 1.0;

/// Potential evaporation after the storm, in m/s.
const STORM_EVAPORATION: f64 = 2.0;

/// Storm port intervals before the checkpoint. The checkpoint falls at the start of the rain.
#[cfg(feature = "cuda")]
const STORM_BEFORE: usize = 2;

/// Storm port intervals after the checkpoint. The rain refills the reach from the third
/// interval, and evaporation drains it again from the eighth.
#[cfg(feature = "cuda")]
const STORM_AFTER: usize = 6;

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
                left: Some(ChannelBoundary::Wall),
                right: Some(ChannelBoundary::Wall),
                lateral_inputs: Vec::new(),
            })
            .collect(),
        pool_sources: None,
        pool_inflows: Vec::new(),
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

/// A reach above its floodplain drains through the west port and the bank link together, then
/// refills and drains again.
///
/// The weir law evaluates correctly rounded `exp` and `log`, so the bank link keeps accepted
/// cell states bit for bit equal on every runtime.
fn port_case() -> Scenario {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
    case.alpha = Reach::rest(0.5, 1.75);
    case
}

/// Port case parts. With `storm`, rain lifts the surface above the reach, then evaporation
/// lowers it again.
fn port_parts(storm: bool) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (mut setup, initial) = port_case().parts()?;
    if storm {
        let surface = &mut setup.surface;
        let cells = ports_fixture::CELLS;
        let interval = |start, end, rain, evaporation| ForcingInterval {
            start,
            end,
            rain: vec![rain; cells],
            potential_evaporation: vec![evaporation; cells],
            withdrawal: vec![0.0; cells],
            boundaries: Vec::new(),
        };
        surface.forcing = ForcingSchedule::new(
            &surface.domain,
            &surface.boundaries,
            vec![
                interval(0.0, STORM[0], 0.0, 0.0),
                interval(STORM[0], STORM[1], STORM_RAIN, 0.0),
                interval(STORM[1], 1000.0, 0.0, STORM_EVAPORATION),
            ],
        )?;
    }
    Ok((setup, initial))
}

/// Runs the port case on one client and checks restart against the whole run.
///
/// Returns the final per-cell states and the final checkpoint package for comparison.
macro_rules! surface_channel_port_execution {
    ($runtime:ty, $client:expr, $storm:expr, $before:expr, $after:expr) => {{
        use hydr::coupling::SurfaceChannelBatch;

        let client = $client;
        let (before, after): (usize, usize) = ($before, $after);
        let (setup, initial) = port_parts($storm)?;
        let build =
            || SurfaceChannelBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone());
        // Accepts `count` intervals. Records port flow into and out of the reach.
        let advance = |mut batch: SurfaceChannelBatch<$runtime>,
                       count: usize|
         -> Result<(SurfaceChannelBatch<$runtime>, [bool; 2])> {
            let mut flows = [false; 2];
            for _ in 0..count {
                let (start, end) = (batch.time(), batch.time() + PORT_DT);
                let held = ports_fixture::forcing(batch.setup(), start, end);
                batch = batch.trial_to(end, &held)?.into_next();
                let step = batch.last_step()?.ok_or("missing interval receipt")?;
                let port = step.ports.first().ok_or("missing port receipt")?;
                assert!(port.paired_gap_m3.abs() <= port.paired_bound_m3, "{port:?}");
                assert!(
                    step.mass_residual_m3.abs() <= step.mass_bound_m3,
                    "{step:?}"
                );
                flows[0] |= port.signed_volume_m3 > 0.0;
                flows[1] |= port.signed_volume_m3 < 0.0;
            }
            Ok((batch, flows))
        };
        let (middle, [_, drained]) = advance(build()?, before)?;
        assert!(
            drained,
            "the reach must drain onto the surface through the port"
        );
        let trusted = middle.snapshot();
        let data = middle.snapshot_data()?;
        let (whole, [refilled, redrained]) = advance(middle, after)?;
        assert!(
            refilled,
            "the surface must refill the reach through the port"
        );
        assert!(
            redrained,
            "the port must reverse again after the checkpoint"
        );
        let expected = whole.snapshot_data()?;
        let ledger = whole.ledger()?;
        assert!(ledger.residual_m3.abs() <= ledger.bound_m3, "{ledger:?}");
        let port_ledger = &whole.port_ledgers()?[0];
        assert!(port_ledger.surface_to_channel_m3 > 0.0 && port_ledger.channel_to_surface_m3 > 0.0);
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
            let (continued, _) = advance(batch, after)?;
            assert_eq!(
                continued.snapshot_data()?,
                expected,
                "{name} after continuing"
            );
        }
        let surface = whole.surface().state()?;
        let alpha = whole.reach(ALPHA).ok_or("missing reach")?.state()?;
        let beta = whole.reach(BETA).ok_or("missing reach")?.state()?;
        vec![((surface, alpha, beta), expected)]
    }};
}

#[test]
fn selected_cpu_client_executes_surface_channel_port_reversal_and_restart() -> Result {
    surface_channel_port_execution!(
        runtime::Runtime,
        runtime::client(),
        false,
        PORT_HALF,
        PORT_HALF
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_surface_channel_port_reversal_and_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = surface_channel_port_execution!(
        runtime::Runtime,
        runtime::client(),
        true,
        STORM_BEFORE,
        STORM_AFTER
    );
    let cuda = surface_channel_port_execution!(
        CudaRuntime,
        CudaClient::new(CudaDevice::new(0))?,
        true,
        STORM_BEFORE,
        STORM_AFTER
    );
    // Accepted cell states match bit for bit. Lifetime counters, ledgers, bounds and iteration
    // residuals come from device reductions and differ in their last bits.
    let config = port_parts(true)?.0.config;
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
