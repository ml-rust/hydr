//! Surveyed pool attachments of `surveyed_endpoints.rs` on three bands: the surface pool end and
//! the channel/pool connection.

use super::{Result, Run, pools, runtime, same_text, surface};
use hydr::channel::{
    ChannelBoundary, ChannelEnd, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
};
use hydr::coupling::{
    ChannelPoolEndpoint, SurfaceChannelForcingInterval, SurfaceChannelInitialState,
    SurfaceChannelPoolEnd, SurfaceChannelReachForcing, SurfaceChannelSetup,
};
use pools::surveyed_reference::{SurveyedCase, THREE_BANDS_CASES};
use surface::pools_fixture::{Case, DATUM, REACH, pond, pool_end};

/// Intervals before the checkpoint, and after it. Each interval is slow on three bands.
const HALF: usize = 1;

/// Seconds of one interval.
const BAND_DT: f64 = 0.125;

/// Rest depth of the three-band reach, in m.
const BAND_REST: f64 = 1.5;

/// The frozen three-band row of `name` on a pool of 1 m².
fn bands_case(name: &str) -> Result<SurveyedCase> {
    THREE_BANDS_CASES
        .iter()
        .find(|case| case.name == name && case.pool_area == 1.0)
        .copied()
        .ok_or_else(|| format!("missing frozen row {name}").into())
}

/// The coordinator of a three-band cell whose right end opens into a pool.
fn pool_end_parts(
    case: &SurveyedCase,
) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let mut base = Case::tanks(0.0, 0.5);
    base.ponds = vec![(
        pond("pool", 0.0, case.pool_area, 4.0)?,
        case.pool_area * case.elevation,
    )];
    base.pool_ends = vec![SurfaceChannelPoolEnd {
        entrance_loss_coefficient: case.loss,
        ..pool_end("end", ChannelEnd::Right, "pool")
    }];
    let (mut setup, mut initial) = base.parts()?;
    setup.config.head_absolute_m = 1e-13;
    setup.config.head_relative = 0.0;
    setup.config.transfer_absolute_m3 = 1e-12;
    setup.config.transfer_relative = 0.0;
    setup.surface.config.gravity = case.gravity;
    let mut reach: ChannelReachDefinition =
        pools::surveyed_fixture::definition(case.profile, case.gravity, 1, case.cell_length_m);
    reach.vertical_datum = DATUM.into();
    if let ChannelGeometry::PrismaticSurveyed { profile, .. } = &mut reach.geometry {
        profile.vertical_datum = DATUM.into();
    }
    setup.reaches[0].definition = reach;
    initial.reaches[0].1 = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![case.cell_length_m * case.interior_area_m2],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    Ok((setup, initial))
}

/// Forcing toward `end_s`: the pool owns the right end. The left end holds water at rest.
fn pool_end_forcing(start_s: f64, end_s: f64) -> SurfaceChannelForcingInterval {
    SurfaceChannelForcingInterval {
        start_s,
        end_s,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Exterior {
                depth_m: BAND_REST,
                velocity_m_per_s: 0.0,
            }),
            right: None,
            lateral_inputs: Vec::new(),
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
    }
}

/// Surface pool end on three bands, paired exactly every interval, then restarted.
macro_rules! pool_end_run {
    ($runtime:ty, $client:expr, $case:expr) => {{
        use hydr::coupling::SurfaceChannelBatch;

        let client = $client;
        let case: SurveyedCase = $case;
        let (setup, initial) = pool_end_parts(&case)?;
        let build =
            || SurfaceChannelBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone());
        let advance = |mut batch: SurfaceChannelBatch<$runtime>,
                       count: usize,
                       run: &mut Run|
         -> Result<SurfaceChannelBatch<$runtime>> {
            for _ in 0..count {
                let target = batch.time() + BAND_DT;
                while batch.time() < target {
                    let held = pool_end_forcing(batch.time(), target);
                    batch = batch.trial_to(target, &held)?.into_next();
                }
                let step = batch.last_step()?.ok_or("missing interval receipt")?;
                let child = batch.reach(REACH).ok_or("missing reach")?;
                let faces = child
                    .last_step()?
                    .ok_or("missing reach receipt")?
                    .face_volume_m3;
                let face = faces.last().ok_or("missing end face")?;
                assert_eq!(
                    step.pool_ends[0].signed_volume_m3.to_bits(),
                    face.to_bits(),
                    "pool end pairing"
                );
                let cells = child.state()?;
                run.cells("volume", cells.iter().map(|c| c.volume_m3));
                run.cells(
                    "momentum",
                    cells.iter().map(|c| c.longitudinal_momentum_m4_per_s),
                );
                let pools = batch.pools().ok_or("missing pools")?.states()?;
                run.cells("pool volume", pools.iter().map(|p| p.volume_m3));
                let ledger = batch.ledger()?;
                run.ledger(ledger.storage_m3, ledger.residual_m3, ledger.bound_m3);
            }
            Ok(batch)
        };
        let mut run = Run::default();
        let middle = advance(build()?, HALF, &mut run)?;
        let trusted = middle.snapshot();
        let data = middle.snapshot_data()?;
        let whole = advance(middle, HALF, &mut run)?;
        let expected = whole.snapshot_data()?;
        let ends = &whole.pool_end_ledgers()?[0];
        assert!(ends.reach_to_pool_m3 > 0.0 || ends.pool_to_reach_m3 > 0.0);
        let restored = build()?.restore(&trusted)?;
        let imported = build()?.import_snapshot(&data)?;
        for (name, batch) in [("restored", restored), ("imported", imported)] {
            same_text(&batch.snapshot_data()?, &data, name);
            let continued = advance(batch, HALF, &mut Run::default())?;
            same_text(&continued.snapshot_data()?, &expected, name);
        }
        run
    }};
}

#[test]
fn selected_cpu_client_executes_surveyed_pool_end_on_three_bands_with_restart() -> Result {
    let run = pool_end_run!(
        runtime::Runtime,
        runtime::client(),
        bands_case("three-bands-leaving")?
    );
    assert_eq!(run.ledgers.len(), 2 * HALF);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_surveyed_pool_end_on_three_bands() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let name = "three-bands-leaving";
    let cpu = pool_end_run!(runtime::Runtime, runtime::client(), bands_case(name)?);
    let cuda = pool_end_run!(
        CudaRuntime,
        CudaClient::new(CudaDevice::new(0))?,
        bands_case(name)?
    );
    super::same(name, &cpu, &cuda)
}

/// Channel/pool attachment on three bands, checked every interval, then restarted.
macro_rules! channel_pool_run {
    ($runtime:ty, $client:expr, $case:expr) => {{
        use hydr::coupling::ChannelPoolBatch;

        let client = $client;
        let case: SurveyedCase = $case;
        let (setup, initial) = pools::surveyed_fixture::single(&case, ChannelPoolEndpoint::Right)?;
        let build =
            || ChannelPoolBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone());
        let advance = |mut batch: ChannelPoolBatch<$runtime>,
                       count: usize,
                       run: &mut Run|
         -> Result<ChannelPoolBatch<$runtime>> {
            for _ in 0..count {
                let target = batch.time() + BAND_DT;
                while batch.time() < target {
                    let held =
                        pools::surveyed_fixture::held(&setup, BAND_REST, batch.time(), target);
                    batch = batch.trial_to(target, &held)?.into_next();
                }
                let cells = batch.reach("reach").ok_or("missing reach")?.state()?;
                run.cells("volume", cells.iter().map(|c| c.volume_m3));
                run.cells(
                    "momentum",
                    cells.iter().map(|c| c.longitudinal_momentum_m4_per_s),
                );
                let pools = batch.pools().states()?;
                run.cells("pool volume", pools.iter().map(|p| p.volume_m3));
                let ledger = batch.ledger()?;
                let scale = ledger.initial_storage_m3.max(ledger.storage_m3).max(1.0);
                let config = &batch.setup().config;
                let bound = config.mass_absolute_m3 + config.mass_relative * scale;
                assert!(ledger.residual_m3.abs() + ledger.roundoff_bound_m3 <= bound);
                run.ledger(ledger.storage_m3, ledger.residual_m3, bound);
            }
            Ok(batch)
        };
        let mut run = Run::default();
        let middle = advance(build()?, HALF, &mut run)?;
        let trusted = middle.snapshot();
        let data = middle.snapshot_data()?;
        let whole = advance(middle, HALF, &mut run)?;
        let expected = whole.snapshot_data()?;
        let connection = &whole.connection_ledgers()?[0];
        assert!(
            connection.channel_to_pool_volume_m3 > 0.0
                || connection.pool_to_channel_volume_m3 > 0.0
        );
        let restored = build()?.restore(&trusted)?;
        let imported = build()?.import_snapshot(&data)?;
        for (name, batch) in [("restored", restored), ("imported", imported)] {
            same_text(&batch.snapshot_data()?, &data, name);
            let continued = advance(batch, HALF, &mut Run::default())?;
            same_text(&continued.snapshot_data()?, &expected, name);
        }
        run
    }};
}

#[test]
fn selected_cpu_client_executes_surveyed_channel_pool_on_three_bands_with_restart() -> Result {
    let run = channel_pool_run!(
        runtime::Runtime,
        runtime::client(),
        bands_case("three-bands-entering")?
    );
    assert_eq!(run.ledgers.len(), 2 * HALF);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_surveyed_channel_pool_on_three_bands() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let name = "three-bands-entering";
    let cpu = channel_pool_run!(runtime::Runtime, runtime::client(), bands_case(name)?);
    let cuda = channel_pool_run!(
        CudaRuntime,
        CudaClient::new(CudaDevice::new(0))?,
        bands_case(name)?
    );
    super::same(name, &cpu, &cuda)
}
