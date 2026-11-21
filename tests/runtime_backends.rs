//! Actual runtime execution and arithmetic admission through public components.

#[path = "runtime_backends/channel_boundaries.rs"]
mod channel_boundaries;
#[path = "runtime_backends/channel_pools.rs"]
mod channel_pools;
#[path = "runtime_backends/channel_reservoir.rs"]
mod channel_reservoir;
#[path = "runtime_backends/channel_tensor_forcing.rs"]
mod channel_tensor_forcing;
#[path = "runtime_backends/channels.rs"]
mod channels;
#[path = "runtime_backends/control_fixture.rs"]
mod control_fixture;
#[path = "runtime_backends/controlled_gates.rs"]
mod controlled_gates;
#[path = "runtime_backends/controlled_pumps.rs"]
mod controlled_pumps;
#[path = "runtime_backends/coupled_level_pool_culverts.rs"]
mod coupled_level_pool_culverts;
#[path = "coupled_level_pools/fixture.rs"]
mod coupled_pool_fixture;
#[allow(dead_code)]
#[path = "culvert_barrels/critical_rows.rs"]
mod critical_rows;
#[path = "cross_sections/profiles.rs"]
mod cross_section_profiles;
#[path = "runtime_backends/cross_sections.rs"]
mod cross_sections;
#[path = "runtime_backends/culvert_barrels.rs"]
mod culvert_barrels;
#[allow(dead_code)]
#[path = "culvert_inlets/lanes.rs"]
mod culvert_inlet_lanes;
#[path = "runtime_backends/culvert_inlets.rs"]
mod culvert_inlets;
#[allow(dead_code)]
#[path = "culvert_barrels/lanes.rs"]
mod culvert_lanes;
#[allow(dead_code)]
#[path = "culvert_laws/lanes.rs"]
mod culvert_law_lanes;
#[path = "runtime_backends/culvert_laws.rs"]
mod culvert_laws;
#[allow(dead_code)]
#[path = "culvert_outlets/lanes.rs"]
mod culvert_outlet_lanes;
#[path = "runtime_backends/culvert_outlets.rs"]
mod culvert_outlets;
#[allow(dead_code)]
#[path = "surface_channels/culverts_fixture.rs"]
mod culverts_fixture;
#[allow(dead_code)]
#[path = "culvert_barrels/dense_rows.rs"]
mod dense_rows;
#[path = "runtime_backends/device_checks.rs"]
mod device_checks;
#[path = "runtime_backends/gate_actuators.rs"]
mod gate_actuators;
#[path = "coupled_pool_inflows/inflow_fixture.rs"]
mod inflow_fixture;
#[path = "runtime_backends/inflows.rs"]
mod inflows;
#[allow(dead_code)]
#[path = "culverts/inlet_rows.rs"]
mod inlet_rows;
#[path = "runtime_backends/interval_volumes.rs"]
mod interval_volumes;
#[path = "runtime_backends/level_pool.rs"]
mod level_pool;
#[allow(dead_code)]
#[path = "culvert_barrels/normal_rows.rs"]
mod normal_rows;
#[path = "runtime_backends/operating_controls.rs"]
mod operating_controls;
#[allow(dead_code)]
#[path = "culverts/outlet_rows.rs"]
mod outlet_rows;
#[allow(dead_code, unused_imports)]
#[path = "surface_channels/pools_fixture.rs"]
mod pools_fixture;
#[allow(dead_code)]
#[path = "culvert_barrels/reference_rows.rs"]
mod reference_rows;
#[path = "runtime_backends/reservoir.rs"]
mod reservoir;
#[path = "runtime_backends/runoff_pools.rs"]
mod runoff_pools;
#[path = "support/runtime.rs"]
mod runtime;
#[path = "runtime_backends/structure_laws.rs"]
mod structure_laws;
#[path = "runtime_backends/subcatchment.rs"]
mod subcatchment;
#[allow(dead_code)]
#[path = "culvert_laws/support.rs"]
mod support;
#[path = "runtime_backends/surface_channel_culverts.rs"]
mod surface_channel_culverts;
#[path = "runtime_backends/surface_channel_pools.rs"]
mod surface_channel_pools;
#[path = "runtime_backends/surface_channels.rs"]
mod surface_channels;
#[path = "runtime_backends/surveyed_endpoints.rs"]
mod surveyed_endpoints;
#[path = "runtime_backends/triangular_channels.rs"]
mod triangular_channels;

use hydr::soil::{
    ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, NativeSoilInput, NativeSoilState,
};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "wgpu")]
use device_checks::{f64_rejection, wgpu_client};
#[cfg(feature = "cuda")]
use device_checks::{same_bits, same_states_bounded_data};

macro_rules! dry_soil_execution {
    ($runtime:ty, $client:expr) => {{
        let parameters = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
        let initial = NativeSoilState {
            imd: 0.2,
            f_ft: 0.01,
            fu_ft: 0.02,
            saturated: true,
            t_s: 5.0,
        };
        let batch = ModifiedGreenAmptBatch::<$runtime>::from_native_states(
            $client,
            &[parameters],
            &[initial],
        )?;
        let trial = batch.trial_native(
            60.0,
            &[NativeSoilInput {
                incoming_rate_ft_per_s: 0.0,
                ponded_depth_ft: 0.0,
            }],
        )?;
        assert_eq!(trial.outputs()?[0].native_rate_ft_per_s, 0.0);
        assert_eq!(
            trial.into_next().native_states()?,
            vec![NativeSoilState {
                t_s: -55.0,
                ..initial
            }]
        );
        assert_eq!(batch.native_states()?, vec![initial]);
    }};
}

#[test]
fn selected_cpu_client_executes_soil_trial_without_mutating_accepted_state() -> Result {
    dry_soil_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_soil_trial_without_mutating_accepted_state() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    dry_soil_execution!(CudaRuntime, client);
    Ok(())
}

/// Still water at depth one on a closed 2 x 2 grid, with the expected cells.
fn still_water_case() -> Result<(
    hydr::solver::SolverSetup,
    hydr::solver::InitialState,
    Vec<hydr::model::CellState>,
)> {
    use hydr::grid::CartesianGrid;
    use hydr::model::{CellState, Domain, FlowState, SpatialReference};
    use hydr::soil::{LandModel, SoilState};
    use hydr::solver::{
        BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig,
        SolverSetup,
    };

    let grid = CartesianGrid::new(2, 2, 1.0, 1.0)?;
    let count = grid.cell_count();
    let reference = SpatialReference::new(
        "synthetic metric plane".into(),
        "synthetic datum".into(),
        [0.0, 0.0],
    )?;
    let domain = Domain::new(
        grid,
        vec![0.0; count],
        vec![0.0; count],
        vec![true; count],
        reference,
    )?;
    let cells = vec![
        CellState {
            h: 1.0,
            qx: 0.0,
            qy: 0.0
        };
        count
    ];
    let flow = FlowState::new(&domain, cells.clone())?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 2.0,
            rain: vec![0.0; count],
            potential_evaporation: vec![0.0; count],
            withdrawal: vec![0.0; count],
            boundaries: vec![],
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let setup = SolverSetup {
        domain,
        land,
        boundaries,
        forcing,
        config: SolverConfig::default(),
    };
    let initial = InitialState {
        time: 0.0,
        flow,
        soil,
    };
    Ok((setup, initial, cells))
}

macro_rules! still_water_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::solver::Solver;

        let (setup, initial, cells) = still_water_case()?;
        let mut solver = Solver::<$runtime>::new($client, setup, initial)?;
        assert_eq!(solver.advance_to(0.5, 1_000)?.accepted_time, 0.5);
        let checkpoint = solver.snapshot();
        let checkpoint_data = checkpoint.data()?;
        assert_eq!(solver.state()?.cells(), cells.as_slice());
        assert_eq!(solver.advance_to(1.0, 1_000)?.accepted_time, 1.0);
        let continued = solver.snapshot().data()?;
        let continued_state = solver.state()?;
        assert_eq!(continued_state.cells(), cells.as_slice());
        assert_eq!(solver.ledger().storage, 4.0);
        assert_eq!(solver.ledger().residual, 0.0);
        solver.restore(&checkpoint)?;
        assert_eq!(solver.snapshot().data()?, checkpoint_data);
        assert_eq!(solver.advance_to(1.0, 1_000)?.accepted_time, 1.0);
        assert_eq!(solver.snapshot().data()?, continued);
        assert_eq!(solver.state()?, continued_state);
        assert_eq!(solver.ledger().storage, 4.0);
        assert_eq!(solver.ledger().residual, 0.0);
    }};
}

#[test]
fn selected_cpu_client_preserves_still_water_and_trusted_restart() -> Result {
    still_water_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_preserves_still_water_and_trusted_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    still_water_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}

macro_rules! coupled_pool_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::coupling::CoupledLevelPoolBatch;

        let (setup, initial) = coupled_pool_fixture::orifice_case([4.0, 2.0])?;
        let batch = CoupledLevelPoolBatch::<$runtime>::new($client, setup, initial)?;
        let original = batch.snapshot_data()?;
        let mut next = batch.trial_to(0.25)?.into_next();
        assert_eq!(batch.snapshot_data()?, original);
        assert_eq!(next.time(), 0.25);
        let checkpoint = next.snapshot();
        let checkpoint_data = checkpoint.data()?;
        next = next.trial_to(0.5)?.into_next();
        let continued = next.snapshot_data()?;
        let volumes = next.pools().states()?;
        coupled_pool_fixture::near(volumes.iter().map(|pool| pool.volume_m3).sum(), 6.0);
        assert!(volumes[0].volume_m3 < 4.0);
        assert!(volumes[1].volume_m3 > 2.0);
        next.restore(&checkpoint)?;
        assert_eq!(next.snapshot_data()?, checkpoint_data);
        next = next.trial_to(0.5)?.into_next();
        assert_eq!(next.snapshot_data()?, continued);
    }};
}

#[test]
fn selected_cpu_client_executes_coupled_pool_exchange_and_trusted_restart() -> Result {
    coupled_pool_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_coupled_pool_exchange_and_trusted_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    coupled_pool_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}

macro_rules! coupled_pool_source_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::coupling::{CoupledLevelPoolBatch, CoupledLevelPoolForcingInterval};
        use hydr::storage::{LevelPoolSourcePolicy, LevelPoolSources};

        let (mut setup, initial) = coupled_pool_fixture::orifice_case([4.0, 2.0])?;
        let policies = vec![
            LevelPoolSourcePolicy {
                dry_basal_footprint_assertion:
                    "Synthetic first segment is the rain-receiving basal footprint".into(),
                bottom_loss_provenance: "Explicit zero bottom demand without inferred soil state"
                    .into(),
            };
            2
        ];
        let forcing = CoupledLevelPoolForcingInterval {
            start_s: 0.0,
            end_s: 1.0,
            sources: Some(vec![
                LevelPoolSources {
                    rain_rate_m_per_s: 0.25,
                    potential_evaporation_rate_m_per_s: 0.0,
                    bottom_loss_rate_m3_per_s: 0.0,
                };
                2
            ]),
            inflows: vec![],
        };
        setup.source_policies = Some(policies);
        let batch = CoupledLevelPoolBatch::<$runtime>::new($client, setup, initial)?;
        let original = batch.snapshot_data()?;
        let mut next = batch.trial_with_forcing_to(0.25, &forcing)?.into_next();
        assert_eq!(batch.snapshot_data()?, original);
        assert_eq!(next.time(), 0.25);
        let dt: f64 = 0.25;
        let transfer = dt * 0.5 * ((dt * dt + 8.0).sqrt() - dt);
        let states = next.pools().states()?;
        coupled_pool_fixture::near(states[0].volume_m3, 4.0 + 0.25 * dt - transfer);
        coupled_pool_fixture::near(states[1].volume_m3, 2.0 + 0.25 * dt + transfer);
        let checkpoint = next.snapshot();
        let checkpoint_data = checkpoint.data()?;
        next = next.trial_with_forcing_to(0.5, &forcing)?.into_next();
        let continued = next.snapshot_data()?;
        let totals = next.pools().totals()?;
        coupled_pool_fixture::near(totals.volume_m3, 6.25);
        coupled_pool_fixture::near(
            totals
                .sources
                .ok_or("Missing runtime source counters")?
                .cumulative_rain_volume_m3,
            0.25,
        );
        next.restore(&checkpoint)?;
        assert_eq!(next.snapshot_data()?, checkpoint_data);
        next = next.trial_with_forcing_to(0.5, &forcing)?.into_next();
        assert_eq!(next.snapshot_data()?, continued);
    }};
}

#[test]
fn selected_cpu_client_executes_coupled_sources_and_trusted_restart() -> Result {
    coupled_pool_source_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_coupled_sources_and_trusted_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    coupled_pool_source_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_client_rejects_f64_profile() -> Result {
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    assert!(!WgpuRuntime::computes_float_dtype(DType::F64));
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    let parameters = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    let error = ModifiedGreenAmptBatch::<WgpuRuntime>::initialized(client, &[parameters])
        .err()
        .ok_or("WGPU admitted unsupported F64 profile")?;
    assert!(error.to_string().contains("F64"));
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_surface_solver_f64_without_fallback() -> Result {
    use hydr::solver::{EngineError, Solver};
    use numr::runtime::wgpu::WgpuRuntime;
    let (setup, initial, _) = still_water_case()?;
    let error = f64_rejection(
        Solver::<WgpuRuntime>::new(wgpu_client()?, setup, initial),
        "surface solver",
    )?;
    let EngineError::Rejected { operation, .. } = error else {
        return Err(format!("Unexpected WGPU surface rejection: {error}").into());
    };
    assert_eq!(operation, "arithmetic environment");
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_coupled_level_pools_f64_without_fallback() -> Result {
    use hydr::coupling::{CoupledLevelPoolBatch, CoupledLevelPoolError};
    use numr::runtime::wgpu::WgpuRuntime;
    let (setup, initial) = coupled_pool_fixture::orifice_case([4.0, 2.0])?;
    let error = f64_rejection(
        CoupledLevelPoolBatch::<WgpuRuntime>::new(wgpu_client()?, setup, initial),
        "coupled level pool",
    )?;
    assert!(
        matches!(
            error,
            CoupledLevelPoolError::Pool(_) | CoupledLevelPoolError::Structure(_)
        ),
        "Unexpected WGPU coupled pool rejection: {error}"
    );
    Ok(())
}
