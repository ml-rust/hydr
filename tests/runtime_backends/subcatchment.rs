use super::{Result, runtime};
use hydr::runoff::{
    SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentGeometry, SubcatchmentInput,
    SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};
use hydr::soil::ModifiedGreenAmptParameters;

/// Lanes cover one subcatchment and a partial tail beyond 32 lanes.
const WIDTHS: [usize; 2] = [1, 33];

/// Half-impervious area routing half its impervious runoff over Green-Ampt soil.
fn parameters() -> Result<SubcatchmentParameters> {
    let foot = 0.3048;
    Ok(SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: 1024.0 * foot * foot,
            width_m: 32.0 * foot,
            slope: 1.0 / 64.0,
            impervious_fraction: 0.5,
            zero_storage_impervious_fraction: 0.5,
            impervious_manning: 1.0 / 32.0,
            pervious_manning: 1.0 / 32.0,
            impervious_depression_m: 0.125 * foot,
            pervious_depression_m: 0.125 * foot,
        },
        Some(ModifiedGreenAmptParameters::from_native(
            0.125,
            1.0 / 1024.0,
            1.0,
        )?),
        SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
    )?)
}

fn definitions(width: usize) -> Result<Vec<SubcatchmentDefinition>> {
    let parameters = parameters()?;
    Ok((0..width)
        .map(|index| SubcatchmentDefinition {
            id: format!("catchment-{index}"),
            parameters: parameters.clone(),
        })
        .collect())
}

fn states(width: usize) -> Result<Vec<SubcatchmentState>> {
    let mut state = SubcatchmentState::initialized(&parameters()?)?;
    state.depths_m = [0.125 * 0.3048, 0.25 * 0.3048, 0.0];
    Ok(vec![state; width])
}

fn inputs(width: usize, rain_rate_m_per_s: f64) -> Vec<SubcatchmentInput> {
    vec![
        SubcatchmentInput {
            rain_rate_m_per_s,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0; 3],
        };
        width
    ]
}

macro_rules! subcatchment_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::runoff::SubcatchmentBatch;

        let client = $client;
        let rain = 0.3048 / 1024.0;
        let mut finals = Vec::new();
        for width in WIDTHS {
            let build = || -> Result<SubcatchmentBatch<$runtime>> {
                Ok(SubcatchmentBatch::<$runtime>::new(
                    client.clone(),
                    definitions(width)?,
                    states(width)?,
                    0.0,
                    SubcatchmentConfig::default(),
                )?)
            };
            let batch = build()?;
            let original = batch.snapshot_data()?;
            let first = batch.trial_to(64.0, &inputs(width, rain))?;
            let repeat = batch.trial_to(64.0, &inputs(width, rain))?;
            assert_eq!(first.outputs()?, repeat.outputs()?);
            assert_eq!(first.diagnostics()?, repeat.diagnostics()?);
            assert_eq!(batch.snapshot_data()?, original);
            let mut next = first.into_next();
            assert_eq!(next.accepted_steps(), 1);
            let trusted = next.snapshot();
            let owned = trusted.data()?;
            next = next.trial_to(128.0, &inputs(width, 0.0))?.into_next();
            let expected = next.snapshot_data()?;
            next.restore(&trusted)?;
            assert_eq!(next.snapshot_data()?, owned);
            next = next.trial_to(128.0, &inputs(width, 0.0))?.into_next();
            assert_eq!(next.snapshot_data()?, expected);
            let mut cold = build()?;
            cold.restore_data(&owned)?;
            assert_eq!(cold.snapshot_data()?, owned);
            cold = cold.trial_to(128.0, &inputs(width, 0.0))?.into_next();
            assert_eq!(cold.snapshot_data()?, expected);
            finals.push((cold.states()?, expected));
        }
        finals
    }};
}

#[test]
fn selected_cpu_client_executes_subcatchment_routing_and_complete_restart() -> Result {
    subcatchment_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_subcatchment_routing_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = subcatchment_execution!(runtime::Runtime, runtime::client());
    let cuda = subcatchment_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    let volume = SubcatchmentConfig::default().reservoir;
    super::same_states_bounded_data(&cpu, &cuda, volume.volume_absolute, volume.volume_relative)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_subcatchment_f64_without_fallback() -> Result {
    use hydr::runoff::{SubcatchmentBatch, SubcatchmentError};
    use numr::runtime::wgpu::WgpuRuntime;
    let error = super::f64_rejection(
        SubcatchmentBatch::<WgpuRuntime>::new(
            super::wgpu_client()?,
            definitions(1)?,
            states(1)?,
            0.0,
            SubcatchmentConfig::default(),
        ),
        "subcatchment",
    )?;
    assert!(
        matches!(error, SubcatchmentError::InvalidInput { .. }),
        "Unexpected WGPU subcatchment rejection: {error}"
    );
    Ok(())
}
