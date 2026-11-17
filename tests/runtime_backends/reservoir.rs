use super::{Result, runtime};
use hydr::runoff::{
    ReservoirConfig, ReservoirElement, ReservoirInput, ReservoirParameters, ReservoirState,
};

/// Lanes cover one reservoir and a partial tail beyond 32 lanes.
const WIDTHS: [usize; 2] = [1, 33];

fn elements(width: usize) -> Result<Vec<ReservoirElement>> {
    let parameters =
        ReservoirParameters::from_native(1024.0, 1024.0, 32.0, 1.0 / 64.0, 1.0 / 32.0, 0.125)?;
    Ok((0..width)
        .map(|index| ReservoirElement {
            id: format!("storage-{index}"),
            parameters,
        })
        .collect())
}

fn inputs(width: usize, incoming_rate_m_per_s: f64) -> Vec<ReservoirInput> {
    vec![
        ReservoirInput {
            incoming_rate_m_per_s,
        };
        width
    ]
}

macro_rules! reservoir_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::runoff::ReservoirBatch;

        let client = $client;
        let mut finals = Vec::new();
        for width in WIDTHS {
            let build = || -> Result<ReservoirBatch<$runtime>> {
                Ok(ReservoirBatch::<$runtime>::new(
                    client.clone(),
                    elements(width)?,
                    vec![
                        ReservoirState {
                            depth_m: 0.25 * 0.3048
                        };
                        width
                    ],
                    0.0,
                    ReservoirConfig::default(),
                )?)
            };
            let batch = build()?;
            let original = batch.snapshot_data()?;
            let first = batch.trial_to(1.0, &inputs(width, 0.0))?;
            let repeat = batch.trial_to(1.0, &inputs(width, 0.0))?;
            assert_eq!(first.outputs()?, repeat.outputs()?);
            assert_eq!(batch.snapshot_data()?, original);
            let mut next = first.into_next();
            assert!(next.ledger().export_volume_m3 > 0.0);
            for state in next.states()? {
                assert!(state.depth_m < 0.25 * 0.3048);
                assert!(state.depth_m > 0.125 * 0.3048);
            }
            let trusted = next.snapshot();
            let owned = trusted.data()?;
            let rain = 0.3048 / 1024.0;
            next = next.trial_to(2.0, &inputs(width, rain))?.into_next();
            let expected = next.snapshot_data()?;
            next.restore(&trusted)?;
            assert_eq!(next.snapshot_data()?, owned);
            next = next.trial_to(2.0, &inputs(width, rain))?.into_next();
            assert_eq!(next.snapshot_data()?, expected);
            let mut cold = build()?;
            cold.restore_data(&owned)?;
            assert_eq!(cold.snapshot_data()?, owned);
            cold = cold.trial_to(2.0, &inputs(width, rain))?.into_next();
            assert_eq!(cold.snapshot_data()?, expected);
            finals.push((cold.states()?, expected));
        }
        finals
    }};
}

#[test]
fn selected_cpu_client_executes_reservoir_recession_and_complete_restart() -> Result {
    reservoir_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_reservoir_recession_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = reservoir_execution!(runtime::Runtime, runtime::client());
    let cuda = reservoir_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    let config = ReservoirConfig::default();
    super::same_states_bounded_data(&cpu, &cuda, config.volume_absolute, config.volume_relative)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_reservoir_f64_without_fallback() -> Result {
    use hydr::runoff::{ReservoirBatch, ReservoirError};
    use numr::runtime::wgpu::WgpuRuntime;
    let error = super::f64_rejection(
        ReservoirBatch::<WgpuRuntime>::new(
            super::wgpu_client()?,
            elements(1)?,
            vec![ReservoirState { depth_m: 0.0 }],
            0.0,
            ReservoirConfig::default(),
        ),
        "reservoir",
    )?;
    assert!(
        matches!(error, ReservoirError::InvalidInput { .. }),
        "Unexpected WGPU reservoir rejection: {error}"
    );
    Ok(())
}
