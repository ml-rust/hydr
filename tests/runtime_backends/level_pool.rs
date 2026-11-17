use super::{Result, runtime};
use hydr::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolInput, LevelPoolKnot,
    LevelPoolState,
};

/// Lanes cover one pool and a partial tail beyond 32 lanes.
const WIDTHS: [usize; 2] = [1, 33];

fn elements(width: usize) -> Result<Vec<LevelPoolElement>> {
    (0..width)
        .map(|index| {
            Ok(LevelPoolElement {
                id: format!("pool-{index}"),
                curve: LevelPoolCurve::new(
                    vec![
                        LevelPoolKnot {
                            volume_m3: 0.0,
                            stage_m: 10.0,
                        },
                        LevelPoolKnot {
                            volume_m3: 4.0,
                            stage_m: 12.0,
                        },
                    ],
                    "synthetic vertical datum".into(),
                    "Explicit synthetic connected-storage table".into(),
                )?,
                connected_pool_assertion: "Explicit synthetic connected approximately level pool"
                    .into(),
                storage_support_identity: format!("storage-support-{index}"),
            })
        })
        .collect()
}

fn inputs(width: usize, incoming_volume_m3: f64, outgoing_volume_m3: f64) -> Vec<LevelPoolInput> {
    vec![
        LevelPoolInput {
            incoming_volume_m3,
            outgoing_volume_m3,
        };
        width
    ]
}

macro_rules! level_pool_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::storage::LevelPoolBatch;

        let client = $client;
        let mut finals = Vec::new();
        for width in WIDTHS {
            let build = || -> Result<LevelPoolBatch<$runtime>> {
                Ok(LevelPoolBatch::<$runtime>::new(
                    client.clone(),
                    elements(width)?,
                    vec![LevelPoolState { volume_m3: 1.0 }; width],
                    0.0,
                    LevelPoolConfig::default(),
                )?)
            };
            let batch = build()?;
            let original = batch.snapshot_data()?;
            let trial = batch.trial_to(1.0, &inputs(width, 0.2, 0.1))?;
            let receipts = trial.diagnostics()?.receipts;
            assert_eq!(receipts.len(), width);
            for receipt in &receipts {
                assert_eq!(receipt.remaining_volume_m3.to_bits(), 0.9_f64.to_bits());
                assert_eq!(receipt.volume_m3.to_bits(), 1.1_f64.to_bits());
                assert_eq!(
                    receipt.outgoing_witness_m3.to_bits(),
                    (-2.0_f64.powi(-55)).to_bits()
                );
                assert_eq!(
                    receipt.incoming_witness_m3.to_bits(),
                    (-2.0_f64.powi(-54)).to_bits()
                );
            }
            assert_eq!(batch.snapshot_data()?, original);
            let mut next = trial.into_next();
            let trusted = next.snapshot();
            let owned = trusted.data()?;
            next = next.trial_to(2.0, &inputs(width, 0.0, 0.5))?.into_next();
            let expected = next.snapshot_data()?;
            for state in next.states()? {
                assert!((state.volume_m3 - 0.6).abs() <= 1e-15);
            }
            next.restore(&trusted)?;
            assert_eq!(next.snapshot_data()?, owned);
            next = next.trial_to(2.0, &inputs(width, 0.0, 0.5))?.into_next();
            assert_eq!(next.snapshot_data()?, expected);
            let mut cold = build()?;
            cold.restore_data(&owned)?;
            assert_eq!(cold.snapshot_data()?, owned);
            cold = cold.trial_to(2.0, &inputs(width, 0.0, 0.5))?.into_next();
            assert_eq!(cold.snapshot_data()?, expected);
            finals.push((cold.states()?, expected));
        }
        finals
    }};
}

#[test]
fn selected_cpu_client_executes_level_pool_exchange_and_complete_restart() -> Result {
    level_pool_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_level_pool_exchange_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = level_pool_execution!(runtime::Runtime, runtime::client());
    let cuda = level_pool_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    let config = LevelPoolConfig::default();
    super::same_states_bounded_data(&cpu, &cuda, config.volume_absolute, config.volume_relative)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_level_pool_f64_without_fallback() -> Result {
    use hydr::storage::{LevelPoolBatch, LevelPoolError};
    use numr::runtime::wgpu::WgpuRuntime;
    let error = super::f64_rejection(
        LevelPoolBatch::<WgpuRuntime>::new(
            super::wgpu_client()?,
            elements(1)?,
            vec![LevelPoolState { volume_m3: 1.0 }],
            0.0,
            LevelPoolConfig::default(),
        ),
        "level pool",
    )?;
    assert!(
        matches!(error, LevelPoolError::InvalidInput { .. }),
        "Unexpected WGPU level pool rejection: {error}"
    );
    Ok(())
}
