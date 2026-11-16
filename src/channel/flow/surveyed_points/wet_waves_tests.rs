use crate::channel::flow::surveyed_points::corpus_coverage::Coverage;
use crate::channel::flow::surveyed_points::{
    chart_test_support,
    source_tests::support as frozen,
    wet_depth::refine_wet_depth,
    wet_depth::tests::support,
    wet_roots::locate_wet_roots,
    wet_wave_types::{WetWaveDomain, WetWaveStatus},
};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

#[path = "wet_waves_tests_controls.rs"]
mod controls;
#[path = "wet_waves_tests_diagnostics.rs"]
mod diagnostics;
#[path = "wet_waves_tests_lifecycle.rs"]
mod lifecycle;
#[path = "wet_waves_tests_frozen.rs"]
mod references;

fn matrix<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    references::run::<R>(client, coverage)?;
    controls::run::<R>(client)?;
    lifecycle::run::<R>(client)?;
    support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[1.0, 1.0, 1.0, 1.0]],
        &chart_test_support::config(),
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(
                pair,
                &rows,
                pair.evaluation.config.root.max_points,
            ))?;
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            let before = work.snapshot();
            let waves = hydraulic(session.certify_wet_waves())?;
            assert_eq!(work.snapshot(), before);
            let wave = &waves[0];
            assert_eq!(
                wave.status.try_to_vec::<i64>()?,
                [WetWaveStatus::Certified as i64]
            );
            assert_eq!(wave.regions.left_exterior.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(wave.regions.left_fan.try_to_vec::<f64>()?, [0.0]);
            assert_eq!(wave.regions.right_fan.try_to_vec::<f64>()?, [0.0]);
            Ok(())
        },
    )
}

#[test]
fn configured_cpu_wet_wave_certificates_preserve_positions_and_continuation() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    matrix::<CpuRuntime>(&client, Coverage::Full)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_wet_wave_certificates_preserve_positions_and_continuation() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
}
