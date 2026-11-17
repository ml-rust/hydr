use super::*;
use crate::channel::flow::surveyed_points::{
    chart_test_support,
    source_tests::support::{Client, Result, components, endpoints, hydraulic},
};
use solvr::optimize::CertifiedBisectionAlgorithms;

#[path = "paired_source_panels_tests_admission.rs"]
mod admission;
#[path = "paired_source_panels_tests_behavior.rs"]
mod behavior;

fn basic<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let config = chart_test_support::config();
    chart_test_support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_depth_jump"],
        &config,
        |pair, work| {
            let rows = Tensor::from_slice(&[0_i64, 0], &[2], client.device())?;
            let empty = Tensor::empty(&[0], DType::I64, client.device())?;
            let level0 = hydraulic(pair.panel_level(0))?;
            let level1 = hydraulic(pair.panel_level(1))?;
            assert_eq!(level0.index(), 0);
            assert_eq!(level1.panels(), 2 * level0.panels());
            let original = hydraulic(pair.source_values([&rows, &empty]))?;
            let baseline = chart_test_support::cache(pair)?;
            let cached = hydraulic(pair.source_values_at_level(&level0, [&rows, &empty]))?;
            assert_eq!(
                components(&original[0].retained)?,
                components(&cached.values[0].retained)?
            );
            assert_eq!(
                endpoints(&original[0].physical)?,
                endpoints(&cached.values[0].physical)?
            );
            let higher = hydraulic(pair.source_values_at_level(&level1, [&rows, &empty]))?;
            assert_eq!(higher.level.index(), 1);
            assert_eq!(pair.panel_cache.len(), 1);
            assert_eq!(chart_test_support::cache(pair)?, baseline);
            let before = work.snapshot();
            let repeated = hydraulic(pair.source_values_at_level(&level1, [&rows, &empty]))?;
            assert_eq!(
                components(&higher.values[0].retained)?,
                components(&repeated.values[0].retained)?
            );
            assert_eq!(work.snapshot(), before);
            assert_eq!(higher.values[0].available.try_to_vec::<f64>()?, [1.0, 1.0]);
            Ok(())
        },
    )
}

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    basic::<R>(client)?;
    behavior::truth_and_dedup::<R>(client)?;
    behavior::work_error::<R>(client)?;
    admission::run::<R>(client)?;
    assert!(storage::<R>(usize::MAX, DType::F64, 2).is_err());
    assert!(storage::<R>(1, DType::F64, usize::MAX).is_err());
    Ok(())
}

#[test]
fn configured_cpu_source_panel_levels_preserve_coarse_cache() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    matrix::<CpuRuntime>(&client)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_source_panel_levels_preserve_coarse_cache() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    matrix::<CudaRuntime>(&client)?;
    admission::foreign_device(&client)
}
