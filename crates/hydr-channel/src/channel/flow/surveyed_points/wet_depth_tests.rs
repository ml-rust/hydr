use crate::channel::flow::surveyed_points::source_tests::support as frozen;
use frozen::Result;
use numr::{dtype::DType, runtime::Runtime};

#[path = "wet_depth_tests_admission.rs"]
mod admission;
#[path = "wet_depth_tests_budget.rs"]
mod budget;
#[path = "wet_depth_tests_lifecycle.rs"]
mod lifecycle;
#[path = "wet_depth_tests_partition.rs"]
mod partition;
#[path = "wet_depth_tests_support.rs"]
pub(in crate::channel::flow::surveyed_points) mod support;

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    lifecycle::run::<R>(client)?;
    partition::run::<R>(client)?;
    admission::run::<R>(client)?;
    budget::run::<R>(client)
}

#[test]
fn configured_cpu_wet_depth_preserves_physical_certificates_and_continuation() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    matrix::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_wet_depth_preserves_physical_certificates_and_continuation() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
