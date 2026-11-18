use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

pub(super) fn client() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    (device, client)
}

pub(super) fn strided(
    device: &CpuDevice,
    values: &[f64],
) -> numr::error::Result<Tensor<CpuRuntime>> {
    let padded: Vec<_> = values.iter().flat_map(|value| [-99.0, *value]).collect();
    Ok(Tensor::from_slice(&padded, &[values.len(), 2], device)?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)))
}
