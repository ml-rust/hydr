use super::channel_tensor_forcing::{boundary_common, snapshot_bits, transport};

#[path = "../channel_reservoir/fixtures.rs"]
mod fixtures;
#[allow(dead_code)]
#[path = "../channel_reservoir/native.rs"]
mod native;

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_executes_native_reservoir_roots_mixed_endpoints_and_complete_restart()
-> super::Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    native::execute::<CudaRuntime>(
        CudaClient::new(CudaDevice::new(0))?,
        native::Selection::device(),
    )
}
