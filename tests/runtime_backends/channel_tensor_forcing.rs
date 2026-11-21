#[cfg(any(feature = "cuda", feature = "wgpu"))]
use super::Result;

#[allow(dead_code)]
#[path = "../channel_boundaries/common.rs"]
pub(super) mod boundary_common;
#[path = "../channel_tensor_forcing/snapshot_bits.rs"]
pub(super) mod snapshot_bits;
#[allow(dead_code)]
#[path = "../channel_tensor_forcing/transport.rs"]
pub(super) mod transport;

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_executes_native_channel_forcing_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    transport::execute::<CudaRuntime>(
        CudaClient::new(CudaDevice::new(0))?,
        transport::device_runs(),
    )
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_native_channel_f64_without_fallback() -> Result {
    use hydr::channel::ChannelReach;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    let shape = boundary_common::Shape::Rectangle {
        gravity: 1.0,
        width: 7.0,
    };
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    let held = transport::forcing(0.0, 1.0 / 1024.0, 2);
    let _native = transport::native::<WgpuRuntime>(&client, &held, 0)?;
    let error = ChannelReach::<WgpuRuntime>::new(
        client,
        boundary_common::definition(shape, 1, 4.0, 4.0, 1.0),
        boundary_common::initial(shape, &[4.0], &[1.0], &[0.0]),
    )
    .err()
    .ok_or("WGPU admits unsupported native channel F64")?;
    assert!(matches!(
        error,
        hydr::channel::ChannelError::InvalidInput { .. }
    ));
    assert!(error.to_string().contains("F64"));
    Ok(())
}
