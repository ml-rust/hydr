#[allow(dead_code)]
#[path = "../channel_pools/fixture.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../channel_pools/native.rs"]
mod native;
#[path = "../channel_pools/reference.rs"]
mod reference;

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_executes_finite_pool_non_knot_cuts_mixed_connections_and_restart() -> super::Result
{
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    native::execute::<CudaRuntime>(
        CudaClient::new(CudaDevice::new(0))?,
        CudaClient::new(CudaDevice::new(0))?,
    )
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_finite_channel_pool_f64_without_fallback() -> super::Result {
    use hydr::channel::ChannelError;
    use hydr::coupling::{ChannelPoolBatch, ChannelPoolEndpoint, ChannelPoolError};
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    let (setup, initial) = fixture::single(reference::CASES[0], ChannelPoolEndpoint::Right)?;
    let error =
        ChannelPoolBatch::<WgpuRuntime>::new(WgpuClient::new(WgpuDevice::new(0))?, setup, initial)
            .err()
            .ok_or("WGPU accepts unsupported finite channel/pool F64")?;
    let ChannelPoolError::Channel { reach_id, source } = error else {
        return Err(format!("Unexpected WGPU coordinator rejection: {error}").into());
    };
    assert_eq!(reach_id, "reach");
    assert!(matches!(source, ChannelError::InvalidInput { .. }));
    assert!(source.to_string().contains("F64"));
    Ok(())
}
