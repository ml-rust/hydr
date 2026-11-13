use super::PointRootSession;
use crate::channel::flow::surveyed_points::{
    chart_test_support,
    source_tests::support::{Result, hydraulic},
    wet_depth::tests::support,
    wet_root_types::WetRootStatus,
};
use numr::{dtype::DType, runtime::Runtime};

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[1, 0, 1])?;
            let before = work.snapshot();
            let session = hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [1, 0, 1]);
            assert_eq!(
                session.locator_status().try_to_vec::<i64>()?,
                [WetRootStatus::NotRequested as i64; 3]
            );
            assert!(session.domains().is_empty());
            assert_eq!(session.callback_points(), 0);
            assert_eq!(session.point_limit(), config.root.max_points);
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

#[test]
fn configured_cpu_point_root_session_captures_nonwet_invocation_rows() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    matrix::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
}
#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_point_root_session_captures_nonwet_invocation_rows() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
