use super::RootInvocation;
use crate::channel::flow::surveyed_points::{
    chart_test_support,
    source_tests::support as frozen,
    wet_depth::{refine_wet_depth, tests::support},
    wet_roots::locate_wet_roots,
};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

#[path = "root_invocation_tests_transplants.rs"]
mod transplants;

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    bookkeeping::<R>(client)?;
    located_transplant::<R>(client)?;
    exhausted_transfer::<R>(client)?;
    empty_transfer::<R>(client)?;
    transplants::run::<R>(client)
}

fn bookkeeping<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let before = work.snapshot();
            let source = hydraulic(pair.root_witness())?;
            let mut first = hydraulic(RootInvocation::new(source.clone(), 2))?;
            let second = hydraulic(RootInvocation::new(source, 2))?;
            let witness = first.witness();
            hydraulic(first.check(&witness))?;
            assert!(second.check(&witness).is_err());
            assert_eq!(first.point_limit(), 2);
            assert_eq!(first.callback_points(), 0);
            assert_eq!(hydraulic(first.remaining())?, 2);
            hydraulic(first.add_points(1))?;
            assert!(first.add_points(usize::MAX).is_err());
            assert!(first.add_points(2).is_err());
            assert_eq!(first.callback_points(), 1);
            hydraulic(first.add_points(1))?;
            assert_eq!(hydraulic(first.remaining())?, 0);
            assert!(first.add_points(1).is_err());
            assert_eq!(first.callback_points(), 2);
            assert_eq!(first.point_limit(), 2);
            assert_eq!(second.callback_points(), 0);
            let moved = first;
            hydraulic(moved.check(&witness))?;
            assert_eq!(moved.callback_points(), 2);
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

fn located_transplant<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let mut first = hydraulic(locate_wet_roots(pair, &rows, 1))?;
            let mut second = hydraulic(locate_wet_roots(pair, &rows, 1))?;
            assert_eq!(first.callback_points(), 1);
            assert_eq!(second.callback_points(), 1);
            assert_eq!(first.point_limit(), 1);
            assert_eq!(second.point_limit(), 1);
            assert!(
                first
                    .invocation
                    .check(&second.domains[0].invocation_witness)
                    .is_err()
            );
            std::mem::swap(&mut first.domains, &mut second.domains);
            let before = work.snapshot();
            let cache = chart_test_support::cache(pair)?;
            let error = refine_wet_depth(pair, second)
                .err()
                .ok_or("Foreign located domain enters depth mapping")?;
            assert!(format!("{error:?}").contains("another invocation"));
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(pair)?, cache);
            Ok(())
        },
    )
}

fn exhausted_transfer<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    use solvr::optimize::CertifiedRootTolerance;
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        3.0,
        1.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0, 0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, 2))?;
            let identity = located.invocation.witness();
            assert_eq!(located.callback_points(), 2);
            assert_eq!(hydraulic(located.invocation.remaining())?, 0);
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            hydraulic(session.invocation.check(&identity))?;
            assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [0, 0]);
            assert_eq!(session.callback_points(), 2);
            assert_eq!(session.point_limit(), 2);
            let before = work.snapshot();
            for _ in 0..2 {
                session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                    atol: 0.0,
                    rtol: 0.0,
                }))?;
                assert_eq!(session.callback_points(), 2);
                assert_eq!(session.point_limit(), 2);
                assert_eq!(hydraulic(session.invocation.remaining())?, 0);
                for domain in session.domains() {
                    hydraulic(domain.check_invocation(&session.invocation))?;
                }
            }
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

fn empty_transfer<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(client, 1.0, 8.0, &[], &config, |pair, work| {
        let rows = support::rows::<R>(client, &[])?;
        let first = hydraulic(locate_wet_roots(pair, &rows, 0))?;
        let second = hydraulic(locate_wet_roots(pair, &rows, 0))?;
        let identity = first.invocation.witness();
        assert!(second.invocation.check(&identity).is_err());
        let before = work.snapshot();
        let mut session = hydraulic(refine_wet_depth(pair, first))?;
        hydraulic(session.invocation.check(&identity))?;
        assert_eq!(session.point_limit(), 0);
        assert_eq!(session.callback_points(), 0);
        assert!(hydraulic(session.certify_wet_star())?.is_empty());
        assert!(hydraulic(session.certify_wet_waves())?.is_empty());
        assert_eq!(work.snapshot(), before);
        Ok(())
    })
}

#[test]
fn configured_cpu_root_invocation_preserves_allowance_and_domain_identity() -> Result {
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
fn actual_cuda_root_invocation_preserves_allowance_and_domain_identity() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
