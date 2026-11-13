use super::super::{
    chart_test_support,
    point_root_session::PointRootSession,
    source_sonic_types::{SonicFanMembership, SonicSide, SourceSonicStatus},
    source_tests::support as frozen,
    wet_depth::tests::support,
    wet_root_types::WetRootStatus,
};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

#[path = "source_sonic_tests_contacts.rs"]
mod contacts;
#[path = "source_sonic_tests_diagnostics.rs"]
mod diagnostics;
#[path = "source_sonic_tests_ownership.rs"]
mod ownership;
#[path = "source_sonic_tests_pointed.rs"]
mod pointed;
#[path = "source_sonic_tests_references.rs"]
mod references;
#[path = "source_sonic_tests_work.rs"]
mod work;

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
        &[[1.0, 0.0, 0.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0, 0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            let owners = support::rows::<R>(client, &[1, 0])?;
            hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
            let points = session.callback_points();
            let before = work.snapshot();
            let result = hydraulic(session.certify_source_sonic())?;
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(
                session.locator_status().try_to_vec::<i64>()?,
                [WetRootStatus::NotRequested as i64; 2]
            );
            assert_eq!(
                result.status[0].try_to_vec::<i64>()?,
                [SourceSonicStatus::PositiveRoot as i64; 2]
            );
            assert_eq!(
                result.status[1].try_to_vec::<i64>()?,
                [SourceSonicStatus::NotRequested as i64; 2]
            );
            assert_eq!(result.domains.len(), 1);
            let domain = &result.domains[0];
            assert_eq!(domain.invocation_owner.try_to_vec::<i64>()?, [1, 0]);
            assert_eq!(domain.parent_rows.try_to_vec::<i64>()?, [0, 0]);
            assert_eq!(domain.fan_membership, SonicFanMembership::NotChecked);
            assert_eq!(domain.status.try_to_vec::<i64>()?, [0, 0]);
            // Independent rectangle sonic root: A=4/9, u=2/3, Q=8/27.
            for (actual, expected) in [
                (&domain.query.area, [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d]),
                (&domain.velocity, [0x3fe5555555555555, 0x3fe5555555555556]),
                (&domain.discharge, [0x3fd2f684bda12f68, 0x3fd2f684bda12f69]),
            ] {
                let expected = expected.map(f64::from_bits);
                for actual in frozen::endpoints(actual)? {
                    assert!(
                        actual[0] <= expected[0] && expected[1] <= actual[1],
                        "{actual:?} excludes {expected:?}"
                    );
                }
            }
            Ok(())
        },
    )?;
    references::run::<R>(client)?;
    pointed::run::<R>(client)?;
    contacts::run::<R>(client)?;
    work::run::<R>(client)?;
    ownership::run::<R>(client)
}

#[test]
fn configured_cpu_source_sonic_roots_preserve_selected_side_and_owner() -> Result {
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
fn actual_cuda_source_sonic_roots_preserve_selected_side_and_owner() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
