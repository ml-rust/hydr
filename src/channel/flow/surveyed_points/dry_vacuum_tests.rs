use super::super::{
    chart_test_support,
    dry_vacuum_types::{DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus},
    point_root_session::PointRootSession,
    source_tests::support as frozen,
    wet_depth::tests::support,
};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

#[path = "dry_vacuum_tests_controls.rs"]
mod controls;
#[path = "dry_vacuum_tests_ownership.rs"]
mod ownership;
#[path = "dry_vacuum_tests_references.rs"]
mod references;

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 0., 0.], [0., 0., 0., 0.]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0, 1, 0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            let owners = support::rows::<R>(client, &[2, 1, 0])?;
            let result = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(result.invocation_owner.try_to_vec::<i64>()?, [2, 1, 0]);
            assert_eq!(result.parent_rows.try_to_vec::<i64>()?, [0, 1, 0]);
            assert_eq!(
                result.regime.try_to_vec::<i64>()?,
                [
                    DryVacuumRegime::LeftWet as i64,
                    DryVacuumRegime::BothDry as i64,
                    DryVacuumRegime::LeftWet as i64
                ]
            );
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [DryVacuumStatus::Complete as i64; 3]
            );
            let fan = result
                .candidates
                .iter()
                .find_map(|candidate| {
                    if let DryVacuumCandidate::Fan(fan) = candidate {
                        Some(fan)
                    } else {
                        None
                    }
                })
                .ok_or("Missing admitted sonic candidate")?;
            for bounds in frozen::endpoints(&fan.query.area)? {
                assert!(bounds[0] <= 4. / 9. && 4. / 9. <= bounds[1]);
            }
            let dry = result
                .candidates
                .iter()
                .find_map(|candidate| {
                    if let DryVacuumCandidate::Dry(dry) = candidate {
                        Some(dry)
                    } else {
                        None
                    }
                })
                .ok_or("Missing exact dry candidate")?;
            for bounds in frozen::endpoints(&dry.depth)? {
                assert_eq!(bounds, [0., 0.]);
            }
            let points = session.callback_points();
            let before = work.snapshot().residual_points;
            let repeated = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(
                repeated.status.try_to_vec::<i64>()?,
                result.status.try_to_vec::<i64>()?
            );
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before);
            Ok(())
        },
    )?;
    references::run::<R>(client)?;
    controls::run::<R>(client)?;
    ownership::run::<R>(client)
}
#[test]
fn configured_cpu_dry_vacuum_candidates_preserve_origin_alternatives() -> Result {
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
fn actual_cuda_dry_vacuum_candidates_preserve_origin_alternatives() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
