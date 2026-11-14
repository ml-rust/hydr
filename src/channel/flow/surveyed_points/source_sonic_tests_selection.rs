use super::super::{
    chart_test_support, point_root_session::PointRootSession, source_sonic_types::SonicSide,
    source_tests::support as frozen, wet_depth::tests::support,
};
use crate::channel::flow::surveyed::error::checked;
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

fn snapshot<R>(
    session: &PointRootSession<'_, '_, R>,
    side: SonicSide,
    owner: i64,
) -> Result<Vec<Vec<u64>>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for request in &session.sonic.requests {
        if request.binding.side != side {
            continue;
        }
        for domain in &request.domains {
            let owners = domain.invocation_owner.try_to_vec::<i64>()?;
            if let Some(position) = owners.iter().position(|value| *value == owner) {
                let rows = support::rows::<R>(
                    session.sources.geometry.client(),
                    &[i64::try_from(position)?],
                )?;
                let query = hydraulic(checked::<R, _>(
                    session.sources.geometry.client(),
                    "selected sonic cached-query snapshot",
                    |a| domain.query.select(a, &rows),
                ))?;
                let mut fields = Vec::new();
                for tensor in [
                    query.area.lower(),
                    query.area.upper(),
                    query.query.parameter.lower(),
                    query.query.parameter.upper(),
                    query.query.scale.high(),
                    query.query.scale.low(),
                    query.query.scale.abs_error(),
                    query.above_start.lower(),
                    query.above_start.upper(),
                    query.below_end.lower(),
                    query.below_end.upper(),
                    &query.candidate,
                ] {
                    fields.push(
                        tensor
                            .try_to_vec::<f64>()?
                            .into_iter()
                            .map(f64::to_bits)
                            .collect(),
                    );
                }
                for tensor in [
                    domain.depth.lower(),
                    domain.depth.upper(),
                    domain.normalized_depth.lower(),
                    domain.normalized_depth.upper(),
                    domain.epsilon.high(),
                    domain.epsilon.low(),
                    domain.epsilon.abs_error(),
                ] {
                    fields.push(
                        tensor
                            .narrow(0, position, 1)?
                            .try_to_vec::<f64>()?
                            .into_iter()
                            .map(f64::to_bits)
                            .collect(),
                    );
                }
                fields.push(
                    domain
                        .status
                        .narrow(0, position, 1)?
                        .try_to_vec::<i64>()?
                        .into_iter()
                        .map(|value| value as u64)
                        .collect(),
                );
                return Ok(fields);
            }
        }
    }
    Err("Selected sonic snapshot owner is absent".into())
}

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let epsilon = 2.0_f64.powi(-50);
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 1., 2. - epsilon], [1., 0., 1., 2. - 2. * epsilon]],
        &config,
        |pair, work| {
            let owners = support::rows::<R>(client, &[0, 1])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            hydraulic(session.start_source_sonic(&owners, SonicSide::Right))?;
            hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
            assert!(
                session.sonic.requests[0]
                    .domains
                    .iter()
                    .all(|d| d.refinement.is_none())
            );
            assert!(
                session.sonic.requests[1]
                    .domains
                    .iter()
                    .all(|d| d.refinement.is_some())
            );
            let untouched_right = snapshot(&session, SonicSide::Right, 0)?;
            let untouched_left = [
                snapshot(&session, SonicSide::Left, 0)?,
                snapshot(&session, SonicSide::Left, 1)?,
            ];
            let locating = session
                .sonic
                .requests
                .iter()
                .map(|r| {
                    (
                        r.endpoint_points,
                        r.bracket_points,
                        r.bracket_expansions,
                        r.tolerance.atol.to_bits(),
                        r.tolerance.rtol.to_bits(),
                    )
                })
                .collect::<Vec<_>>();
            let target_owner = support::rows::<R>(client, &[1])?;
            let upper = session.sonic.requests[0].domains[0]
                .normalized_depth
                .upper()
                .narrow(0, 1, 1)?
                .item::<f64>()?;
            let target = upper * 0.25;
            assert!(target > 0. && target < upper * 0.5);
            let right_depth_before = [
                session.sonic.requests[0].domains[0]
                    .depth
                    .lower()
                    .narrow(0, 1, 1)?
                    .item::<f64>()?,
                session.sonic.requests[0].domains[0]
                    .depth
                    .upper()
                    .narrow(0, 1, 1)?
                    .item::<f64>()?,
            ];
            let right_callbacks_before = session.sonic.requests[0]
                .domains
                .iter()
                .map(|domain| domain.callback_points)
                .sum::<usize>();
            let right_session_before = session.callback_points();
            let right_work_before = work.snapshot().residual_points;
            hydraulic(session.refine_source_sonic_selected(
                &target_owner,
                SonicSide::Right,
                CertifiedRootTolerance {
                    atol: target,
                    rtol: 0.,
                },
            ))?;
            let right_work = work
                .snapshot()
                .residual_points
                .checked_sub(right_work_before);
            assert!(right_work.is_some_and(|points| points > 0));
            assert_eq!(
                session.callback_points().checked_sub(right_session_before),
                right_work
            );
            assert_eq!(
                session.sonic.requests[0]
                    .domains
                    .iter()
                    .map(|domain| domain.callback_points)
                    .sum::<usize>()
                    .checked_sub(right_callbacks_before),
                right_work
            );
            assert_eq!(untouched_right, snapshot(&session, SonicSide::Right, 0)?);
            for (owner, before) in untouched_left.iter().enumerate() {
                assert_eq!(
                    *before,
                    snapshot(&session, SonicSide::Left, i64::try_from(owner)?)?
                );
            }
            assert_eq!(session.sonic.requests[0].domains.len(), 2);
            let mut preserved_deferred = false;
            let mut refined_right = false;
            for domain in &session.sonic.requests[0].domains {
                if domain.invocation_owner.try_to_vec::<i64>()? == [1] {
                    assert!(domain.refinement.is_some());
                    assert!(domain.callback_points > 0 && domain.mapped_points > 0);
                    let depth = frozen::endpoints(&domain.depth)?[0];
                    let normalized = frozen::endpoints(&domain.normalized_depth)?[0];
                    assert!(depth[0] > 0. && normalized[0] > 0.);
                    assert!(frozen::endpoints(&domain.query.area)?[0][0] > 0.);
                    assert!(
                        depth[1] - depth[0] < right_depth_before[1] - right_depth_before[0]
                            || normalized[1] - normalized[0] <= target,
                        "Demanded Right owner retains its previous depth interval: {depth:?}, normalized={normalized:?}, target={target:e}"
                    );
                    refined_right = true;
                }
                if domain.refinement.is_none()
                    && domain.invocation_owner.try_to_vec::<i64>()? == [0]
                {
                    assert_eq!(
                        [
                            domain.iterations,
                            domain.callback_points,
                            domain.mapped_points
                        ],
                        [0; 3]
                    );
                    preserved_deferred = true;
                }
            }
            assert!(preserved_deferred);
            assert!(refined_right);
            let right_after = [
                snapshot(&session, SonicSide::Right, 0)?,
                snapshot(&session, SonicSide::Right, 1)?,
            ];
            let left_zero = snapshot(&session, SonicSide::Left, 0)?;
            let domain = &session.sonic.requests[1].domains[0];
            let lower = domain
                .normalized_depth
                .lower()
                .narrow(0, 1, 1)?
                .item::<f64>()?;
            let upper = domain
                .normalized_depth
                .upper()
                .narrow(0, 1, 1)?
                .item::<f64>()?;
            let target = (upper - lower) * 0.25;
            assert!(target > 0.);
            let left_depth_before = [
                domain.depth.lower().narrow(0, 1, 1)?.item::<f64>()?,
                domain.depth.upper().narrow(0, 1, 1)?.item::<f64>()?,
            ];
            let left_callbacks_before = domain.callback_points;
            let left_mapped_before = domain.mapped_points;
            let left_session_before = session.callback_points();
            let left_work_before = work.snapshot().residual_points;
            hydraulic(session.refine_source_sonic_selected(
                &target_owner,
                SonicSide::Left,
                CertifiedRootTolerance {
                    atol: target,
                    rtol: 0.,
                },
            ))?;
            let left_work = work
                .snapshot()
                .residual_points
                .checked_sub(left_work_before);
            assert!(left_work.is_some_and(|points| points > 0));
            assert_eq!(
                session.callback_points().checked_sub(left_session_before),
                left_work
            );
            let mut refined_left = false;
            for domain in &session.sonic.requests[1].domains {
                let owners = domain.invocation_owner.try_to_vec::<i64>()?;
                if let Some(position) = owners.iter().position(|owner| *owner == 1) {
                    assert!(domain.refinement.is_some());
                    assert!(domain.mapped_points > left_mapped_before);
                    assert_eq!(
                        domain.callback_points.checked_sub(left_callbacks_before),
                        left_work
                    );
                    let depth = frozen::endpoints(&domain.depth)?[position];
                    let normalized = frozen::endpoints(&domain.normalized_depth)?[position];
                    assert!(depth[0] > 0. && normalized[0] > 0.);
                    assert!(frozen::endpoints(&domain.query.area)?[position][0] > 0.);
                    assert!(
                        depth[1] - depth[0] < left_depth_before[1] - left_depth_before[0]
                            || normalized[1] - normalized[0] <= target,
                        "Demanded Left owner retains its previous depth interval: {depth:?}, normalized={normalized:?}, target={target:e}"
                    );
                    refined_left = true;
                }
            }
            assert!(refined_left);
            assert_eq!(left_zero, snapshot(&session, SonicSide::Left, 0)?);
            for (owner, before) in right_after.iter().enumerate() {
                assert_eq!(
                    *before,
                    snapshot(&session, SonicSide::Right, i64::try_from(owner)?)?
                );
            }
            assert_eq!(
                locating,
                session
                    .sonic
                    .requests
                    .iter()
                    .map(|r| (
                        r.endpoint_points,
                        r.bracket_points,
                        r.bracket_expansions,
                        r.tolerance.atol.to_bits(),
                        r.tolerance.rtol.to_bits()
                    ))
                    .collect::<Vec<_>>()
            );
            assert_eq!(session.callback_points(), work.snapshot().residual_points);
            assert_eq!(session.point_limit(), config.root.max_points);
            Ok(())
        },
    )
}

#[test]
fn configured_cpu_selected_source_sonic_preserves_unrequested_states() -> Result {
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
fn actual_cuda_selected_source_sonic_preserves_unrequested_states() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
