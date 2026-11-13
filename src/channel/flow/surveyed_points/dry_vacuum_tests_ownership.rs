use super::super::super::source_sonic_types::SonicSide;
use super::{
    DryVacuumCandidate, PointRootSession, Result, chart_test_support, frozen, hydraulic, support,
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    selected_snapshots::<R>(client)?;
    foreign_output::<R>(client)?;
    preflight::<R>(client)?;
    foreign_state::<R>(client)
}
fn selected_snapshots<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 0., 0.], [0., 0., 1., 0.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 1, 0])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let started = support::rows::<R>(client, &[2, 0])?;
            hydraulic(session.start_source_sonic(&started, SonicSide::Left))?;
            let before = work.snapshot();
            let points = session.callback_points();
            let owners = support::rows::<R>(client, &[0])?;
            let result = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(work.snapshot().root_iterations, before.root_iterations);
            assert_eq!(session.callback_points(), points);
            assert_eq!(session.sonic.requests.len(), 1);
            let fan = result
                .candidates
                .iter()
                .find_map(|v| {
                    if let DryVacuumCandidate::Fan(f) = v {
                        Some(f)
                    } else {
                        None
                    }
                })
                .ok_or("Missing demanded existing sonic root")?;
            assert_eq!(fan.invocation_owner.try_to_vec::<i64>()?, [0]);
            assert_eq!(fan.parent_rows.try_to_vec::<i64>()?, [0]);
            assert_eq!(fan.query.query.original_rows.try_to_vec::<i64>()?, [101]);
            assert_eq!(result.candidates.len(), 1);
            let original =
                frozen::endpoints(&session.sonic.requests[0].domains[0].query.query.parameter)?;
            let evidence = frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?;
            let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
            let changed = Tensor::full_scalar(&[1], DType::I64, 99., client.device())?;
            client.copy_into(fan.query.query.parameter.lower(), &zero)?;
            client.copy_into(fan.depth.lower(), &zero)?;
            client.copy_into(&result.invocation_owner, &changed)?;
            client.copy_into(&result.parent_rows, &changed)?;
            client.copy_into(result.edges[0].head.physical.lower(), &zero)?;
            assert_eq!(
                frozen::endpoints(&session.sonic.requests[0].domains[0].query.query.parameter)?,
                original
            );
            assert_eq!(
                frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?,
                evidence
            );
            assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [0, 1, 0]);
            let repeated = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(repeated.invocation_owner.try_to_vec::<i64>()?, [0]);
            hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                atol: 1e-17,
                rtol: 0.,
            }))?;
            let resumed = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(resumed.candidates.len(), 1);
            assert!(session.callback_points() >= points);
            assert!(session.callback_points() <= config.root.max_points);
            let right = hydraulic(session.certify_dry_vacuum(&support::rows::<R>(client, &[1])?))?;
            assert_eq!(right.candidates.len(), 1);
            assert_eq!(session.sonic.requests.len(), 2);
            Ok(())
        },
    )
}
fn foreign_output<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[0., 0., 0., 0.]],
        &config,
        |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let result = {
                let mut first = hydraulic(PointRootSession::new(pair, &owners, 1))?;
                let result = hydraulic(first.certify_dry_vacuum(&owners))?;
                hydraulic(first.invocation.check(&result.invocation_witness))?;
                result
            };
            let second = hydraulic(PointRootSession::new(pair, &owners, 1))?;
            let before = work.snapshot();
            assert!(second.invocation.check(&result.invocation_witness).is_err());
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}
fn preflight<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    for alteration in 0..4 {
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[[1., 0., 0., 0.]],
            &config,
            |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let parents = support::rows::<R>(client, &[0, 0])?;
                let mut session = hydraulic(PointRootSession::new(
                    pair,
                    &parents,
                    config.root.max_points,
                ))?;
                hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
                for bad in [&[-1][..], &[2][..], &[0, 0][..]] {
                    let before = work.snapshot();
                    let cache = chart_test_support::cache(session.sources())?;
                    assert!(
                        session
                            .certify_dry_vacuum(&support::rows::<R>(client, bad)?)
                            .is_err()
                    );
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_test_support::cache(session.sources())?, cache);
                }
                match alteration {
                    0 => session.sonic.requests[0].terminal = true,
                    1 => session.sonic.requests[0].binding.side = SonicSide::Right,
                    2 => {
                        session.sonic.requests[0].domains[0].parent_rows =
                            support::rows::<R>(client, &[1])?
                    }
                    3 => session.sources.evaluation.panels *= 2,
                    _ => return Err("Unexpected preflight alteration".into()),
                }
                let before = work.snapshot();
                let cache = chart_test_support::cache(session.sources())?;
                assert!(
                    session
                        .certify_dry_vacuum(&support::rows::<R>(client, &[1])?)
                        .is_err()
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(session.sources())?, cache);
                Ok(())
            },
        )?;
    }
    Ok(())
}
fn foreign_state<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 0., 0.]],
        &config,
        |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let donor = {
                let mut first =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                hydraulic(first.start_source_sonic(&owners, SonicSide::Left))?;
                first
                    .sonic
                    .requests
                    .pop()
                    .ok_or("Missing foreign request")?
            };
            let mut second =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            second.sonic.requests.push(donor);
            let before = work.snapshot();
            let cache = chart_test_support::cache(second.sources())?;
            assert!(second.certify_dry_vacuum(&owners).is_err());
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(second.sources())?, cache);
            Ok(())
        },
    )
}
