use super::{
    PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen, hydraulic,
    support,
};
use crate::channel::flow::surveyed_points::{
    root_invocation::RootInvocation, wet_root_types::WetRootStatus,
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
    mixed_requests::<R>(client)?;
    mutations::<R>(client)?;
    nested_continuation::<R>(client)?;
    foreign_invocation::<R>(client, false)?;
    foreign_invocation::<R>(client, true)?;
    stale_policy::<R>(client)?;
    owned_snapshots::<R>(client)
}

fn mixed_requests<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[2, 0, 2, 1])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            hydraulic(session.start_wet(&support::rows::<R>(client, &[2, 0])?))?;
            assert_eq!(
                session.domains()[0].invocation_owner.try_to_vec::<i64>()?,
                [2, 0]
            );
            assert_eq!(
                session.domains()[0].parent_rows.try_to_vec::<i64>()?,
                [2, 2]
            );
            let located_points = session.callback_points();
            hydraulic(
                session.start_source_sonic(&support::rows::<R>(client, &[1])?, SonicSide::Left),
            )?;
            hydraulic(
                session.start_source_sonic(&support::rows::<R>(client, &[3])?, SonicSide::Right),
            )?;
            assert_eq!(
                session.locator_status().try_to_vec::<i64>()?,
                [
                    WetRootStatus::Located as i64,
                    WetRootStatus::NotRequested as i64,
                    WetRootStatus::Located as i64,
                    WetRootStatus::NotRequested as i64
                ]
            );
            let before = work.snapshot();
            let cache = chart_test_support::cache(session.sources())?;
            for bad in [vec![1, 1], vec![-1], vec![4]] {
                assert!(
                    session
                        .start_source_sonic(&support::rows::<R>(client, &bad)?, SonicSide::Left)
                        .is_err()
                );
            }
            assert!(
                session
                    .start_source_sonic(&support::rows::<R>(client, &[1])?, SonicSide::Left)
                    .is_err()
            );
            assert!(
                session
                    .start_wet(&support::rows::<R>(client, &[0])?)
                    .is_err()
            );
            assert!(
                session
                    .start_wet(&support::rows::<R>(client, &[1])?)
                    .is_err()
            );
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            // Duplicate source rows remain independent invocation owners and can start separate histories.
            hydraulic(
                session.start_source_sonic(&support::rows::<R>(client, &[0])?, SonicSide::Left),
            )?;
            hydraulic(
                session.start_source_sonic(&support::rows::<R>(client, &[2])?, SonicSide::Left),
            )?;
            hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                atol: 1e-17,
                rtol: 0.0,
            }))?;
            let result = hydraulic(session.certify_source_sonic())?;
            super::diagnostics::evidence(&result)?;
            assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, [2, 0, 2, 1]);
            for domain in &result.domains {
                let owners = domain.invocation_owner.try_to_vec::<i64>()?;
                let rows = domain.parent_rows.try_to_vec::<i64>()?;
                let originals = domain.query.query.original_rows.try_to_vec::<i64>()?;
                for (index, owner) in owners.into_iter().enumerate() {
                    assert_eq!(rows[index], [2, 0, 2, 1][usize::try_from(owner)?]);
                    assert_eq!(originals[index], 101 + rows[index]);
                }
            }
            let sonic_points = session
                .sonic
                .requests
                .iter()
                .map(|request| {
                    request.endpoint_points
                        + request.bracket_points
                        + request
                            .domains
                            .iter()
                            .map(|domain| domain.callback_points)
                            .sum::<usize>()
                })
                .sum::<usize>();
            let wet_points = session
                .domains()
                .iter()
                .map(|domain| domain.callback_points)
                .sum::<usize>();
            assert_eq!(
                session.callback_points(),
                located_points + sonic_points + wet_points
            );
            assert_eq!(session.callback_points(), work.snapshot().residual_points);
            assert_eq!(session.point_limit(), config.root.max_points);
            Ok(())
        },
    )
}

fn mutations<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    for mutation in 0..6 {
        let state = if mutation == 5 {
            [1.0, 2.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 1.0, 0.0]
        };
        support::with_rectangle::<R, _>(client, 1.0, 4.0, &[state], &config, |pair, work| {
            let rows = support::rows::<R>(client, &[0, 0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            let owner = support::rows::<R>(client, &[0])?;
            hydraulic(session.start_source_sonic(&owner, SonicSide::Left))?;
            match mutation {
                0 => session.sonic.requests[0].domains[0].side = SonicSide::Right,
                1 => {
                    session.sonic.requests[0].domains[0].parent_rows =
                        support::rows::<R>(client, &[1])?
                }
                2 => {
                    session.sonic.requests[0].domains[0].invocation_owner =
                        support::rows::<R>(client, &[1])?
                }
                3 => {
                    session.sonic.requests[0].domains[0].iterations = config.root.max_iterations + 1
                }
                4 => {
                    session.invocation = hydraulic(RootInvocation::new(
                        hydraulic(session.sources.root_witness())?,
                        config.root.max_points,
                    ))?
                }
                5 => {
                    assert!(session.sonic.requests[0].domains.is_empty());
                    assert_eq!(
                        session.sonic.requests[0].status.try_to_vec::<i64>()?,
                        [SourceSonicStatus::NoRootSource as i64]
                    );
                    session.sonic.requests[0].binding.side = SonicSide::Right;
                }
                _ => return Err("Unexpected binding mutation".into()),
            }
            let before = work.snapshot();
            let cache = chart_test_support::cache(session.sources())?;
            assert!(session.certify_source_sonic().is_err());
            assert!(
                session
                    .refine_source_sonic(CertifiedRootTolerance {
                        atol: config.root.atol,
                        rtol: config.root.rtol
                    })
                    .is_err()
            );
            assert!(
                session
                    .start_source_sonic(&support::rows::<R>(client, &[1])?, SonicSide::Right)
                    .is_err()
            );
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            Ok(())
        })?;
    }
    Ok(())
}

fn nested_continuation<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0, 0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            for owner in [0, 1] {
                hydraulic(
                    session.start_source_sonic(
                        &support::rows::<R>(client, &[owner])?,
                        SonicSide::Left,
                    ),
                )?;
            }
            let donor = session.sonic.requests[0]
                .domains
                .pop()
                .ok_or("Missing donor root")?;
            let state = donor
                .refinement
                .ok_or("Donor lacks initialized continuation")?;
            assert!(session.sonic.requests[1].domains[0].refinement.is_some());
            session.sonic.requests[1].domains[0].refinement = Some(state);
            let before = work.snapshot();
            let cache = chart_test_support::cache(session.sources())?;
            assert!(session.certify_source_sonic().is_err());
            assert!(
                session
                    .refine_source_sonic(CertifiedRootTolerance {
                        atol: config.root.atol,
                        rtol: config.root.rtol
                    })
                    .is_err()
            );
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            Ok(())
        },
    )
}

fn foreign_invocation<R>(client: &R::Client, initialized: bool) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let id = if initialized {
        "scalar_rectangle_right_dry_sonic"
    } else {
        "retained_vacuum_inside_film"
    };
    let side = if initialized {
        SonicSide::Left
    } else {
        SonicSide::Right
    };
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let rows = support::rows::<R>(client, &[0])?;
        let donor = {
            let mut first = hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            hydraulic(first.start_source_sonic(&rows, side))?;
            assert_eq!(
                first.sonic.requests[0].domains[0].refinement.is_some(),
                initialized
            );
            first.sonic.requests[0]
                .domains
                .pop()
                .ok_or("Missing foreign source-sonic domain")?
        };
        let mut second = hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
        hydraulic(second.start_source_sonic(&rows, side))?;
        second.sonic.requests[0].domains[0] = donor;
        let before = work.snapshot();
        let cache = chart_test_support::cache(second.sources())?;
        assert!(second.certify_source_sonic().is_err());
        assert!(
            second
                .refine_source_sonic(CertifiedRootTolerance {
                    atol: config.root.atol,
                    rtol: config.root.rtol
                })
                .is_err()
        );
        assert_eq!(work.snapshot(), before);
        assert_eq!(chart_test_support::cache(second.sources())?, cache);
        Ok(())
    })
}

fn stale_policy<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            hydraulic(session.start_source_sonic(&rows, SonicSide::Left))?;
            session.sources.evaluation.panels *= 2;
            let before = work.snapshot();
            let cache = chart_test_support::cache(session.sources())?;
            assert!(session.start_source_sonic(&rows, SonicSide::Right).is_err());
            assert!(
                session
                    .refine_source_sonic(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0
                    })
                    .is_err()
            );
            assert!(session.certify_source_sonic().is_err());
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            Ok(())
        },
    )
}

fn owned_snapshots<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, _| {
            let rows = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            hydraulic(session.start_source_sonic(&rows, SonicSide::Left))?;
            let parameter =
                frozen::endpoints(&session.sonic.requests[0].domains[0].query.query.parameter)?;
            let depth = frozen::endpoints(&session.sonic.requests[0].domains[0].depth)?;
            let evidence = frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?;
            let result = hydraulic(session.certify_source_sonic())?;
            let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
            let changed = Tensor::full_scalar(&[1], DType::I64, 99.0, client.device())?;
            client.copy_into(result.domains[0].query.query.parameter.lower(), &zero)?;
            client.copy_into(result.domains[0].depth.lower(), &zero)?;
            client.copy_into(result.evidence[0].source_residual.physical.lower(), &zero)?;
            client.copy_into(&result.invocation_rows, &changed)?;
            assert_eq!(
                frozen::endpoints(&session.sonic.requests[0].domains[0].query.query.parameter)?,
                parameter
            );
            assert_eq!(
                frozen::endpoints(&session.sonic.requests[0].domains[0].depth)?,
                depth
            );
            assert_eq!(
                frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?,
                evidence
            );
            assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [0]);
            hydraulic(session.certify_source_sonic())?;
            Ok(())
        },
    )
}
