use super::{
    PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen, hydraulic,
    support,
};
use crate::channel::flow::{
    surveyed::{
        error::{HydraulicError, PrismaticRiemannRejection},
        work::QueryWork,
    },
    surveyed_points::wet_root_types::WetRootStatus,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    narrow_capture::<R>(client)?;
    independent_rounds::<R>(client)?;
    post_callback_error::<R>(client)?;
    accounting_mismatch::<R>(client)?;
    wet_error_seals_certificates::<R>(client)
}

fn narrow_capture<R>(client: &R::Client) -> Result
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
            let rows = support::rows::<R>(client, &[0; 100])?;
            let mut session = hydraulic(PointRootSession::new(pair, &rows, 2))?;
            let requested = support::rows::<R>(client, &[73])?;
            hydraulic(session.start_source_sonic(&requested, SonicSide::Left))?;
            assert_eq!(session.callback_points(), 2);
            assert_eq!(session.point_limit(), 2);
            let before = work.snapshot();
            for _ in 0..2 {
                hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                    atol: 0.0,
                    rtol: 0.0,
                }))?;
            }
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(session.callback_points(), 2);
            assert!(
                session
                    .start_source_sonic(&requested, SonicSide::Left)
                    .is_err()
            );
            let result = hydraulic(session.certify_source_sonic())?;
            let statuses = result.status[0].try_to_vec::<i64>()?;
            for (owner, &status) in statuses.iter().enumerate() {
                assert_eq!(
                    status,
                    if owner == 73 {
                        SourceSonicStatus::PositiveRoot as i64
                    } else {
                        SourceSonicStatus::NotRequested as i64
                    }
                );
            }
            assert_eq!(
                result.domains[0].invocation_owner.try_to_vec::<i64>()?,
                [73]
            );
            assert_eq!(result.domains[0].parent_rows.try_to_vec::<i64>()?, [0]);
            assert_eq!(result.domains[0].iterations, 0);
            let another = support::rows::<R>(client, &[74])?;
            hydraulic(session.start_source_sonic(&another, SonicSide::Left))?;
            assert_eq!(
                session.sonic.status[0].try_to_vec::<i64>()?[74],
                SourceSonicStatus::WorkLimit as i64
            );
            assert_eq!(session.callback_points(), 2);
            Ok(())
        },
    )?;
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, _| {
            let rows = support::rows::<R>(client, &[0; 100])?;
            let mut session = hydraulic(PointRootSession::new(pair, &rows, 1))?;
            hydraulic(session.start_wet(&support::rows::<R>(client, &[2])?))?;
            assert_eq!(session.callback_points(), 1);
            assert_eq!(session.point_limit(), 1);
            assert_eq!(
                session.locator_status().try_to_vec::<i64>()?[2],
                WetRootStatus::Located as i64
            );
            assert_eq!(
                session.domains()[0].invocation_owner.try_to_vec::<i64>()?,
                [2]
            );
            Ok(())
        },
    )
}

fn independent_rounds<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.root.max_iterations = 1;
    config.root.atol = 0.0;
    config.root.rtol = 0.0;
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let mut session = hydraulic(PointRootSession::new(pair, &rows, 14))?;
            for side in [SonicSide::Left, SonicSide::Right] {
                hydraulic(session.start_source_sonic(&rows, side))?;
            }
            assert_eq!(session.callback_points(), 14);
            assert_eq!(work.snapshot().root_iterations, 2);
            assert_eq!(work.snapshot().maximum_root_iterations, 1);
            let before = work.snapshot();
            hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            assert_eq!(session.callback_points(), 14);
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            for request in &session.sonic.requests {
                assert_eq!(request.domains[0].iterations, 1);
            }
            Ok(())
        },
    )
}

fn post_callback_error<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.max_native_query_points = 6;
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 0.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            let error = session
                .start_source_sonic(&rows, SonicSide::Left)
                .err()
                .ok_or("Native work error did not surface")?;
            assert!(matches!(
                error,
                HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                }
            ));
            assert_eq!(work.snapshot().residual_points, 6);
            assert_eq!(session.callback_points(), 6);
            assert!(session.sonic.requests[0].terminal);
            let before = work.snapshot();
            assert!(session.start_source_sonic(&rows, SonicSide::Left).is_err());
            assert!(
                session
                    .refine_source_sonic(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0
                    })
                    .is_err()
            );
            assert!(session.certify_source_sonic().is_err());
            assert_eq!(session.callback_points(), 6);
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

fn accounting_mismatch<R>(client: &R::Client) -> Result
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
            let mut session = hydraulic(PointRootSession::new(pair, &rows, 10))?;
            let before = session.checkpoint();
            hydraulic(work.charge(QueryWork::Residual, 1))?;
            assert!(session.reconcile(before, true).is_err());
            assert_eq!(session.callback_points(), 1);
            assert!(session.terminal);
            let snapshot = work.snapshot();
            assert!(session.start_wet(&rows).is_err());
            assert!(session.start_source_sonic(&rows, SonicSide::Left).is_err());
            assert!(
                session
                    .refine_source_sonic(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0
                    })
                    .is_err()
            );
            assert_eq!(session.callback_points(), 1);
            assert_eq!(work.snapshot(), snapshot);
            Ok(())
        },
    )
}

fn wet_error_seals_certificates<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.max_native_query_points = 3;
    support::with_rectangle::<R, _>(
        client,
        1.0,
        3.0,
        &[[1.0, 1.0, 1.0, 1.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            hydraulic(session.start_source_sonic(&rows, SonicSide::Left))?;
            assert_eq!(
                session.sonic.status[0].try_to_vec::<i64>()?,
                [SourceSonicStatus::SourceContact as i64]
            );
            hydraulic(session.start_wet(&rows))?;
            assert_eq!(session.callback_points(), 3);
            assert!(
                session
                    .refine_wet_depth_in_place(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0
                    })
                    .is_err()
            );
            assert!(session.wet.terminal);
            let before = work.snapshot();
            assert!(session.certify_wet_star().is_err());
            assert!(session.certify_wet_waves().is_err());
            // Independent source-contact state still certifies without another residual or quadrature point.
            let sonic = hydraulic(session.certify_source_sonic())?;
            assert_eq!(sonic.domains.len(), 1);
            assert_eq!(
                frozen::endpoints(&sonic.domains[0].discharge)?,
                [[1.0, 1.0]]
            );
            assert_eq!(session.callback_points(), 3);
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}
