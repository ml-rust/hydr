use super::super::super::{
    point_candidate_types::{PointCandidateStatus, PointRegime},
    source_sonic_types::{SonicSide, SourceSonicStatus},
};
use super::{PointRootSession, Result, chart_test_support, hydraulic, support};
use crate::channel::flow::surveyed::error::{HydraulicError, PrismaticRiemannRejection};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    shared_ceiling::<R>(client)?;
    callback_error::<R>(client)?;
    wet_error::<R>(client)
}
fn shared_ceiling<R>(client: &R::Client) -> Result
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
            let mut parents = vec![0_i64; 100];
            parents[99] = 1;
            let parents = support::rows::<R>(client, &parents)?;
            let mut session = hydraulic(PointRootSession::new(pair, &parents, 2))?;
            let first = support::rows::<R>(client, &[73])?;
            hydraulic(session.start_source_sonic(&first, SonicSide::Left))?;
            assert_eq!(session.callback_points(), 2);
            assert_eq!(session.point_limit(), 2);
            let before = work.snapshot();
            let result =
                hydraulic(session.certify_point_candidates(&support::rows::<R>(client, &[74])?))?;
            assert_eq!(
                result.sonic_status[0].try_to_vec::<i64>()?,
                [SourceSonicStatus::WorkLimit as i64]
            );
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [PointCandidateStatus::Pending as i64]
            );
            assert_eq!(session.callback_points(), 2);
            assert_eq!(work.snapshot(), before);
            let again =
                hydraulic(session.certify_point_candidates(&support::rows::<R>(client, &[74])?))?;
            assert_eq!(
                again.status.try_to_vec::<i64>()?,
                [PointCandidateStatus::Pending as i64]
            );
            assert_eq!(session.callback_points(), 2);
            assert_eq!(work.snapshot(), before);
            let dry =
                hydraulic(session.certify_point_candidates(&support::rows::<R>(client, &[99])?))?;
            assert_eq!(
                dry.regime.try_to_vec::<i64>()?,
                [PointRegime::BothDry as i64]
            );
            assert_eq!(
                dry.status.try_to_vec::<i64>()?,
                [PointCandidateStatus::Complete as i64]
            );
            assert_eq!(work.snapshot(), before);
            assert!(session.start_source_sonic(&first, SonicSide::Left).is_err());
            assert_eq!(session.callback_points(), 2);
            Ok(())
        },
    )
}
fn callback_error<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.max_native_query_points = 6;
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
            assert!(matches!(
                session.certify_point_candidates(&owners),
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                })
            ));
            assert_eq!(session.callback_points(), 6);
            assert_eq!(work.snapshot().residual_points, 6);
            assert!(session.sonic.requests[0].terminal);
            let before = work.snapshot();
            let cache = chart_test_support::cache(session.sources())?;
            assert!(session.certify_point_candidates(&owners).is_err());
            assert!(
                session
                    .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                    .is_err()
            );
            assert_eq!(session.callback_points(), 6);
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            Ok(())
        },
    )
}
fn wet_error<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.max_native_query_points = 3;
    support::with_rectangle::<R, _>(
        client,
        1.,
        3.,
        &[[1., 1., 1., 1.]],
        &config,
        |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
            hydraulic(session.start_wet(&owners))?;
            assert_eq!(session.callback_points(), 3);
            assert!(
                session
                    .refine_wet_depth_in_place(CertifiedRootTolerance {
                        atol: 1e-20,
                        rtol: 0.
                    })
                    .is_err()
            );
            assert!(session.wet.terminal);
            let before = work.snapshot();
            let cache = chart_test_support::cache(session.sources())?;
            assert!(session.certify_point_candidates(&owners).is_err());
            assert_eq!(session.callback_points(), 3);
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            Ok(())
        },
    )
}
