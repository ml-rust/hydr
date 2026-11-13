use super::{
    Result, chart_test_support, frozen, hydraulic, locate_wet_roots, refine_wet_depth, support,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    witness_precedes_work::<R>(client)?;
    identities_and_work::<R>(client)?;
    original_budget::<R>(client)?;
    empty::<R>(client)
}

fn witness_precedes_work<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let foreign = support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[1.0, 1.0, 1.0, 1.0]],
        &config,
        |pair, _| hydraulic(pair.root_witness()),
    )?;
    for stale in [false, true] {
        support::with_rectangle::<R, _>(
            client,
            1.0,
            8.0,
            &[[1.0, 1.0, 1.0, 1.0]],
            &config,
            |pair, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                let before = work.snapshot();
                let cache = chart_test_support::cache(session.sources())?;
                if stale {
                    session.sources.evaluation.panels *= 2;
                } else {
                    session.witness = foreign.clone();
                }
                assert!(session.certify_wet_waves().is_err());
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(session.sources())?, cache);
                Ok(())
            },
        )?;
    }
    Ok(())
}

fn identities_and_work<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.root.atol = 1e-4;
    config.root.rtol = 0.0;
    chart_test_support::with_pair::<R, _>(
        client,
        &[
            "scalar_rectangle_depth_jump",
            "retained_vacuum_inside_film",
            "uniform_cap_retained_deficit_dyadic_velocity",
        ],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[2, 0, 2, 1])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            let locator_status = session.locator_status().try_to_vec::<i64>()?;
            let points = session.callback_points();
            let limit = session.point_limit();
            let identities = session
                .domains()
                .iter()
                .map(|d| {
                    Ok((
                        d.invocation_owner.try_to_vec::<i64>()?,
                        d.parent_rows.try_to_vec::<i64>()?,
                        d.query.query.original_rows.try_to_vec::<i64>()?,
                        frozen::endpoints(&d.query.query.parameter)?,
                        d.iterations,
                        d.callback_points,
                        d.mapped_points,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            // Warm source caches, then measure exactly one star evaluation as the hydraulic-work baseline.
            hydraulic(session.certify_wet_star())?;
            let before_star = work.snapshot();
            hydraulic(session.certify_wet_star())?;
            let after_star = work.snapshot();
            let waves = hydraulic(session.certify_wet_waves())?;
            let after_wave = work.snapshot();
            assert_eq!(
                after_wave.simpson_value_points - after_star.simpson_value_points,
                after_star.simpson_value_points - before_star.simpson_value_points
            );
            assert_eq!(
                after_wave.simpson_derivative_points - after_star.simpson_derivative_points,
                after_star.simpson_derivative_points - before_star.simpson_derivative_points
            );
            assert_eq!(after_wave.residual_points, after_star.residual_points);
            assert_eq!(after_wave.root_iterations, after_star.root_iterations);
            assert_eq!(after_wave.root_solves, after_star.root_solves);
            assert_eq!(session.callback_points(), points);
            assert_eq!(session.point_limit(), limit);
            assert_eq!(
                session.locator_status().try_to_vec::<i64>()?,
                locator_status
            );
            assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [2, 0, 2, 1]);
            for ((wave, domain), identity) in waves.iter().zip(session.domains()).zip(identities) {
                assert_eq!(wave.star.invocation_owner.try_to_vec::<i64>()?, identity.0);
                assert_eq!(wave.star.parent_rows.try_to_vec::<i64>()?, identity.1);
                assert_eq!(
                    wave.star.query.query.original_rows.try_to_vec::<i64>()?,
                    identity.2
                );
                assert_eq!(
                    frozen::endpoints(&wave.star.query.query.parameter)?,
                    identity.3
                );
                assert_eq!(
                    (
                        domain.iterations,
                        domain.callback_points,
                        domain.mapped_points
                    ),
                    (identity.4, identity.5, identity.6)
                );
            }
            let cache = chart_test_support::cache(session.sources())?;
            let repeated = hydraulic(session.certify_wet_waves())?;
            assert_eq!(repeated.len(), waves.len());
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 1e-7,
                rtol: 0.0,
            }))?;
            assert!(session.callback_points() >= points);
            assert_eq!(session.point_limit(), limit);
            Ok(())
        },
    )
}

fn original_budget<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.max_native_query_points = 1;
    support::with_rectangle::<R, _>(
        client,
        3.0,
        1.0,
        &[[1.0, 1.0, 1.0, 1.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, 1))?;
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            let before = work.snapshot();
            hydraulic(session.certify_wet_waves())?;
            hydraulic(session.certify_wet_waves())?;
            assert_eq!(work.snapshot(), before);
            assert_eq!(session.point_limit(), 1);
            assert_eq!(session.callback_points(), 1);
            // Explicit exact-floor coverage keeps the exhausted original point ceiling.
            let mut session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            hydraulic(session.certify_wet_waves())?;
            assert_eq!(session.callback_points(), 1);
            assert_eq!(session.point_limit(), 1);
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

fn empty<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(client, 1.0, 8.0, &[], &config, |pair, work| {
        let rows = support::rows::<R>(client, &[])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, 0))?;
        let mut session = hydraulic(refine_wet_depth(pair, located))?;
        let before = work.snapshot();
        assert!(hydraulic(session.certify_wet_waves())?.is_empty());
        assert_eq!(work.snapshot(), before);
        Ok(())
    })
}
