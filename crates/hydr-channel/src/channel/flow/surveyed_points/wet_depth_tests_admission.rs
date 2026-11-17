use super::{
    frozen::{Result, hydraulic},
    support::{self, DepthClient},
};
use crate::channel::flow::surveyed::error::HydraulicError;
use crate::channel::flow::surveyed_points::{
    chart_test_support, wet_depth::refine_wet_depth, wet_roots::locate_wet_roots,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    source_identity_precedes_early_acceptance::<R>(client)?;
    construction_policy_precedes_location::<R>(client)?;
    location_policy_precedes_early_acceptance::<R>(client)?;
    invalid_tolerance_does_not_evaluate::<R>(client)?;
    empty_session_preserves_zero_work::<R>(client)
}

fn construction_policy_precedes_location<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    let changed = crate::channel::flow::surveyed::PrismaticRiemannConfig {
        gravity_m_per_s2: 2.0,
        ..config.clone()
    };
    for panels in [false, true] {
        support::with_rectangle_policy::<R, _>(
            client,
            3.0,
            1.0,
            &[[1.0, 0.0, 1.0, 0.0]],
            &config,
            &changed,
            |pair, replacement, work| {
                let rows = support::rows::<R>(client, &[0])?;
                hydraulic(pair.source_values([&rows, &rows]))?;
                let cache = chart_test_support::cache(pair)?;
                assert!(cache.iter().all(|entry| entry.ready == [1.0]));
                let before = work.snapshot();
                if panels {
                    pair.evaluation.panels += 2;
                } else {
                    pair.evaluation.config = replacement;
                }
                let error = locate_wet_roots(pair, &rows, config.root.max_points)
                    .err()
                    .ok_or("Changed construction policy permits root location")?;
                assert!(
                    matches!(&error, HydraulicError::Numerical { .. }),
                    "{error:?}"
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(pair)?, cache);
                let error = pair
                    .source_values([&rows, &rows])
                    .err()
                    .ok_or("Changed construction policy permits cached source reuse")?;
                assert!(
                    matches!(&error, HydraulicError::Numerical { .. }),
                    "{error:?}"
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(pair)?, cache);
                Ok(())
            },
        )?;
    }
    Ok(())
}

fn source_identity_precedes_early_acceptance<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    let section = chart_test_support::rectangle::<R>(client, 3.0, 1.0)?;
    let located = support::with_section::<R, _>(
        client,
        &section,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, _| {
            let rows = support::rows::<R>(client, &[0])?;
            hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))
        },
    )?;
    support::with_section::<R, _>(
        client,
        &section,
        &[[1.0, 0.0, 1.0, 1.0]],
        &config,
        |other, work| {
            assert!(
                other
                    .select_rows(&located.domains[0].parent_rows, &located.domains[0].query)
                    .is_ok()
            );
            let before = work.snapshot();
            let cache = chart_test_support::cache(other)?;
            assert!(refine_wet_depth(other, located).is_err());
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(other)?, cache);
            Ok(())
        },
    )
}

fn location_policy_precedes_early_acceptance<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    let changed = crate::channel::flow::surveyed::PrismaticRiemannConfig {
        gravity_m_per_s2: 2.0,
        ..config.clone()
    };
    for panels in [false, true] {
        support::with_rectangle_policy::<R, _>(
            client,
            3.0,
            1.0,
            &[[1.0, 0.0, 1.0, 0.0]],
            &config,
            &changed,
            |pair, replacement, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let before = work.snapshot();
                let cache = chart_test_support::cache(pair)?;
                if panels {
                    pair.evaluation.panels += 2;
                } else {
                    pair.evaluation.config = replacement;
                }
                assert!(refine_wet_depth(pair, located).is_err());
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(pair)?, cache);
                Ok(())
            },
        )?;
    }
    Ok(())
}

fn invalid_tolerance_does_not_evaluate<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        4.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            for [atol, rtol] in [
                [-1.0, 0.0],
                [f64::NAN, 0.0],
                [f64::INFINITY, 0.0],
                [0.0, -1.0],
                [0.0, f64::NAN],
                [0.0, f64::INFINITY],
                [2.0 * config.root.atol, config.root.rtol],
                [config.root.atol, 2.0 * config.root.rtol],
            ] {
                let rows = support::rows::<R>(client, &[0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let session = hydraulic(refine_wet_depth(pair, located))?;
                let before = work.snapshot();
                assert!(
                    session
                        .refine_depth_to(CertifiedRootTolerance { atol, rtol })
                        .is_err()
                );
                assert_eq!(work.snapshot(), before);
            }
            Ok(())
        },
    )
}

fn empty_session_preserves_zero_work<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(client, 1.0, 4.0, &[], &config, |pair, work| {
        let rows = support::rows::<R>(client, &[])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, 0))?;
        let session = hydraulic(refine_wet_depth(pair, located))?;
        assert_eq!(session.point_limit(), 0);
        assert_eq!(session.callback_points(), 0);
        assert_eq!(session.invocation_rows().numel(), 0);
        assert_eq!(session.locator_status().numel(), 0);
        assert!(session.domains().is_empty());
        let before = work.snapshot();
        let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        }))?;
        assert_eq!(session.callback_points(), 0);
        assert_eq!(work.snapshot(), before);
        Ok(())
    })
}
