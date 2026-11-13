use super::{
    frozen::{Result, expansion, hydraulic},
    support::{self, DepthClient},
};
use crate::channel::flow::{
    surveyed::error::{HydraulicError, PrismaticRiemannRejection},
    surveyed_points::{
        chart_test_support, wet_depth::refine_wet_depth, wet_depth_types::WetDepthStatus,
        wet_root_types::WetRootStatus, wet_roots::locate_wet_roots,
    },
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    locator_allowance_stays_narrow::<R>(client)?;
    independent_iteration_ceilings::<R>(client)?;
    callback_work_error_remains_typed::<R>(client)
}

fn locator_allowance_stays_narrow<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        3.0,
        1.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, 1))?;
            assert_eq!(located.point_limit, 1);
            assert_eq!(located.callback_points, 1);
            assert_eq!(
                located.status.try_to_vec::<i64>()?,
                [WetRootStatus::Located as i64]
            );
            let session = hydraulic(refine_wet_depth(pair, located))?;
            assert_eq!(session.point_limit(), 1);
            assert!(session.domains()[0].is_deferred());
            let before = work.snapshot();
            let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            assert_eq!(session.point_limit(), 1);
            assert_eq!(session.callback_points(), 1);
            assert_eq!(
                session.domains()[0].status.try_to_vec::<i64>()?,
                [WetDepthStatus::WorkLimit as i64]
            );
            assert!(session.domains()[0].is_deferred());
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

fn independent_iteration_ceilings<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.root.atol = 0.0;
    config.root.rtol = 0.0;
    config.root.max_iterations = 1;
    let section = chart_test_support::rectangle::<R>(client, 1.0, 4.0)?;
    let owners = support::rows::<R>(client, &[0, 0])?;
    let epsilon = 3.0 * 2.0_f64.powi(-80);
    let fields = [
        [[4.0_f64.to_bits(), 0, 0], [1.0_f64.to_bits(), 0, 0]],
        [[0, 0, 0], [(-2.0_f64).to_bits(), 0, 0]],
        [[1.0_f64.to_bits(), 0, 0], [1.0_f64.to_bits(), 0, 0]],
        [[0, 0, 0], [2.0_f64.to_bits(), (-epsilon).to_bits(), 0]],
    ];
    let inputs = fields
        .iter()
        .map(|field| expansion::<R>(client, field, true))
        .collect::<Result<Vec<_>>>()?;
    chart_test_support::with_inputs(client, section, owners, inputs, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[0, 1])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        assert_eq!(
            located.status.try_to_vec::<i64>()?,
            [WetRootStatus::Located as i64; 2]
        );
        assert_eq!(located.domains.len(), 2);
        let points = located.callback_points;
        let session = hydraulic(refine_wet_depth(pair, located))?;
        assert_eq!(session.domains().len(), 2);
        assert!(
            session
                .domains()
                .iter()
                .all(|domain| domain.iterations == 1)
        );
        assert_eq!(work.snapshot().root_iterations, 2);
        assert_eq!(work.snapshot().maximum_root_iterations, 1);
        for domain in session.domains() {
            let parents = domain.parent_rows.try_to_vec::<i64>()?;
            let areas =
                support::area_bounds(session.sources(), &domain.parent_rows, &domain.query)?;
            for (parent, [lower, upper]) in parents.into_iter().zip(areas) {
                if parent == 1 {
                    // F(A)=4*sqrt(A)-3*2^-80 has the exact root 9*2^-164.
                    let star = 9.0 * 2.0_f64.powi(-164);
                    assert!(lower <= star && star <= upper);
                }
            }
        }
        assert_eq!(
            session.callback_points(),
            points
                + session
                    .domains()
                    .iter()
                    .map(|domain| domain.callback_points)
                    .sum::<usize>()
        );
        let before = work.snapshot();
        let points = session.callback_points();
        let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        }))?;
        assert!(
            session
                .domains()
                .iter()
                .all(|domain| domain.iterations == 1)
        );
        assert_eq!(session.callback_points(), points);
        assert_eq!(work.snapshot().residual_points, before.residual_points);
        assert_eq!(work.snapshot().root_iterations, before.root_iterations);
        assert_eq!(work.snapshot().maximum_root_iterations, 1);
        Ok(())
    })
}

fn callback_work_error_remains_typed<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = crate::channel::flow::surveyed::PrismaticRiemannConfig {
        max_native_query_points: 1,
        ..chart_test_support::config()
    };
    support::with_rectangle::<R, _>(
        client,
        3.0,
        1.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let session = hydraulic(refine_wet_depth(pair, located))?;
            assert_eq!(work.snapshot().residual_points, 1);
            let result = session.refine_depth_to(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            });
            assert!(matches!(
                result,
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                })
            ));
            assert_eq!(work.snapshot().residual_points, 1);
            assert_eq!(work.snapshot().root_iterations, 0);
            Ok(())
        },
    )
}
