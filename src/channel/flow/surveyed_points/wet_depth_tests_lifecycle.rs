use super::{
    frozen::{self, Result, hydraulic},
    support::{self, DepthClient},
};
use crate::channel::flow::{
    surveyed::error,
    surveyed_points::{
        chart_test_support,
        potential_range::select_expansion,
        wet_depth::refine_wet_depth,
        wet_depth_types::{DepthRefinementState, WetDepthStatus},
        wet_root_types::WetRootStatus,
        wet_roots::locate_wet_roots,
    },
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    deferred_certificates::<R>(client)?;
    pointed_dry_adjacent_depth::<R>(client)?;
    exact_parameter_keeps_depth_uncertainty::<R>(client)?;
    mapped_refinement::<R>(client)?;
    selected_caches_stay_fixed::<R>(client)?;
    mixed_classifications::<R>(client)
}

fn pointed_dry_adjacent_depth<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = crate::channel::flow::surveyed::PrismaticRiemannConfig {
        gravity_m_per_s2: 2.0,
        ..chart_test_support::config()
    };
    support::with_pointed_film::<R, _>(client, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[0])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        let before = work.snapshot();
        let session = hydraulic(refine_wet_depth(pair, located))?;
        assert_eq!(work.snapshot(), before);
        assert_eq!(session.domains().len(), 1);
        let domain = &session.domains()[0];
        assert!(domain.is_deferred());
        assert_eq!(
            domain.status.try_to_vec::<i64>()?,
            [WetDepthStatus::DepthConverged as i64]
        );
        // A=h² and F(A)=8*A^(1/4)-2^-80 give h*=2^-166 at capacity depth4.
        let area = support::area_bounds(session.sources(), &domain.parent_rows, &domain.query)?[0];
        let star = 2.0_f64.powi(-332);
        assert!(area[0] <= star && star <= area[1]);
        let depth = frozen::endpoints(&domain.depth)?[0];
        let normalized = frozen::endpoints(&domain.normalized_depth)?[0];
        assert!(depth[0] <= 2.0_f64.powi(-166) && depth[1] >= 2.0_f64.powi(-166));
        assert!(normalized[0] <= 2.0_f64.powi(-168) && normalized[1] >= 2.0_f64.powi(-168));
        Ok(())
    })
}

fn deferred_certificates<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    let ids = [
        "scalar_rectangle_uniform_rest",
        "retained_vacuum_inside_film",
        "uniform_cap_retained_deficit_dyadic_velocity",
    ];
    chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[0, 1, 2])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        assert_eq!(
            located.status.try_to_vec::<i64>()?,
            [WetRootStatus::Located as i64; 3]
        );
        let original = located
            .domains
            .iter()
            .map(|domain| support::snapshot(pair, domain))
            .collect::<Result<Vec<_>>>()?;
        let points = located.callback_points;
        let evidence_count = located.evidence.len();
        let before = work.snapshot();
        let session = hydraulic(refine_wet_depth(pair, located))?;
        assert_eq!(session.evidence().len(), evidence_count);
        assert_eq!(
            session.tolerance().atol.to_bits(),
            config.root.atol.to_bits()
        );
        assert_eq!(
            session.tolerance().rtol.to_bits(),
            config.root.rtol.to_bits()
        );
        assert_eq!(session.callback_points(), points);
        assert_eq!(work.snapshot(), before);
        assert_eq!(session.domains().len(), original.len());
        let mut film = false;
        let mut cap = false;
        for (domain, expected) in session.domains().iter().zip(original) {
            assert!(domain.is_deferred());
            assert!(
                domain
                    .status
                    .try_to_vec::<i64>()?
                    .iter()
                    .all(|&status| status == WetDepthStatus::DepthConverged as i64)
            );
            assert_eq!(domain.iterations, 0);
            assert_eq!(domain.callback_points, 0);
            let DepthRefinementState::Deferred { original } = &domain.refinement else {
                return Err("Depth-converged locator bracket unexpectedly initialized".into());
            };
            assert_eq!(support::snapshot(session.sources(), original)?, expected);
            let parents = domain.parent_rows.try_to_vec::<i64>()?;
            let areas =
                support::area_bounds(session.sources(), &domain.parent_rows, &domain.query)?;
            let normalized = frozen::endpoints(&domain.normalized_depth)?;
            for (index, parent) in parents.into_iter().enumerate() {
                if parent == 1 {
                    let star = 2.0_f64.powi(-164);
                    assert!(areas[index][0] <= star && star <= areas[index][1]);
                    assert!(areas[index][1] < 1e-40);
                    let depth = star / 4.0;
                    assert!(normalized[index][0] <= depth && depth <= normalized[index][1]);
                    film = true;
                }
                if parent == 2 {
                    assert!(normalized[index][0] <= 1.0 && normalized[index][1] >= 1.0);
                    cap = true;
                }
            }
            hydraulic(error::checked::<R, _>(
                client,
                "retained depth cap identity",
                |a| {
                    let parents = domain.parent_rows.try_to_vec::<i64>()?;
                    if let Some(index) = parents.iter().position(|&parent| parent == 2) {
                        let local = Tensor::from_slice(
                            &[i64::try_from(index).map_err(|_| {
                                error::HydraulicError::Numerical {
                                    context: "Depth test index exceeds I64".into(),
                                }
                            })?],
                            &[1],
                            client.device(),
                        )?;
                        let query = domain.query.select(a, &local)?;
                        let parent = Tensor::from_slice(&[2_i64], &[1], client.device())?;
                        let truth = select_expansion(
                            a,
                            &session.sources().velocity.input().left_area,
                            &parent,
                        )?;
                        assert_eq!(truth.high().item::<f64>()?, 14.0);
                        assert_eq!(truth.low().item::<f64>()?, -2.0_f64.powi(-80));
                        let bands = session.sources().bands.select(a, &parent)?;
                        let boundaries = session.sources().boundaries.select(a, &parent)?;
                        use crate::channel::section::PhysicalEndpoint;
                        for (endpoint, lower) in [
                            (PhysicalEndpoint::Lower, true),
                            (PhysicalEndpoint::Upper, false),
                        ] {
                            let area = query
                                .endpoint(a, &bands, endpoint)?
                                .retained_endpoint_area(a, &bands, &boundaries)?;
                            let difference = a.expansion_sub(&area, &truth)?;
                            let difference = a.expansion_interval(&difference)?;
                            if lower {
                                assert!(difference.lower().item::<f64>()? <= 0.0);
                            } else {
                                assert!(difference.upper().item::<f64>()? >= 0.0);
                            }
                        }
                    }
                    Ok(())
                },
            ))?;
        }
        assert!(film && cap);
        Ok(())
    })
}

fn exact_parameter_keeps_depth_uncertainty<R>(client: &R::Client) -> Result
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
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let session = hydraulic(refine_wet_depth(pair, located))?;
            assert!(session.domains()[0].is_deferred());
            let before = work.snapshot();
            let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            let domain = &session.domains()[0];
            assert_eq!(
                domain.status.try_to_vec::<i64>()?,
                [WetDepthStatus::AccuracyFloor as i64]
            );
            assert!(!domain.is_deferred());
            assert_eq!(domain.query.query.parameter.lower().item::<f64>()?, 0.0);
            assert_eq!(domain.query.query.parameter.upper().item::<f64>()?, 0.0);
            let depth = frozen::endpoints(&domain.normalized_depth)?[0];
            assert!(depth[0] < depth[1]);
            assert!(depth[0] <= 1.0 / 3.0 && depth[1] >= 1.0 / 3.0);
            assert_eq!(domain.iterations, 0);
            assert_eq!(
                work.snapshot().residual_points - before.residual_points,
                domain.callback_points
            );
            let before = work.snapshot();
            let used = session.callback_points();
            let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            assert_eq!(session.callback_points(), used);
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(work.snapshot().root_iterations, before.root_iterations);
            Ok(())
        },
    )
}

fn mapped_refinement<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.root.atol = 0.05;
    config.root.rtol = 0.0;
    chart_test_support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_depth_jump"],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let points = located.callback_points;
            let session = hydraulic(refine_wet_depth(pair, located))?;
            assert_eq!(session.domains().len(), 1);
            let first = session.domains()[0].iterations;
            assert!(first > 0 && !session.domains()[0].is_deferred());
            let before = work.snapshot();
            let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 1e-5,
                rtol: 0.0,
            }))?;
            let second = session.domains()[0].iterations;
            assert!(second > first);
            assert_eq!(
                work.snapshot().residual_points - before.residual_points,
                second - first
            );
            assert_eq!(
                work.snapshot().root_iterations - before.root_iterations,
                second - first
            );
            assert_eq!(work.snapshot().root_solves - before.root_solves, 1);
            let before = work.snapshot();
            let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 1e-10,
                rtol: 0.0,
            }))?;
            let domain = &session.domains()[0];
            assert!(domain.iterations > second);
            assert_eq!(
                work.snapshot().residual_points - before.residual_points,
                domain.iterations - second
            );
            assert_eq!(
                work.snapshot().root_iterations - before.root_iterations,
                domain.iterations - second
            );
            assert_eq!(work.snapshot().root_solves - before.root_solves, 1);
            assert!(domain.iterations <= config.root.max_iterations);
            assert_eq!(
                domain.status.try_to_vec::<i64>()?,
                [WetDepthStatus::DepthConverged as i64]
            );
            assert_eq!(session.callback_points(), points + domain.callback_points);
            assert_eq!(work.snapshot().residual_points, session.callback_points());
            assert_eq!(work.snapshot().root_iterations, domain.iterations);
            assert_eq!(work.snapshot().maximum_root_iterations, domain.iterations);
            let normalized = frozen::endpoints(&domain.normalized_depth)?[0];
            assert!(normalized[1] - normalized[0] <= 1e-10);
            Ok(())
        },
    )
}

fn mixed_classifications<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    let ids = [
        "scalar_rectangle_uniform_rest",
        "scalar_rectangle_vacuum",
        "scalar_rectangle_vacuum_threshold",
        "scalar_three_bands_cap_compression",
    ];
    chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[2, 0, 1, 0, 3])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        let statuses = located.status.try_to_vec::<i64>()?;
        let before = work.snapshot();
        let session = hydraulic(refine_wet_depth(pair, located))?;
        assert_eq!(
            session.invocation_rows().try_to_vec::<i64>()?,
            [2, 0, 1, 0, 3]
        );
        assert_eq!(session.locator_status().try_to_vec::<i64>()?, statuses);
        assert_eq!(work.snapshot(), before);
        let owners = session
            .domains()
            .iter()
            .map(|domain| domain.invocation_owner.try_to_vec::<i64>())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(owners.concat(), [1, 3]);
        Ok(())
    })
}

fn selected_caches_stay_fixed<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let config = chart_test_support::config();
    let ids = ["retained_vacuum_inside_film", "scalar_rectangle_two_shocks"];
    chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[0])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        let fixed = chart_test_support::cache(pair)?;
        assert!(fixed.iter().all(|entry| entry.ready[1] == 0.0));
        let before = work.snapshot();
        let session = hydraulic(refine_wet_depth(pair, located))?;
        assert_eq!(work.snapshot(), before);
        let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        }))?;
        assert_eq!(chart_test_support::cache(session.sources())?, fixed);
        assert!(session.domains().iter().all(|domain| !domain.is_deferred()));
        Ok(())
    })
}
