use std::cell::{Cell, RefCell};

use super::{
    chart_admission, chart_root_callback, chart_root_derivative, paired_sources::PairedSources,
};
use crate::channel::{
    flow::surveyed::{
        error::{HydraulicResult, checked},
        root_calls::resolve_root,
    },
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedRootResult},
};

/// Solver outcomes retain invocation identity and only certified root-domain queries.
pub(super) struct ChartRoots<R: Runtime<DType = DType>> {
    pub(super) invocation_rows: Tensor<R>,
    pub(super) root_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) result: CertifiedRootResult<R>,
}

/// Refine one fixed supported affine chart without interpreting hydraulic convergence.
pub(super) fn solve_chart<R>(
    sources: &mut PairedSources<'_, R>,
    parent_rows: &Tensor<R>,
    chart: &SupportedAreaQuery<R>,
    max_points: usize,
) -> HydraulicResult<ChartRoots<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>,
{
    let config = sources.evaluation.config;
    chart_admission::budget(config, parent_rows.numel(), max_points, "Chart root")?;

    // Source/query identity and support checks precede every cache mutation and work charge.
    let selected = sources.select_rows(parent_rows, chart)?;
    let c = sources.geometry.client();
    let (chart, derivative) = checked::<R, _>(c, "point chart root admission", |a| {
        let chart = chart_admission::supported(a, &selected, chart)?;
        let derivative =
            chart_root_derivative::lower(a, &selected.bands, &chart, config.gravity_m_per_s2)?;
        Ok((chart, derivative))
    })?;
    let invocation_rows = selected.rows;
    let work = sources.evaluation.work;
    let mut options = config.root.clone();
    options.atol = 0.0;
    options.rtol = 0.0;
    options.max_points = max_points;
    work.begin_root()?;
    let cause = Cell::new(None);
    let borrowed = RefCell::new(&mut *sources);
    let result = c.bisect_certified(
        |query| {
            chart_root_callback::evaluate(&borrowed, &invocation_rows, None, &chart, query, &cause)
        },
        chart.query.parameter.lower(),
        chart.query.parameter.upper(),
        &derivative,
        &options,
    );
    let result = resolve_root(result, cause.take())?;
    work.finish_root(result.iterations())?;
    let (root_rows, query) = checked::<R, _>(c, "point chart root domains", |a| {
        let root_rows = c.index_select(&invocation_rows, 0, result.root_owner())?;
        let bands = selected.bands.select(a, result.root_owner())?;
        let chart = chart.select(a, result.root_owner())?;
        let parameter = a.bounds(result.root_lower(), result.root_upper())?;
        let query = chart.with_parameter_bounds(a, &bands, &parameter)?;
        Ok((root_rows, query))
    })?;
    Ok(ChartRoots {
        invocation_rows,
        root_rows,
        query,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::error;
    use crate::channel::flow::surveyed_points::{
        potential_increment_fixtures::original_film,
        source_fixture_support::{Client, Result, hydraulic},
    };
    use numr::ops::BinaryOps;
    use solvr::optimize::CertifiedRootStatus;

    mod support {
        use crate::channel::flow::surveyed_points::{
            chart_fixture,
            chart_roots::ChartRoots,
            paired_sources::PairedSources,
            source_fixture_support::{Client, Result},
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::common::arithmetic::CertifiedExpansion;

        pub(super) use chart_fixture::{
            Anchor, chart, config, rectangle, select, uncertain_pair, with_pair,
        };

        pub(super) fn physical_endpoints<R>(
            pair: &PairedSources<'_, R>,
            roots: &ChartRoots<R>,
        ) -> Result<[CertifiedExpansion<R>; 2]>
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>,
        {
            assert_eq!(
                roots.query.query.parameter.lower().try_to_vec::<f64>()?,
                roots.result.root_lower().try_to_vec::<f64>()?
            );
            assert_eq!(
                roots.query.query.parameter.upper().try_to_vec::<f64>()?,
                roots.result.root_upper().try_to_vec::<f64>()?
            );
            chart_fixture::endpoint_areas(pair, &roots.root_rows, &roots.query)
        }
    }

    mod controls {
        use super::{super::solve_chart, support};
        use crate::channel::flow::surveyed_points::{
            chart_fixture::cache,
            source_fixture_support::{Client, Result, hydraulic},
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::{
            integrate::CertifiedSimpsonAlgorithms,
            optimize::{CertifiedBisectionAlgorithms, CertifiedRootStatus},
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            for (max_iterations, max_points) in [(0, 1 << 20), (256, 2)] {
                let mut config = support::config();
                config.root.max_iterations = max_iterations;
                support::with_pair::<R, _>(
                    client,
                    &["scalar_rectangle_uniform_rest"],
                    &config,
                    |pair, work| {
                        let chart = support::chart(
                            pair,
                            support::Anchor::Source,
                            [1.0_f64.to_bits(), 0],
                            &[[-0.25, 0.5]],
                        )?;
                        let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                        let before = work.snapshot();
                        let before_cache = cache(pair)?;
                        assert!(solve_chart(pair, &parents, &chart, 0).is_err());
                        assert!(
                            solve_chart(pair, &parents, &chart, config.root.max_points + 1)
                                .is_err()
                        );
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(cache(pair)?, before_cache);
                        let roots = hydraulic(solve_chart(pair, &parents, &chart, max_points))?;
                        assert_eq!(
                            roots.result.status().try_to_vec::<i64>()?,
                            [CertifiedRootStatus::WorkLimit as i64]
                        );
                        assert_eq!(roots.root_rows.try_to_vec::<i64>()?, [0]);
                        assert_eq!(roots.result.iterations(), 0);
                        assert_eq!(roots.result.evaluated_points(), 2);
                        assert!(
                            roots.result.root_lower().item::<f64>()? <= 0.0
                                && roots.result.root_upper().item::<f64>()? >= 0.0
                        );
                        Ok(())
                    },
                )?;
            }
            stationary::<R>(client)?;
            tolerances::<R>(client)?;
            admission::<R>(client)?;
            cache_reuse::<R>(client)?;
            super::domains::run::<R>(client)?;
            super::scaling::run::<R>(client)
        }

        fn stationary<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let mut config = support::config();
            config.gravity_m_per_s2 = 3.0;
            support::with_pair::<R, _>(
                client,
                &["scalar_rectangle_stationary_shock"],
                &config,
                |pair, work| {
                    let chart = support::chart(
                        pair,
                        support::Anchor::Zero,
                        [1.0_f64.to_bits(), 0],
                        &[[1.5, 3.0]],
                    )?;
                    let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                    let roots =
                        hydraulic(solve_chart(pair, &rows, &chart, config.root.max_points))?;
                    assert_eq!(roots.root_rows.try_to_vec::<i64>()?, [0]);
                    // Frozen paired Arb truth is exactly zero at area2 for these original source bits.
                    assert!(roots.result.root_lower().item::<f64>()? <= 2.0);
                    assert!(roots.result.root_upper().item::<f64>()? >= 2.0);
                    assert_eq!(
                        work.snapshot().residual_points,
                        roots.result.evaluated_points()
                    );
                    Ok(())
                },
            )
        }

        fn tolerances<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let mut baseline = None;
            for tolerance in [0.0, 0.1] {
                let mut config = support::config();
                config.root.atol = tolerance;
                config.root.rtol = tolerance;
                let observed = support::with_pair::<R, _>(
                    client,
                    &["retained_vacuum_inside_film"],
                    &config,
                    |pair, work| {
                        let film = &super::original_film::SONIC_BRACKET;
                        let query = support::chart(
                            pair,
                            support::Anchor::Zero,
                            film.scale_components,
                            &[[0.0, f64::from_bits(film.parameter_bounds[0])]],
                        )?;
                        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                        let roots =
                            hydraulic(solve_chart(pair, &rows, &query, config.root.max_points))?;
                        Ok((
                            roots.result.status().try_to_vec::<i64>()?,
                            roots.result.root_lower().try_to_vec::<f64>()?,
                            roots.result.root_upper().try_to_vec::<f64>()?,
                            roots.result.iterations(),
                            roots.result.evaluated_points(),
                            work.snapshot(),
                        ))
                    },
                )?;
                if let Some(expected) = &baseline {
                    assert_eq!(&observed, expected);
                }
                baseline = Some(observed);
            }
            Ok(())
        }

        fn admission<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let config = support::config();
            support::with_pair::<R, _>(
                client,
                &["scalar_rectangle_uniform_rest"],
                &config,
                |pair, work| {
                    let chart = support::chart(
                        pair,
                        support::Anchor::Source,
                        [1.0_f64.to_bits(), 0],
                        &[[-0.25, 0.5]],
                    )?;
                    let before = (cache(pair)?, work.snapshot());
                    let invalid = [
                        Tensor::from_slice(&[-1_i64], &[1], client.device())?,
                        Tensor::from_slice(&[1_i64], &[1], client.device())?,
                        Tensor::from_slice(&[0_i64], &[], client.device())?,
                        Tensor::from_slice(&[0.0_f64], &[1], client.device())?,
                    ];
                    for rows in invalid {
                        assert!(solve_chart(pair, &rows, &chart, config.root.max_points).is_err());
                        assert_eq!((cache(pair)?, work.snapshot()), before);
                    }
                    support::with_pair::<R, _>(
                        client,
                        &["scalar_rectangle_uniform_rest"],
                        &config,
                        |foreign, _| {
                            let foreign_chart = support::chart(
                                foreign,
                                support::Anchor::Source,
                                [1.0_f64.to_bits(), 0],
                                &[[-0.25, 0.5]],
                            )?;
                            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                            assert!(
                                solve_chart(pair, &rows, &foreign_chart, config.root.max_points)
                                    .is_err()
                            );
                            assert_eq!((cache(pair)?, work.snapshot()), before);
                            Ok(())
                        },
                    )?;
                    Ok(())
                },
            )?;
            support::with_pair::<R, _>(
                client,
                &[
                    "scalar_rectangle_uniform_rest",
                    "scalar_triangle_uniform_flow",
                ],
                &config,
                |pair, work| {
                    let query = support::chart(
                        pair,
                        support::Anchor::Source,
                        [1.0_f64.to_bits(), 0],
                        &[[-0.25, 0.5]; 2],
                    )?;
                    let reversed = Tensor::from_slice(&[1_i64, 0], &[2], client.device())?;
                    let before = (cache(pair)?, work.snapshot());
                    assert!(solve_chart(pair, &reversed, &query, config.root.max_points).is_err());
                    assert_eq!((cache(pair)?, work.snapshot()), before);
                    Ok(())
                },
            )?;
            support::with_pair::<R, _>(
                client,
                &["scalar_rectangle_left_dry_sonic"],
                &config,
                |pair, work| {
                    let query = support::chart(
                        pair,
                        support::Anchor::Zero,
                        [1.0_f64.to_bits(), 0],
                        &[[0.5, 1.5]],
                    )?;
                    let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                    let before = (cache(pair)?, work.snapshot());
                    assert!(solve_chart(pair, &rows, &query, config.root.max_points).is_err());
                    assert_eq!((cache(pair)?, work.snapshot()), before);
                    Ok(())
                },
            )
        }

        fn cache_reuse<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let config = support::config();
            support::with_pair::<R, _>(
                client,
                &["uniform_cap_retained_deficit_dyadic_velocity"],
                &config,
                |pair, work| {
                    let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                    let query = support::chart(
                        pair,
                        support::Anchor::Zero,
                        [1.0_f64.to_bits(), 0],
                        &[[0.0, 1.0]],
                    )?;
                    assert!(cache(pair)?.iter().all(|entry| entry.ready == [0.0]));
                    let mut previous = None;
                    for _ in 0..2 {
                        let before = work.snapshot();
                        let roots = hydraulic(solve_chart(pair, &rows, &query, 2))?;
                        let after = work.snapshot();
                        assert_eq!(
                            roots.result.status().try_to_vec::<i64>()?,
                            [CertifiedRootStatus::NoRoot as i64]
                        );
                        assert_eq!(roots.root_rows.numel(), 0);
                        assert_eq!(
                            after.residual_points - before.residual_points,
                            roots.result.evaluated_points()
                        );
                        let current = cache(pair)?;
                        assert!(current.iter().all(|entry| entry.ready == [1.0]));
                        let values = after.simpson_value_points - before.simpson_value_points;
                        let derivatives =
                            after.simpson_derivative_points - before.simpson_derivative_points;
                        if let Some((ready, cold_values, cold_derivatives)) = &previous {
                            assert_eq!(&current, ready);
                            assert!(values < *cold_values);
                            assert!(derivatives < *cold_derivatives);
                        } else {
                            assert!(values > 0 && derivatives > 0);
                            previous = Some((current, values, derivatives));
                        }
                    }
                    Ok(())
                },
            )
        }
    }

    mod domains {
        use super::{super::solve_chart, support};
        use crate::channel::{
            flow::surveyed_points::source_fixture_support::{Client, Result, endpoints, hydraulic},
            section::PhysicalEndpoint,
        };
        use numr::{
            dtype::DType,
            runtime::{Device, Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::{
            integrate::CertifiedSimpsonAlgorithms,
            optimize::{CertifiedBisectionAlgorithms, CertifiedRootStatus},
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let config = support::config();
            support::uncertain_pair::<R, _>(client, &config, |pair, work| {
                let query = support::chart(
                    pair,
                    support::Anchor::Zero,
                    [1.0_f64.to_bits(), 0],
                    &[[1.0, 1.0]],
                )?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let residual =
                    hydraulic(pair.endpoint_on_rows(&rows, &query, PhysicalEndpoint::Lower))?;
                assert_eq!(residual.available.try_to_vec::<f64>()?, [1.0]);
                assert!(residual.retained.abs_error().item::<f64>()? > 0.0);
                let physical = endpoints(&residual.physical)?;
                assert_eq!(physical.len(), 1);
                assert!(
                    physical[0][0] <= 0.0
                        && physical[0][1] >= 0.0
                        && physical[0][0] < physical[0][1]
                );
                let before = work.snapshot();
                let roots = hydraulic(solve_chart(pair, &rows, &query, config.root.max_points))?;
                assert_eq!(roots.invocation_rows.try_to_vec::<i64>()?, [0]);
                assert_eq!(
                    roots.result.status().try_to_vec::<i64>()?,
                    [CertifiedRootStatus::UnresolvedBracket as i64]
                );
                for value in [
                    &roots.root_rows,
                    roots.result.root_owner(),
                    &roots.query.query.owners,
                    &roots.query.query.original_rows,
                ] {
                    assert_eq!(value.shape(), &[0]);
                    assert_eq!(value.dtype(), DType::I64);
                    assert!(value.device().is_same(client.device()));
                }
                for value in [
                    roots.result.root_lower(),
                    roots.result.root_upper(),
                    roots.query.query.parameter.lower(),
                    roots.query.query.parameter.upper(),
                ] {
                    assert_eq!(value.shape(), &[0]);
                    assert_eq!(value.dtype(), DType::F64);
                    assert!(value.device().is_same(client.device()));
                }
                assert_eq!(
                    work.snapshot().residual_points - before.residual_points,
                    roots.result.evaluated_points()
                );
                Ok(())
            })?;
            support::with_pair::<R, _>(
                client,
                &["scalar_rectangle_uniform_rest"],
                &config,
                |pair, work| {
                    let full = support::chart(
                        pair,
                        support::Anchor::Source,
                        [1.0_f64.to_bits(), 0],
                        &[[-0.25, 0.5]],
                    )?;
                    let empty = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
                    let query = support::select(pair, &full, &empty)?;
                    let before = work.snapshot();
                    let roots = hydraulic(solve_chart(pair, &empty, &query, 0))?;
                    for value in [
                        &roots.invocation_rows,
                        &roots.root_rows,
                        roots.result.status(),
                        roots.result.root_owner(),
                        &roots.query.query.owners,
                        &roots.query.query.original_rows,
                    ] {
                        assert_eq!(value.shape(), &[0]);
                        assert_eq!(value.dtype(), DType::I64);
                        assert!(value.device().is_same(client.device()));
                    }
                    for value in [
                        roots.result.root_lower(),
                        roots.result.root_upper(),
                        roots.query.query.parameter.lower(),
                        roots.query.query.parameter.upper(),
                        roots.query.query.scale.high(),
                        roots.query.query.scale.low(),
                        roots.query.query.scale.abs_error(),
                    ] {
                        assert_eq!(value.shape(), &[0]);
                        assert_eq!(value.dtype(), DType::F64);
                        assert!(value.device().is_same(client.device()));
                    }
                    assert_eq!(roots.result.iterations(), 0);
                    assert_eq!(roots.result.evaluated_points(), 0);
                    let after = work.snapshot();
                    assert_eq!(after.root_solves, before.root_solves + 1);
                    assert_eq!(after.residual_points, before.residual_points);
                    assert_eq!(after.simpson_value_points, before.simpson_value_points);
                    assert_eq!(
                        after.simpson_derivative_points,
                        before.simpson_derivative_points
                    );
                    Ok(())
                },
            )
        }
    }

    mod scaling {
        use crate::channel::flow::surveyed_points::source_fixture_support::{
            Client, Result, hydraulic,
        };
        use crate::channel::{
            flow::surveyed::error,
            section::{AreaAnchor, AreaDirection, AreaQuery, HydraulicGeometry},
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>,
        {
            for (width_exponent, depth_exponent, area_exponent, gravity_exponent, bound_exponent) in
                [(500, 250, 750, -1074, -411), (-500, -250, -750, 1000, 376)]
            {
                let width = 2.0_f64.powi(width_exponent);
                let depth = 2.0_f64.powi(depth_exponent);
                let area = 2.0_f64.powi(area_exponent);
                let gravity = if gravity_exponent == -1074 {
                    f64::from_bits(1)
                } else {
                    2.0_f64.powi(gravity_exponent)
                };
                let context = format!(
                    "Derivative profile width=2^{width_exponent}, cap_depth=2^{depth_exponent}, scale=2^{area_exponent}, gravity=2^{gravity_exponent}"
                );
                let section = super::support::rectangle::<R>(client, width, depth)
                    .map_err(|error| format!("{context}: {error}"))?;
                let geometry = HydraulicGeometry::from_batch(&section);
                hydraulic(error::checked::<R, _>(
                    client,
                    "chart-scaled derivative without raw range loss",
                    |a| {
                        let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                        let bands = geometry.gather(a, &owners)?;
                        let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                        let half = Tensor::full_scalar(&[1], DType::F64, 0.5, client.device())?;
                        let scale = a.expansion_point(&Tensor::from_slice(
                            &[area],
                            &[1],
                            client.device(),
                        )?)?;
                        // Interior support avoids requiring a zero cap distance from independent certificates.
                        let parameter = a.bounds(&zero, &half)?;
                        let query = AreaQuery::band_affine(
                            a,
                            &bands,
                            AreaAnchor::Zero,
                            AreaDirection::Add,
                            &scale,
                            &parameter,
                        )?
                        .prove_support(a, &bands)?;
                        let lower =
                            crate::channel::flow::surveyed_points::chart_root_derivative::lower(
                                a, &bands, &query, gravity,
                            )?;
                        let lower = lower.item::<f64>()?;
                        assert!(
                            lower.is_finite()
                                && lower > 0.0
                                && lower <= 2.0_f64.powi(bound_exponent)
                        );
                        Ok(())
                    },
                ))
                .map_err(|error| format!("{context}: {error}"))?;
            }
            Ok(())
        }
    }

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
    {
        let config = support::config();
        let ids = [
            "scalar_rectangle_uniform_rest",
            "scalar_triangle_uniform_flow",
            "scalar_rectangle_uniform_rest",
        ];
        support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
            let chart = support::chart(
                pair,
                support::Anchor::Source,
                [1.0_f64.to_bits(), 0],
                &[[-0.25, 0.25], [0.25, 0.5], [-0.25, 0.25]],
            )?;
            let parent_values = [2_i64, 0, 2, 1];
            let parents = Tensor::from_slice(&parent_values, &[4], client.device())?;
            let chart = support::select(pair, &chart, &parents)?;
            let result = hydraulic(solve_chart(pair, &parents, &chart, config.root.max_points))?;
            assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
            let statuses = result.result.status().try_to_vec::<i64>()?;
            assert_eq!(statuses.len(), 4);
            assert_eq!(statuses[3], CertifiedRootStatus::NoRoot as i64);
            assert!(statuses[..3].iter().all(|&code| matches!(
                CertifiedRootStatus::from_code(code),
                Some(
                    CertifiedRootStatus::Converged
                        | CertifiedRootStatus::AccuracyFloor
                        | CertifiedRootStatus::WorkLimit
                )
            )));
            assert_eq!(result.root_rows.try_to_vec::<i64>()?, [2, 0, 2]);
            assert_eq!(result.result.root_owner().try_to_vec::<i64>()?, [0, 1, 2]);
            assert_eq!(
                result.query.query.original_rows.try_to_vec::<i64>()?,
                [103, 101, 103]
            );
            let expected_owner = pair.sources[0].owners().try_to_vec::<i64>()?[0];
            assert_eq!(
                result.query.query.owners.try_to_vec::<i64>()?,
                [expected_owner; 3]
            );
            let lower = result.result.root_lower().try_to_vec::<f64>()?;
            let upper = result.result.root_upper().try_to_vec::<f64>()?;
            for (&lo, &hi) in lower.iter().zip(&upper) {
                assert!(lo <= 0.0 && hi >= 0.0);
            }
            assert_eq!(work.snapshot().root_solves, 1);
            assert_eq!(work.snapshot().root_iterations, result.result.iterations());
            assert_eq!(
                work.snapshot().residual_points,
                result.result.evaluated_points()
            );
            client.copy_into(&parents, &Tensor::zeros(&[4], DType::I64, client.device())?)?;
            assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
            assert_eq!(result.root_rows.try_to_vec::<i64>()?, [2, 0, 2]);
            assert_eq!(
                result.query.query.original_rows.try_to_vec::<i64>()?,
                [103, 101, 103]
            );
            Ok(())
        })?;
        support::with_pair::<R, _>(
            client,
            &["retained_vacuum_inside_film"],
            &config,
            |pair, work| {
                let scale = original_film::SONIC_BRACKET.scale_components;
                let upper = f64::from_bits(original_film::SONIC_BRACKET.parameter_bounds[0]);
                let chart = support::chart(pair, support::Anchor::Zero, scale, &[[0.0, upper]])?;
                let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let roots = hydraulic(solve_chart(pair, &parents, &chart, config.root.max_points))?;
                assert_eq!(roots.root_rows.try_to_vec::<i64>()?, [0]);
                let star = f64::from_bits(original_film::STAR.area_bounds[0]);
                assert_eq!(
                    original_film::STAR.area_bounds[0],
                    original_film::STAR.area_bounds[1]
                );
                let [lower, upper] = support::physical_endpoints(pair, &roots)?;
                let (lo, hi) = hydraulic(error::checked::<R, _>(
                    client,
                    "returned film root physical endpoints",
                    |a| {
                        Ok((
                            a.expansion_interval(&lower)?.lower().item::<f64>()?,
                            a.expansion_interval(&upper)?.upper().item::<f64>()?,
                        ))
                    },
                ))?;
                assert!(
                    lo <= star && star <= hi,
                    "film root [{lo:e}, {hi:e}] excludes {star:e}"
                );
                let relative_gate = f64::from_bits(0x3ddb7cdfd9d7bdba);
                assert!(
                    (hi - lo) / star <= relative_gate,
                    "film width: [{lo:e}, {hi:e}]"
                );
                assert!((star - lo).max(hi - star) / star <= relative_gate);
                assert_eq!(
                    work.snapshot().residual_points,
                    roots.result.evaluated_points()
                );
                Ok(())
            },
        )?;
        support::with_pair::<R, _>(
            client,
            &["uniform_cap_retained_deficit_dyadic_velocity"],
            &config,
            |pair, _| {
                let deficit = f64::from_bits(0x3af0000000000000);
                let chart = support::chart(
                    pair,
                    support::Anchor::Cap,
                    [1.0_f64.to_bits(), 0],
                    &[[0.0, 2.0 * deficit]],
                )?;
                let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let roots = hydraulic(solve_chart(pair, &parents, &chart, config.root.max_points))?;
                assert_eq!(roots.root_rows.try_to_vec::<i64>()?, [0]);
                assert!(roots.result.root_lower().item::<f64>()? <= deficit);
                assert!(roots.result.root_upper().item::<f64>()? >= deficit);
                let [lower, upper] = support::physical_endpoints(pair, &roots)?;
                hydraulic(error::checked::<R, _>(
                    client,
                    "returned deficit root retained physical endpoints",
                    |a| {
                        let high = Tensor::from_slice(&[14.0], &[1], client.device())?;
                        let low = Tensor::from_slice(&[-deficit], &[1], client.device())?;
                        let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                        let truth = a.expansion_bounds(&high, &low, &zero)?;
                        let lower_difference = a.expansion_sub(&lower, &truth)?;
                        let upper_difference = a.expansion_sub(&upper, &truth)?;
                        assert!(
                            a.expansion_interval(&lower_difference)?
                                .lower()
                                .item::<f64>()?
                                <= 0.0
                        );
                        assert!(
                            a.expansion_interval(&upper_difference)?
                                .upper()
                                .item::<f64>()?
                                >= 0.0
                        );
                        Ok(())
                    },
                ))?;
                Ok(())
            },
        )?;
        controls::run::<R>(client)
    }

    #[test]
    fn configured_cpu_chart_roots_preserve_certificates_and_film_precision() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let base = CpuClient::new(CpuDevice::new());
        let before = base.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = base.with_parallelism(config);
        matrix::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(base.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_chart_roots_preserve_certificates_and_film_precision() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
