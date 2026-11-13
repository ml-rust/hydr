use super::{super::solve_chart, support};
use crate::channel::flow::surveyed_points::{
    chart_test_support::cache,
    source_tests::support::{Client, Result, hydraulic},
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
                assert!(solve_chart(pair, &parents, &chart, config.root.max_points + 1).is_err());
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
            let roots = hydraulic(solve_chart(pair, &rows, &chart, config.root.max_points))?;
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
                let roots = hydraulic(solve_chart(pair, &rows, &query, config.root.max_points))?;
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
                        solve_chart(pair, &rows, &foreign_chart, config.root.max_points).is_err()
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
