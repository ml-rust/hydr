use super::super::{bracket_chart, support};
use crate::channel::flow::{
    surveyed::error::{HydraulicError, PrismaticRiemannRejection},
    surveyed_points::{
        chart_test_support::cache,
        source_tests::support::{Client, Result, hydraulic},
    },
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms, CertifiedUpperBracketStatus,
    },
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    admission::<R>(client)?;
    cache_reuse::<R>(client)?;
    callback_rejection::<R>(client)
}

fn admission<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
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
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[-0.125], &[1], client.device())?;
            let before = (cache(pair)?, work.snapshot());
            for points in [0, config.root.max_points + 1] {
                assert!(bracket_chart(pair, &rows, &chart, &seed, points).is_err());
                assert_eq!((cache(pair)?, work.snapshot()), before);
            }
            for invalid in [
                Tensor::from_slice(&[-0.5_f64], &[1], client.device())?,
                Tensor::from_slice(&[1.0_f64], &[1], client.device())?,
                Tensor::from_slice(&[f64::NAN], &[1], client.device())?,
                Tensor::from_slice(&[f64::INFINITY], &[1], client.device())?,
                Tensor::from_slice(&[0_i64], &[1], client.device())?,
                Tensor::from_slice(&[0.0_f64], &[], client.device())?,
                Tensor::from_slice(&[0.0_f64, 0.0], &[2], client.device())?,
            ] {
                assert!(
                    bracket_chart(pair, &rows, &chart, &invalid, config.root.max_points).is_err()
                );
                assert_eq!((cache(pair)?, work.snapshot()), before);
            }
            for invalid in [
                Tensor::from_slice(&[-1_i64], &[1], client.device())?,
                Tensor::from_slice(&[1_i64], &[1], client.device())?,
                Tensor::from_slice(&[0.0_f64], &[1], client.device())?,
            ] {
                assert!(
                    bracket_chart(pair, &invalid, &chart, &seed, config.root.max_points).is_err()
                );
                assert_eq!((cache(pair)?, work.snapshot()), before);
            }
            support::with_pair::<R, _>(
                client,
                &["scalar_rectangle_uniform_rest"],
                &config,
                |foreign, _| {
                    let query = support::chart(
                        foreign,
                        support::Anchor::Source,
                        [1.0_f64.to_bits(), 0],
                        &[[-0.25, 0.5]],
                    )?;
                    assert!(
                        bracket_chart(pair, &rows, &query, &seed, config.root.max_points).is_err()
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
        &["scalar_rectangle_left_dry_sonic"],
        &config,
        |pair, work| {
            let chart = support::chart(
                pair,
                support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[0.5, 1.5]],
            )?;
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[1.0], &[1], client.device())?;
            let before = (cache(pair)?, work.snapshot());
            assert!(bracket_chart(pair, &rows, &chart, &seed, config.root.max_points).is_err());
            assert_eq!((cache(pair)?, work.snapshot()), before);
            Ok(())
        },
    )
}

fn cache_reuse<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = support::config();
    support::with_pair::<R, _>(
        client,
        &["uniform_cap_retained_deficit_dyadic_velocity"],
        &config,
        |pair, work| {
            let chart = support::chart(
                pair,
                support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[0.0, 1.0]],
            )?;
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[1.0], &[1], client.device())?;
            assert!(cache(pair)?.iter().all(|entry| entry.ready == [0.0]));
            let mut cold = None;
            for _ in 0..2 {
                let before = work.snapshot();
                let result = hydraulic(bracket_chart(pair, &rows, &chart, &seed, 2))?;
                let after = work.snapshot();
                assert_eq!(
                    result.result.status().try_to_vec::<i64>()?,
                    [CertifiedUpperBracketStatus::NoRoot as i64]
                );
                assert_eq!(
                    after.residual_points - before.residual_points,
                    result.result.callback_points()
                );
                let values = after.simpson_value_points - before.simpson_value_points;
                let derivatives =
                    after.simpson_derivative_points - before.simpson_derivative_points;
                let ready = cache(pair)?;
                assert!(ready.iter().all(|entry| entry.ready == [1.0]));
                if let Some((expected, old_values, old_derivatives)) = &cold {
                    assert_eq!(&ready, expected);
                    assert!(values < *old_values && derivatives < *old_derivatives);
                } else {
                    assert!(values > 0 && derivatives > 0);
                    cold = Some((ready, values, derivatives));
                }
            }
            Ok(())
        },
    )
}

fn callback_rejection<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let mut config = support::config();
    config.max_native_query_points = 1;
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
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[0.25], &[1], client.device())?;
            let error = bracket_chart(pair, &rows, &chart, &seed, config.root.max_points)
                .err()
                .ok_or("Native callback budget unexpectedly accepted two endpoint evaluations")?;
            assert!(matches!(
                error,
                HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                }
            ));
            assert_eq!(work.snapshot().bracket_solves, 1);
            assert_eq!(work.snapshot().residual_points, 1);
            assert_eq!(work.snapshot().bracket_expansion_rounds, 0);
            Ok(())
        },
    )
}
