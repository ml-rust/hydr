use super::{bracket_chart, support};
use crate::channel::flow::surveyed_points::source_tests::support::{Client, Result, hydraulic};
use numr::{
    dtype::DType,
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms, CertifiedUpperBracketStatus,
    },
};

#[path = "chart_bracket_tests_admission.rs"]
mod checks;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    domains::<R>(client)?;
    work_limits::<R>(client)?;
    checks::run::<R>(client)
}

fn domains<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
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
                &[[0.0, 1.5]],
            )?;
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[0.5], &[1], client.device())?;
            let result = hydraulic(bracket_chart(
                pair,
                &rows,
                &chart,
                &seed,
                config.root.max_points,
            ))?;
            assert_eq!(
                result.result.status().try_to_vec::<i64>()?,
                [CertifiedUpperBracketStatus::NoRoot as i64]
            );
            assert_eq!(result.result.upper().item::<f64>()?, 1.5);
            assert!(result.result.upper_residual().upper.item::<f64>()? < 0.0);
            assert_eq!(result.bracket_owner.numel(), 0);
            assert_eq!(result.bracket_rows.numel(), 0);
            assert_eq!(result.query.query.parameter.lower().numel(), 0);
            assert_eq!(
                work.snapshot().residual_points,
                result.result.callback_points()
            );
            Ok(())
        },
    )?;
    let config = support::config();
    support::uncertain_pair::<R, _>(client, &config, |pair, work| {
        let chart = support::chart(
            pair,
            support::Anchor::Zero,
            [1.0_f64.to_bits(), 0],
            &[[1.0, 1.0]],
        )?;
        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let seed = Tensor::from_slice(&[1.0], &[1], client.device())?;
        let result = hydraulic(bracket_chart(
            pair,
            &rows,
            &chart,
            &seed,
            config.root.max_points,
        ))?;
        assert_eq!(
            result.result.status().try_to_vec::<i64>()?,
            [CertifiedUpperBracketStatus::Unresolved as i64]
        );
        let residual = result.result.lower_residual();
        assert!(residual.lower.item::<f64>()? <= 0.0 && residual.upper.item::<f64>()? >= 0.0);
        assert!(residual.lower.item::<f64>()? < residual.upper.item::<f64>()?);
        assert_eq!(result.bracket_rows.numel(), 0);
        assert_eq!(result.query.query.owners.numel(), 0);
        assert_eq!(
            work.snapshot().residual_points,
            result.result.callback_points()
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
            let rows = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
            let chart = support::select(pair, &full, &rows)?;
            let seed = Tensor::<R>::empty(&[0], DType::F64, client.device())?;
            let result = hydraulic(bracket_chart(pair, &rows, &chart, &seed, 0))?;
            for value in [
                &result.invocation_rows,
                &result.bracket_owner,
                &result.bracket_rows,
                result.result.status(),
                &result.query.query.original_rows,
            ] {
                assert_eq!(value.shape(), &[0]);
                assert_eq!(value.dtype(), DType::I64);
                assert!(value.device().is_same(client.device()));
            }
            for value in [
                result.result.lower(),
                result.result.upper(),
                &result.result.lower_residual().lower,
                &result.result.lower_residual().upper,
                &result.result.upper_residual().lower,
                &result.result.upper_residual().upper,
            ] {
                assert_eq!(value.shape(), &[0]);
                assert_eq!(value.dtype(), DType::F64);
            }
            assert_eq!(result.result.callback_points(), 0);
            assert_eq!(result.result.expansion_rounds(), 0);
            assert_eq!(work.snapshot().bracket_solves, 1);
            assert_eq!(work.snapshot().residual_points, 0);
            Ok(())
        },
    )
}

fn work_limits<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    for (expansions, points) in [(0, 1 << 20), (4096, 2)] {
        let mut config = support::config();
        config.max_bracket_expansions = expansions;
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
                let result = hydraulic(bracket_chart(pair, &rows, &chart, &seed, points))?;
                assert_eq!(
                    result.result.status().try_to_vec::<i64>()?,
                    [CertifiedUpperBracketStatus::WorkLimit as i64]
                );
                assert_eq!(result.result.callback_points(), 2);
                assert_eq!(result.result.expansion_rounds(), 0);
                assert_eq!(result.bracket_rows.numel(), 0);
                assert_eq!(result.query.query.parameter.lower().numel(), 0);
                assert_eq!(work.snapshot().residual_points, 2);
                assert_eq!(work.snapshot().bracket_expansion_rounds, 0);
                Ok(())
            },
        )?;
    }
    Ok(())
}
