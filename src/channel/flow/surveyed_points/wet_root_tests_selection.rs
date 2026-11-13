use crate::channel::flow::surveyed_points::{
    chart_test_support::{self, cache},
    source_tests::support::{Client, Result, hydraulic},
    wet_root_types::WetRootStatus,
    wet_roots::locate_wet_roots,
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    mapping::<R>(client)?;
    subsets::<R>(client)?;
    budgets::<R>(client)?;
    multi_chart_budget::<R>(client)
}

fn mapping<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = chart_test_support::config();
    let ids = [
        "retained_vacuum_inside_film",
        "scalar_rectangle_vacuum",
        "scalar_rectangle_vacuum_threshold",
        "scalar_three_bands_cap_compression",
        "scalar_rectangle_uniform_rest",
        "scalar_rectangle_weak_depth_jump",
    ];
    chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let parent_values = [4_i64, 1, 4, 0, 2, 3];
        let storage = Tensor::from_slice(
            &[99_i64, 4, 99, 1, 99, 4, 99, 0, 99, 2, 99, 3],
            &[6, 2],
            client.device(),
        )?;
        let parents = storage.narrow(1, 1, 1)?.squeeze(Some(1));
        let result = hydraulic(locate_wet_roots(pair, &parents, config.root.max_points))?;
        assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [
                WetRootStatus::Located as i64,
                WetRootStatus::Vacuum as i64,
                WetRootStatus::Located as i64,
                WetRootStatus::Located as i64,
                WetRootStatus::TouchingDry as i64,
                WetRootStatus::CapExceeded as i64
            ]
        );
        super::certificates::check(pair, &result)?;
        assert_eq!(work.snapshot().residual_points, result.callback_points());
        let physical = result
            .domains
            .iter()
            .map(|domain| super::support::physical_bounds(pair, &domain.parent_rows, &domain.query))
            .collect::<Result<Vec<_>>>()?;
        client.copy_into(&parents, &Tensor::zeros(&[6], DType::I64, client.device())?)?;
        assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
        super::certificates::check(pair, &result)?;
        for (domain, before) in result.domains.iter().zip(physical) {
            assert_eq!(
                super::support::physical_bounds(pair, &domain.parent_rows, &domain.query)?,
                before
            );
        }
        assert!(cache(pair)?.iter().all(|entry| entry.ready[5] == 0.0));
        Ok(())
    })
}

fn subsets<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = chart_test_support::config();
    let ids = [
        "retained_vacuum_inside_film",
        "scalar_rectangle_depth_jump",
        "scalar_rectangle_two_shocks",
        "scalar_rectangle_uniform_rest",
    ];
    chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let rows = Tensor::from_slice(&[0_i64, 0], &[2], client.device())?;
        let before = cache(pair)?;
        let mut cold_simpson = None;
        for _ in 0..2 {
            let before_work = work.snapshot();
            let result = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [WetRootStatus::Located as i64; 2]
            );
            super::certificates::check(pair, &result)?;
            let after = work.snapshot();
            assert_eq!(
                after.residual_points - before_work.residual_points,
                result.callback_points()
            );
            assert!(result.callback_points() >= 2);
            let current = cache(pair)?;
            for (entry, original) in current.iter().zip(&before) {
                assert_eq!(entry.ready[1..], original.ready[1..]);
            }
            let simpson = after.simpson_value_points - before_work.simpson_value_points;
            if let Some(cold) = cold_simpson {
                assert!(simpson <= cold);
            } else {
                cold_simpson = Some(simpson);
            }
        }
        Ok(())
    })
}

fn budgets<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = chart_test_support::config();
    let ids = [
        "scalar_rectangle_uniform_rest",
        "scalar_rectangle_vacuum",
        "retained_vacuum_inside_film",
    ];
    for budget in [1, 2, 3] {
        chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
            let rows = Tensor::from_slice(&[0_i64, 1, 2], &[3], client.device())?;
            let result = hydraulic(locate_wet_roots(pair, &rows, budget))?;
            let expected = [
                WetRootStatus::Located as i64,
                if budget == 3 {
                    WetRootStatus::Vacuum as i64
                } else {
                    WetRootStatus::WorkLimit as i64
                },
                WetRootStatus::WorkLimit as i64,
            ];
            assert_eq!(result.status.try_to_vec::<i64>()?, expected);
            assert_eq!(result.callback_points(), if budget == 3 { 3 } else { 1 });
            assert_eq!(work.snapshot().residual_points, result.callback_points());
            super::certificates::check(pair, &result)
        })?;
    }
    chart_test_support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_uniform_rest"],
        &config,
        |pair, work| {
            let rows = Tensor::from_slice(&[0_i64, 0, 0], &[3], client.device())?;
            let result = hydraulic(locate_wet_roots(pair, &rows, 2))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [WetRootStatus::WorkLimit as i64; 3]
            );
            assert_eq!(result.callback_points(), 0);
            assert_eq!(work.snapshot().residual_points, 0);
            assert!(result.domains.is_empty() && result.evidence.is_empty());
            Ok(())
        },
    )
}

fn multi_chart_budget<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = chart_test_support::config();
    for budget in [4, config.root.max_points] {
        chart_test_support::with_pair::<R, _>(
            client,
            &["retained_vacuum_inside_film"],
            &config,
            |pair, work| {
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let result = hydraulic(locate_wet_roots(pair, &rows, budget))?;
                let counts = work.snapshot();
                assert_eq!(result.callback_points(), counts.residual_points);
                assert!(result.callback_points() <= budget);
                assert!(counts.bracket_solves > 0);
                if budget == 4 {
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [WetRootStatus::WorkLimit as i64]
                    );
                    assert!(result.domains.is_empty());
                } else {
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [WetRootStatus::Located as i64]
                    );
                    assert!(counts.bracket_solves >= 2);
                    super::certificates::check(pair, &result)?;
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}
