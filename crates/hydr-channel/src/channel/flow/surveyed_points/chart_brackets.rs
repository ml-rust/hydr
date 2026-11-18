use std::cell::{Cell, RefCell};

use super::{chart_admission, chart_root_callback, paired_sources::PairedSources};
use crate::channel::{
    flow::surveyed::{
        error::{Arithmetic, HydraulicResult, checked},
        root_calls::resolve_root,
    },
    section::{CrossSectionError, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions, CertifiedUpperBracketResult,
        CertifiedUpperBracketStatus,
    },
};

/// Full chart outcomes and compact domains with certified root existence.
pub(super) struct ChartBrackets<R: Runtime<DType = DType>> {
    #[cfg(test)]
    pub(super) invocation_rows: Tensor<R>,
    pub(super) bracket_owner: Tensor<R>,
    pub(super) bracket_rows: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) result: CertifiedUpperBracketResult<R>,
}

/// Expand within one supported chart. Its limit is not necessarily the physical capacity.
pub(super) fn bracket_chart<R>(
    sources: &mut PairedSources<'_, R>,
    parent_rows: &Tensor<R>,
    chart: &SupportedAreaQuery<R>,
    initial_upper: &Tensor<R>,
    max_points: usize,
) -> HydraulicResult<ChartBrackets<R>>
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
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = sources.evaluation.config;
    chart_admission::budget(config, parent_rows.numel(), max_points, "Chart bracket")?;
    let selected = sources.select_rows(parent_rows, chart)?;
    let c = sources.geometry.client();
    let (chart, initial_upper) = checked::<R, _>(c, "point chart bracket admission", |a| {
        let chart = chart_admission::supported(a, &selected, chart)?;
        let initial_upper = seed(a, initial_upper, &chart)?;
        Ok((chart, initial_upper))
    })?;
    let invocation_rows = selected.rows;
    let work = sources.evaluation.work;
    work.begin_bracket()?;
    let cause = Cell::new(None);
    let borrowed = RefCell::new(&mut *sources);
    let result = c.bracket_upper_certified(
        |query| {
            chart_root_callback::evaluate(&borrowed, &invocation_rows, None, &chart, query, &cause)
        },
        chart.query.parameter.lower(),
        &initial_upper,
        chart.query.parameter.upper(),
        &CertifiedUpperBracketOptions {
            max_expansions: config.max_bracket_expansions,
            max_points,
        },
    );
    let result = resolve_root(result, cause.take())?;
    work.finish_bracket(result.expansion_rounds())?;
    let (bracket_owner, bracket_rows, query) =
        checked::<R, _>(c, "point chart certified bracket domains", |a| {
            let bracketed = Tensor::from_slice(
                &[CertifiedUpperBracketStatus::Bracketed as i64],
                &[],
                c.device(),
            )?;
            let bracket_owner = c.nonzero(&c.eq(result.status(), &bracketed)?)?.flatten()?;
            let bracket_rows = c.index_select(&invocation_rows, 0, &bracket_owner)?;
            let bands = selected.bands.select(a, &bracket_owner)?;
            let chart = chart.select(a, &bracket_owner)?;
            let lower = c.index_select(result.lower(), 0, &bracket_owner)?;
            let upper = c.index_select(result.upper(), 0, &bracket_owner)?;
            let parameter = a.bounds(&lower, &upper)?;
            let query = chart.with_parameter_bounds(a, &bands, &parameter)?;
            Ok((bracket_owner, bracket_rows, query))
        })?;
    Ok(ChartBrackets {
        #[cfg(test)]
        invocation_rows,
        bracket_owner,
        bracket_rows,
        query,
        result,
    })
}

fn seed<R>(
    a: &mut Arithmetic<'_, R>,
    initial_upper: &Tensor<R>,
    chart: &SupportedAreaQuery<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let count = chart.query.parameter.lower().numel();
    let c = a.client();
    if initial_upper.shape() != [count]
        || initial_upper.dtype() != a.dtype()
        || !initial_upper.device().is_same(c.device())
    {
        return Err(CrossSectionError::InvalidInput {
            section: None,
            context: format!(
                "Chart bracket seed shape={:?}, dtype={:?}, device={}. Supply shape=[{count}], dtype={:?}, device={}",
                initial_upper.shape(), initial_upper.dtype(), initial_upper.device().id(),
                a.dtype(), c.device().id(),
            ),
        }.into());
    }
    let captured = a.point(initial_upper)?;
    if c.nonzero(&c.lt(captured.lower(), chart.query.parameter.lower())?)?
        .numel()
        != 0
        || c.nonzero(&c.gt(captured.upper(), chart.query.parameter.upper())?)?
            .numel()
            != 0
    {
        return Err(CrossSectionError::InvalidInput {
            section: None,
            context: "Chart bracket seed lies outside its parameter domain. Supply lower <= seed <= upper limit".into(),
        }.into());
    }
    Ok(captured.lower().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::{
        surveyed::error,
        surveyed_points::{
            chart_fixture as support,
            chart_roots::solve_chart,
            potential_increment_fixtures::original_film,
            source_fixture_support::{Client, Result, hydraulic},
        },
    };
    use numr::ops::BinaryOps;
    use solvr::optimize::CertifiedBisectionAlgorithms;

    mod controls {
        use super::{bracket_chart, support};
        use crate::channel::flow::surveyed_points::source_fixture_support::{
            Client, Result, hydraulic,
        };
        use numr::{
            dtype::DType,
            runtime::{Device, Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::{
            integrate::CertifiedSimpsonAlgorithms,
            optimize::{
                CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
                CertifiedUpperBracketStatus,
            },
        };

        mod checks {
            use super::super::{bracket_chart, support};
            use crate::channel::flow::{
                surveyed::error::{HydraulicError, PrismaticRiemannRejection},
                surveyed_points::{
                    chart_fixture::cache,
                    source_fixture_support::{Client, Result, hydraulic},
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
                    CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
                    CertifiedUpperBracketStatus,
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
                                bracket_chart(
                                    pair,
                                    &rows,
                                    &chart,
                                    &invalid,
                                    config.root.max_points
                                )
                                .is_err()
                            );
                            assert_eq!((cache(pair)?, work.snapshot()), before);
                        }
                        for invalid in [
                            Tensor::from_slice(&[-1_i64], &[1], client.device())?,
                            Tensor::from_slice(&[1_i64], &[1], client.device())?,
                            Tensor::from_slice(&[0.0_f64], &[1], client.device())?,
                        ] {
                            assert!(
                                bracket_chart(
                                    pair,
                                    &invalid,
                                    &chart,
                                    &seed,
                                    config.root.max_points
                                )
                                .is_err()
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
                                    bracket_chart(
                                        pair,
                                        &rows,
                                        &query,
                                        &seed,
                                        config.root.max_points
                                    )
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
                        assert!(
                            bracket_chart(pair, &rows, &chart, &seed, config.root.max_points)
                                .is_err()
                        );
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
                        let error = bracket_chart(
                            pair,
                            &rows,
                            &chart,
                            &seed,
                            config.root.max_points,
                        )
                        .err()
                        .ok_or(
                            "Native callback budget unexpectedly accepted two endpoint evaluations",
                        )?;
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
        }

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
                assert!(
                    residual.lower.item::<f64>()? <= 0.0 && residual.upper.item::<f64>()? >= 0.0
                );
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
    }

    fn parameters<R: Runtime<DType = DType>>(bracket: &ChartBrackets<R>) -> Result {
        let owners = bracket.bracket_owner.try_to_vec::<i64>()?;
        let lower = bracket.result.lower().try_to_vec::<f64>()?;
        let upper = bracket.result.upper().try_to_vec::<f64>()?;
        let selected_lower = owners
            .iter()
            .map(|&row| usize::try_from(row).map(|row| lower[row]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let selected_upper = owners
            .iter()
            .map(|&row| usize::try_from(row).map(|row| upper[row]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(
            bracket.query.query.parameter.lower().try_to_vec::<f64>()?,
            selected_lower
        );
        assert_eq!(
            bracket.query.query.parameter.upper().try_to_vec::<f64>()?,
            selected_upper
        );
        Ok(())
    }

    fn contains<R>(pair: &PairedSources<'_, R>, bracket: &ChartBrackets<R>, truth: f64) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        parameters(bracket)?;
        let [lower, upper] = support::endpoint_areas(pair, &bracket.bracket_rows, &bracket.query)?;
        hydraulic(error::checked::<R, _>(
            pair.geometry.client(),
            "retained bracket physical containment",
            |a| {
                assert!(a.expansion_interval(&lower)?.lower().item::<f64>()? <= truth);
                assert!(a.expansion_interval(&upper)?.upper().item::<f64>()? >= truth);
                Ok(())
            },
        ))
    }

    fn matrix<R>(client: &R::Client) -> Result
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
            &[
                "scalar_rectangle_uniform_rest",
                "scalar_triangle_uniform_flow",
                "scalar_rectangle_uniform_rest",
            ],
            &config,
            |pair, work| {
                let full = support::chart(
                    pair,
                    support::Anchor::Source,
                    [1.0_f64.to_bits(), 0],
                    &[[-0.25, 0.25], [0.25, 0.5], [-0.25, 0.25]],
                )?;
                let parents = Tensor::from_slice(&[2_i64, 0, 2, 1], &[4], client.device())?;
                let chart = support::select(pair, &full, &parents)?;
                let seed = Tensor::from_slice(
                    &[9.0, -0.125, 9.0, -0.125, 9.0, -0.125, 9.0, 0.375],
                    &[4, 2],
                    client.device(),
                )?
                .narrow(1, 1, 1)?
                .squeeze(Some(1));
                assert!(seed.offset() > 0 && !seed.is_contiguous());
                let bracket = hydraulic(bracket_chart(
                    pair,
                    &parents,
                    &chart,
                    &seed,
                    config.root.max_points,
                ))?;
                let found = CertifiedUpperBracketStatus::Bracketed as i64;
                let absent = CertifiedUpperBracketStatus::NoRoot as i64;
                assert_eq!(
                    bracket.result.status().try_to_vec::<i64>()?,
                    [found, found, found, absent]
                );
                assert_eq!(bracket.invocation_rows.try_to_vec::<i64>()?, [2, 0, 2, 1]);
                assert_eq!(bracket.bracket_owner.try_to_vec::<i64>()?, [0, 1, 2]);
                assert_eq!(bracket.bracket_rows.try_to_vec::<i64>()?, [2, 0, 2]);
                assert_eq!(
                    bracket.query.query.original_rows.try_to_vec::<i64>()?,
                    [103, 101, 103]
                );
                parameters(&bracket)?;
                assert!(bracket.result.expansion_rounds() > 0);
                assert_eq!(work.snapshot().bracket_solves, 1);
                assert_eq!(
                    work.snapshot().bracket_expansion_rounds,
                    bracket.result.expansion_rounds()
                );
                assert_eq!(
                    work.snapshot().residual_points,
                    bracket.result.callback_points()
                );
                let endpoints = bracket.result.upper().try_to_vec::<f64>()?;
                client.copy_into(&parents, &Tensor::zeros(&[4], DType::I64, client.device())?)?;
                client.copy_into(&seed, &Tensor::zeros(&[4], DType::F64, client.device())?)?;
                assert_eq!(bracket.invocation_rows.try_to_vec::<i64>()?, [2, 0, 2, 1]);
                assert_eq!(bracket.result.upper().try_to_vec::<f64>()?, endpoints);
                Ok(())
            },
        )?;
        stationary::<R>(client)?;
        film::<R>(client)?;
        deficit::<R>(client)?;
        controls::run::<R>(client)
    }

    fn stationary<R>(client: &R::Client) -> Result
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
                    &[[1.0, 3.0]],
                )?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let seed = Tensor::from_slice(&[1.25], &[1], client.device())?;
                let bracket = hydraulic(bracket_chart(
                    pair,
                    &rows,
                    &chart,
                    &seed,
                    config.root.max_points,
                ))?;
                assert_eq!(
                    bracket.result.status().try_to_vec::<i64>()?,
                    [CertifiedUpperBracketStatus::Bracketed as i64]
                );
                assert!(bracket.result.expansion_rounds() > 0);
                contains(pair, &bracket, 2.0)?;
                assert_eq!(
                    work.snapshot().residual_points,
                    bracket.result.callback_points()
                );
                Ok(())
            },
        )
    }

    fn film<R>(client: &R::Client) -> Result
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
            &["retained_vacuum_inside_film"],
            &config,
            |pair, work| {
                let film = &original_film::SONIC_BRACKET;
                let limit = f64::from_bits(film.parameter_bounds[0]);
                let chart = support::chart(
                    pair,
                    support::Anchor::Zero,
                    film.scale_components,
                    &[[0.0, limit]],
                )?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let seed = Tensor::from_slice(&[limit / 32.0], &[1], client.device())?;
                let bracket = hydraulic(bracket_chart(
                    pair,
                    &rows,
                    &chart,
                    &seed,
                    config.root.max_points,
                ))?;
                assert_eq!(
                    bracket.result.status().try_to_vec::<i64>()?,
                    [CertifiedUpperBracketStatus::Bracketed as i64]
                );
                assert!(bracket.result.expansion_rounds() > 0);
                let star = f64::from_bits(original_film::STAR.area_bounds[0]);
                contains(pair, &bracket, star)?;
                let remaining = config
                    .root
                    .max_points
                    .checked_sub(bracket.result.callback_points())
                    .ok_or("Bracket callback count exceeds the configured budget")?;
                let roots = hydraulic(solve_chart(
                    pair,
                    &bracket.bracket_rows,
                    &bracket.query,
                    remaining,
                ))?;
                let [lower, upper] = support::endpoint_areas(pair, &roots.root_rows, &roots.query)?;
                hydraulic(error::checked::<R, _>(
                    client,
                    "film bracket and root physical precision",
                    |a| {
                        let lo = a.expansion_interval(&lower)?.lower().item::<f64>()?;
                        let hi = a.expansion_interval(&upper)?.upper().item::<f64>()?;
                        assert!(lo <= star && star <= hi);
                        let gate = f64::from_bits(0x3ddb7cdfd9d7bdba);
                        assert!((hi - lo) / star <= gate);
                        assert!((star - lo).max(hi - star) / star <= gate);
                        Ok(())
                    },
                ))?;
                assert_eq!(
                    work.snapshot().residual_points,
                    bracket.result.callback_points() + roots.result.evaluated_points()
                );
                Ok(())
            },
        )
    }

    fn deficit<R>(client: &R::Client) -> Result
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
            |pair, _| {
                let deficit = f64::from_bits(0x3af0000000000000);
                let chart = support::chart(
                    pair,
                    support::Anchor::Cap,
                    [1.0_f64.to_bits(), 0],
                    &[[0.0, 2.0 * deficit]],
                )?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let seed = Tensor::from_slice(&[deficit / 4.0], &[1], client.device())?;
                let bracket = hydraulic(bracket_chart(
                    pair,
                    &rows,
                    &chart,
                    &seed,
                    config.root.max_points,
                ))?;
                assert_eq!(
                    bracket.result.status().try_to_vec::<i64>()?,
                    [CertifiedUpperBracketStatus::Bracketed as i64]
                );
                assert!(bracket.result.expansion_rounds() > 0);
                parameters(&bracket)?;
                let [lower, upper] =
                    support::endpoint_areas(pair, &bracket.bracket_rows, &bracket.query)?;
                hydraulic(error::checked::<R, _>(
                    client,
                    "retained deficit bracket physical endpoints",
                    |a| {
                        let high = Tensor::from_slice(&[14.0], &[1], client.device())?;
                        let low = Tensor::from_slice(&[-deficit], &[1], client.device())?;
                        let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                        let truth = a.expansion_bounds(&high, &low, &zero)?;
                        let lo = a.expansion_sub(&lower, &truth)?;
                        let hi = a.expansion_sub(&upper, &truth)?;
                        assert!(a.expansion_interval(&lo)?.lower().item::<f64>()? <= 0.0);
                        assert!(a.expansion_interval(&hi)?.upper().item::<f64>()? >= 0.0);
                        Ok(())
                    },
                ))
            },
        )
    }

    #[test]
    fn configured_cpu_chart_brackets_preserve_domains_and_shared_work() -> Result {
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
    fn actual_cuda_chart_brackets_preserve_domains_and_shared_work() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
