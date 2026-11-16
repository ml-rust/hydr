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
#[path = "chart_bracket_tests.rs"]
mod tests;
