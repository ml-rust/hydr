use std::cell::{Cell, RefCell};

use super::{chart_root_callback, chart_root_derivative, paired_sources::PairedSources};
use crate::channel::{
    flow::surveyed::{
        error::{HydraulicError, HydraulicResult, PrismaticRiemannRejection, checked},
        root_calls::resolve_root,
    },
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
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
    if max_points > config.root.max_points {
        return Err(work_limit(&format!(
            "Chart root max_points={max_points} exceeds config.root.max_points={}. Supply the remaining configured budget",
            config.root.max_points,
        )));
    }
    let admission = parent_rows.numel().checked_mul(2).ok_or_else(|| {
        work_limit(&format!(
            "Chart root admission count 2*rows={} overflows usize. Split the invocation",
            parent_rows.numel(),
        ))
    })?;
    if admission > max_points {
        return Err(work_limit(&format!(
            "Chart root admission needs {admission} points, max_points={max_points}. Supply sufficient remaining points or split the invocation",
        )));
    }

    // Source/query identity and support checks precede every cache mutation and work charge.
    let selected = sources.select_rows(parent_rows, chart)?;
    let c = sources.geometry.client();
    let (chart, derivative) = checked::<R, _>(c, "point chart root admission", |a| {
        let zero = Tensor::zeros(&[], a.dtype(), c.device())?;
        if c.nonzero(&c.le(&selected.velocity.both_wet, &zero)?)?
            .numel()
            != 0
        {
            return Err(unresolved(
                "Chart root requires two wet sources. Classify dry sources before solving",
            ));
        }
        let chart = chart.with_parameter_bounds(a, &selected.bands, &chart.query.parameter)?;
        let scale = a.expansion_interval(&chart.query.scale)?;
        if c.nonzero(&c.le(scale.lower(), &zero)?)?.numel() != 0 {
            return Err(unresolved(
                "Chart root scale lacks a strictly positive complete lower bound. Choose a resolved affine chart",
            ));
        }
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
        |query| chart_root_callback::evaluate(&borrowed, &invocation_rows, &chart, query, &cause),
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

pub(super) fn unresolved(context: &str) -> HydraulicError {
    HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::GeometryUnresolved,
        query: None,
        section_owner: None,
        context: context.into(),
    }
}

fn work_limit(context: &str) -> HydraulicError {
    HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::WorkLimit,
        query: None,
        section_owner: None,
        context: context.into(),
    }
}

#[cfg(test)]
#[path = "chart_root_tests.rs"]
mod tests;
