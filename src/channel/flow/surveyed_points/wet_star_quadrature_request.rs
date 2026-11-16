use super::{
    paired_sources::PairedSources,
    point_quadrature_capacity::{self, NativeRootUse},
    point_quadrature_progress::{self as progress_rows, Record},
    point_quadrature_types::{PointQuadratureOutcome as Outcome, PointQuadratureProgress},
    root_invocation::RootInvocation,
    wet_depth_selection,
    wet_depth_types::{DepthRefinementState, WetDepthDomain, WetDepthStatus},
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance},
};

/// Advances the demanded rows of one wet domain one Simpson level.
/// A partly demanded deferred domain splits, and its undemanded rows stay deferred.
pub(super) fn advance<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    domain: WetDepthDomain<R>,
    tolerance: &CertifiedRootTolerance,
    demand: &Tensor<R>,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
) -> HydraulicResult<Vec<WetDepthDomain<R>>>
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
        + CertifiedMappedBisectionAlgorithms<R>,
{
    let c = sources.geometry.client();
    let selected = c
        .nonzero(&c.index_select(demand, 0, &domain.invocation_owner)?)?
        .flatten()?;
    if selected.numel() == 0 {
        return Ok(vec![domain]);
    }
    mark(
        c,
        &domain,
        &selected,
        Outcome::Unresolved,
        association,
        progress,
    )?;
    let todo = admitted(sources, &domain, &selected, association, progress)?;
    if todo.numel() == 0 {
        return Ok(vec![domain]);
    }
    let native = match &domain.refinement {
        DepthRefinementState::Deferred { .. } => None,
        DepthRefinementState::Initialized { state } => Some(NativeRootUse {
            limits: &state.limits,
            iterations: domain.iterations,
            callback_points: domain.callback_points,
        }),
    };
    if !point_quadrature_capacity::permits(sources, invocation, native, &domain.query, &todo)? {
        mark(c, &domain, &todo, Outcome::WorkLimit, association, progress)?;
        return Ok(vec![domain]);
    }
    let mut output = Vec::with_capacity(2);
    let (mut domain, rows) = if domain.is_deferred() && todo.numel() != domain.parent_rows.numel() {
        let (selected, complement) = split(c, &domain, &todo)?;
        output.push(complement);
        let rows = c.arange(0., selected.parent_rows.numel() as f64, 1., DType::I64)?;
        (selected, rows)
    } else {
        (domain, todo)
    };
    let old_levels = domain.levels.selected(c, &rows)?;
    let old_lower = c.index_select(domain.query.query.parameter.lower(), 0, &rows)?;
    let old_upper = c.index_select(domain.query.query.parameter.upper(), 0, &rows)?;
    domain.levels = raised(c, &domain, &rows)?;
    let domain = if domain.is_deferred() {
        super::wet_depth_refinement::advance(sources, invocation, domain, tolerance)?
    } else {
        super::wet_depth_refresh::refresh_selected(sources, invocation, domain, tolerance, &rows)?
    };
    let old = old_levels.levels().clone();
    let new = domain.levels.selected(c, &rows)?.levels().clone();
    checked::<R, _>(c, "wet star quadrature committed progress", |a| {
        let owners = c.index_select(&domain.invocation_owner, 0, &rows)?;
        let lower = c.index_select(domain.query.query.parameter.lower(), 0, &rows)?;
        let upper = c.index_select(domain.query.query.parameter.upper(), 0, &rows)?;
        let improved = progress_rows::improved(c, [&old_lower, &old_upper], [&lower, &upper])?;
        let status = c.index_select(&domain.status, 0, &rows)?;
        let outcome = progress_rows::committed(c, &rows, &status)?;
        progress_rows::record(
            a,
            association,
            progress,
            Record {
                owners: &owners,
                old: &old,
                new: &new,
                outcome: &outcome,
                improved: &improved,
            },
        )
    })?;
    output.push(domain);
    Ok(output)
}

/// Local selected rows whose next level stays within every ceiling. Others record `LevelLimit`.
fn admitted<R>(
    sources: &PairedSources<'_, R>,
    domain: &WetDepthDomain<R>,
    selected: &Tensor<R>,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = sources.geometry.client();
    let zero = Tensor::zeros(domain.parent_rows.shape(), DType::F64, c.device())?;
    let selected_mask = c.index_put(
        &zero,
        0,
        selected,
        &Tensor::ones(selected.shape(), DType::F64, c.device())?,
    )?;
    let mut admitted = zero.clone();
    for index in domain.levels.policies() {
        let rows = domain.levels.rows_at(c, *index)?;
        if rows.numel() == 0 {
            continue;
        }
        let mask = c.index_put(
            &zero,
            0,
            &rows,
            &Tensor::ones(rows.shape(), DType::F64, c.device())?,
        )?;
        let mask = c.mul(&mask, &selected_mask)?;
        let rows = c.nonzero(&mask)?.flatten()?;
        if rows.numel() == 0 {
            continue;
        }
        if point_quadrature_capacity::next_level(sources, *index)? {
            admitted = c.maximum(&admitted, &mask)?;
        } else {
            mark(c, domain, &rows, Outcome::LevelLimit, association, progress)?;
        }
    }
    Ok(c.nonzero(&admitted)?.flatten()?)
}

/// Each admitted policy group moves to its next level.
fn raised<R>(
    c: &R::Client,
    domain: &WetDepthDomain<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<super::wet_star_levels::WetStarLevels<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + SortingOps<R> + RuntimeClient<R>,
{
    let local = domain.levels.selected(c, rows)?;
    let mut levels = domain.levels.clone();
    for index in local.policies() {
        let group = local.rows_at(c, *index)?;
        if group.numel() == 0 {
            continue;
        }
        let next = index.checked_add(1).ok_or_else(|| {
            super::point_root_session::invalid("wet star panel level overflows usize")
        })?;
        levels = levels.with_rows_at(c, &c.index_select(rows, 0, &group)?, next)?;
    }
    Ok(levels)
}

/// Splits a deferred domain into its admitted rows and the rest, keeping each row's status.
fn split<R>(
    c: &R::Client,
    domain: &WetDepthDomain<R>,
    todo: &Tensor<R>,
) -> HydraulicResult<(WetDepthDomain<R>, WetDepthDomain<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(c, "wet star quadrature deferred partition", |a| {
        let zero = Tensor::zeros(domain.parent_rows.shape(), a.dtype(), c.device())?;
        let mask = c.index_put(
            &zero,
            0,
            todo,
            &Tensor::ones(todo.shape(), a.dtype(), c.device())?,
        )?;
        let complement = c.nonzero(&c.eq(&mask, &zero)?)?.flatten()?;
        let mut selected =
            wet_depth_selection::deferred(a, domain, todo, WetDepthStatus::Unresolved)?;
        selected.status = c.index_select(&domain.status, 0, todo)?;
        let mut rest =
            wet_depth_selection::deferred(a, domain, &complement, WetDepthStatus::Unresolved)?;
        rest.status = c.index_select(&domain.status, 0, &complement)?;
        Ok((selected, rest))
    })
}

fn mark<R>(
    c: &R::Client,
    domain: &WetDepthDomain<R>,
    rows: &Tensor<R>,
    outcome: Outcome,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let owners = c.index_select(&domain.invocation_owner, 0, rows)?;
    let levels = domain.levels.selected(c, rows)?.levels().clone();
    progress_rows::mark(c, &owners, &levels, outcome, association, progress)
}
