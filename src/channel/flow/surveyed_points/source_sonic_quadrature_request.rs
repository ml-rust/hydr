use super::{
    paired_sources::PairedSources,
    point_quadrature_capacity::{self, NativeRootUse},
    point_quadrature_progress::{self as progress_rows, Record},
    point_quadrature_types::{PointQuadratureOutcome as Outcome, PointQuadratureProgress},
    root_invocation::RootInvocation,
    source_sonic_numerical::SourceSonicNumericalEvidence,
    source_sonic_types::{SourceSonicDomain, SourceSonicRequest, SourceSonicStatus},
};
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection,
    error::{HydraulicError, HydraulicResult, checked},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedMappedBisectionAlgorithms};

pub(super) fn advance<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    request: &mut SourceSonicRequest<R>,
    demand: &Tensor<R>,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
) -> HydraulicResult<()>
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
    let mut output = Vec::with_capacity(request.domains.len());
    for mut domain in std::mem::take(&mut request.domains) {
        let selected = c
            .nonzero(&c.index_select(demand, 0, &domain.invocation_owner)?)?
            .flatten()?;
        if selected.numel() == 0 {
            output.push(domain);
            continue;
        }
        let initial = if domain.classification == SourceSonicStatus::PositiveRoot {
            Outcome::Unresolved
        } else {
            Outcome::NotApplicable
        };
        mark(sources, &domain, &selected, initial, association, progress)?;
        if domain.classification != SourceSonicStatus::PositiveRoot {
            output.push(domain);
            continue;
        }
        let todo = checked::<R, _>(c, "sonic quadrature authorized next levels", |a| {
            let zero = Tensor::zeros(domain.parent_rows.shape(), a.dtype(), c.device())?;
            let mut admitted = zero.clone();
            let selected_mask = c.index_put(
                &zero,
                0,
                &selected,
                &Tensor::ones(selected.shape(), a.dtype(), c.device())?,
            )?;
            for index in domain.numerical.policies() {
                let level = Tensor::full_scalar(&[], DType::I64, *index as f64, c.device())?;
                let mask = c.where_cond(
                    &c.eq(domain.numerical.levels(), &level)?,
                    &selected_mask,
                    &zero,
                )?;
                let rows = c.nonzero(&mask)?.flatten()?;
                if rows.numel() == 0 {
                    continue;
                }
                if point_quadrature_capacity::next_level(sources, *index)? {
                    admitted = c.maximum(&admitted, &mask)?;
                } else {
                    mark(
                        sources,
                        &domain,
                        &rows,
                        Outcome::LevelLimit,
                        association,
                        progress,
                    )?;
                }
            }
            Ok(c.nonzero(&admitted)?.flatten()?)
        })?;
        if todo.numel() == 0 {
            output.push(domain);
            continue;
        }
        let native = domain.refinement.as_ref().map(|state| NativeRootUse {
            limits: &state.limits,
            iterations: domain.iterations,
            callback_points: domain.callback_points,
        });
        if !point_quadrature_capacity::permits(sources, invocation, native, &domain.query, &todo)? {
            mark(
                sources,
                &domain,
                &todo,
                Outcome::WorkLimit,
                association,
                progress,
            )?;
            output.push(domain);
            continue;
        }
        let Some(numerical) = prime(sources, &domain, &todo)? else {
            mark(
                sources,
                &domain,
                &todo,
                Outcome::WorkLimit,
                association,
                progress,
            )?;
            output.push(domain);
            continue;
        };
        domain.numerical = numerical;
        let (domain, rows) =
            if domain.refinement.is_none() && todo.numel() != domain.parent_rows.numel() {
                let mask = Tensor::zeros(
                    domain.parent_rows.shape(),
                    domain.depth.lower().dtype(),
                    c.device(),
                )?;
                let mask = c.index_put(
                    &mask,
                    0,
                    &todo,
                    &Tensor::ones(todo.shape(), mask.dtype(), c.device())?,
                )?;
                let complement = c
                    .nonzero(&c.eq(
                        &mask,
                        &Tensor::zeros(mask.shape(), mask.dtype(), c.device())?,
                    )?)?
                    .flatten()?;
                output.push(super::source_sonic_selection::deferred(
                    sources,
                    &domain,
                    &complement,
                )?);
                let selected = super::source_sonic_selection::deferred(sources, &domain, &todo)?;
                let rows = c.arange(0., selected.parent_rows.numel() as f64, 1., DType::I64)?;
                (selected, rows)
            } else {
                (domain, todo)
            };
        let old_levels = domain.numerical.levels().clone();
        let old_lower = domain.query.query.parameter.lower().clone();
        let old_upper = domain.query.query.parameter.upper().clone();
        let domain = if domain.refinement.is_some() {
            super::source_sonic_refinement::refresh_selected(
                sources,
                invocation,
                &request.binding,
                domain,
                &request.tolerance,
                &rows,
            )?
        } else {
            super::source_sonic_refinement::advance(
                sources,
                invocation,
                &request.binding,
                domain,
                &request.tolerance,
            )?
        };
        checked::<R, _>(c, "sonic quadrature committed progress", |a| {
            let owners = c.index_select(&domain.invocation_owner, 0, &rows)?;
            let new_levels = c.index_select(domain.numerical.levels(), 0, &rows)?;
            let old_levels = c.sub_scalar(&c.index_select(&old_levels, 0, &rows)?, 1.)?;
            let lower = c.index_select(domain.query.query.parameter.lower(), 0, &rows)?;
            let upper = c.index_select(domain.query.query.parameter.upper(), 0, &rows)?;
            let old_lower = c.index_select(&old_lower, 0, &rows)?;
            let old_upper = c.index_select(&old_upper, 0, &rows)?;
            let improved = progress_rows::improved(c, [&old_lower, &old_upper], [&lower, &upper])?;
            let status = c.index_select(&domain.status, 0, &rows)?;
            let outcome = progress_rows::committed(c, &rows, &status)?;
            progress_rows::record(
                a,
                association,
                progress,
                Record {
                    owners: &owners,
                    old: &old_levels,
                    new: &new_levels,
                    outcome: &outcome,
                    improved: &improved,
                },
            )
        })?;
        output.push(domain);
    }
    request.domains = output;
    Ok(())
}

fn prime<R>(
    sources: &mut PairedSources<'_, R>,
    domain: &SourceSonicDomain<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<Option<SourceSonicNumericalEvidence<R>>>
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
    let zero = Tensor::zeros(
        domain.parent_rows.shape(),
        domain.depth.lower().dtype(),
        c.device(),
    )?;
    let selected = c.index_put(
        &zero,
        0,
        rows,
        &Tensor::ones(rows.shape(), zero.dtype(), c.device())?,
    )?;
    let mut numerical = domain.numerical.clone();
    for index in domain.numerical.policies() {
        let level = Tensor::full_scalar(&[], DType::I64, *index as f64, c.device())?;
        let mask = c.where_cond(&c.eq(domain.numerical.levels(), &level)?, &selected, &zero)?;
        let local = c.nonzero(&mask)?.flatten()?;
        if local.numel() == 0 {
            continue;
        }
        let parent = c.index_select(&domain.parent_rows, 0, &local)?;
        let empty = Tensor::empty(&[0], DType::I64, c.device())?;
        let token = sources.panel_level(index.checked_add(1).ok_or_else(|| {
            super::point_root_session::invalid("sonic panel level overflows usize")
        })?)?;
        let requested = if domain.side.index() == 0 {
            [&parent, &empty]
        } else {
            [&empty, &parent]
        };
        let values = match sources.source_values_at_level(&token, requested) {
            Ok(values) => values,
            Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                ..
            }) => return Ok(None),
            Err(error) => return Err(error),
        };
        numerical = numerical.refreshed(sources, &local, &values)?;
    }
    Ok(Some(numerical))
}

fn mark<R>(
    sources: &PairedSources<'_, R>,
    domain: &SourceSonicDomain<R>,
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
    let c = sources.geometry.client();
    let owners = c.index_select(&domain.invocation_owner, 0, rows)?;
    let levels = c.index_select(domain.numerical.levels(), 0, rows)?;
    progress_rows::mark(c, &owners, &levels, outcome, association, progress)
}
