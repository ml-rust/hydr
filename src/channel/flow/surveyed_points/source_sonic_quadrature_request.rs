use super::{
    paired_sources::PairedSources,
    root_invocation::RootInvocation,
    source_sonic_numerical::SourceSonicNumericalEvidence,
    source_sonic_quadrature_types::{
        SourceSonicQuadratureOutcome as Outcome, SourceSonicQuadratureProgress,
    },
    source_sonic_types::{SourceSonicDomain, SourceSonicRequest, SourceSonicStatus},
    wet_depth_types::WetDepthStatus,
};
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection,
    error::{Arithmetic, HydraulicError, HydraulicResult, checked},
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
    progress: &mut SourceSonicQuadratureProgress<R>,
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
                let current = sources.panel_level(*index)?;
                let next = index.checked_add(1).ok_or_else(|| {
                    super::point_root_session::invalid("sonic panel level overflows usize")
                })?;
                let permitted = next <= sources.evaluation.config.max_refinements
                    && next < usize::BITS as usize
                    && current.panels().checked_mul(2).is_some_and(|panels| {
                        panels <= sources.evaluation.config.max_simpson_panels
                    });
                if permitted {
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
        if !super::source_sonic_quadrature_capacity::permits(sources, invocation, &domain, &todo)? {
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
            let nested = c.mul(&c.ge(&lower, &old_lower)?, &c.le(&upper, &old_upper)?)?;
            let strict = c.maximum(&c.gt(&lower, &old_lower)?, &c.lt(&upper, &old_upper)?)?;
            let improved = c.mul(&nested, &strict)?;
            let status = c.index_select(&domain.status, 0, &rows)?;
            let work = Tensor::full_scalar(
                &[],
                DType::I64,
                WetDepthStatus::WorkLimit as i64 as f64,
                c.device(),
            )?;
            let unresolved = Tensor::full_scalar(
                &[],
                DType::I64,
                WetDepthStatus::Unresolved as i64 as f64,
                c.device(),
            )?;
            let outcome = c.where_cond(
                &c.eq(&status, &work)?,
                &code::<R>(c, &rows, Outcome::WorkLimit)?,
                &c.where_cond(
                    &c.eq(&status, &unresolved)?,
                    &code::<R>(c, &rows, Outcome::Unresolved)?,
                    &code::<R>(c, &rows, Outcome::Advanced)?,
                )?,
            )?;
            record(
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

struct Record<'a, R: Runtime<DType = DType>> {
    owners: &'a Tensor<R>,
    old: &'a Tensor<R>,
    new: &'a Tensor<R>,
    outcome: &'a Tensor<R>,
    improved: &'a Tensor<R>,
}
fn record<R>(
    a: &Arithmetic<'_, R>,
    association: &Tensor<R>,
    progress: &mut SourceSonicQuadratureProgress<R>,
    value: Record<'_, R>,
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
    let c = a.client();
    let positions = super::point_flux_association::positions(a, association, value.owners)?;
    progress.old_level = c.index_put(&progress.old_level, 0, &positions, value.old)?;
    progress.new_level = c.index_put(&progress.new_level, 0, &positions, value.new)?;
    progress.outcome = c.index_put(&progress.outcome, 0, &positions, value.outcome)?;
    progress.bracket_improved =
        c.index_put(&progress.bracket_improved, 0, &positions, value.improved)?;
    Ok(())
}
fn mark<R>(
    sources: &PairedSources<'_, R>,
    domain: &SourceSonicDomain<R>,
    rows: &Tensor<R>,
    outcome: Outcome,
    association: &Tensor<R>,
    progress: &mut SourceSonicQuadratureProgress<R>,
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
    checked::<R, _>(c, "sonic quadrature unchanged progress", |a| {
        let owners = c.index_select(&domain.invocation_owner, 0, rows)?;
        let levels = c.index_select(domain.numerical.levels(), 0, rows)?;
        let outcome = code::<R>(c, rows, outcome)?;
        let zero = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
        record(
            a,
            association,
            progress,
            Record {
                owners: &owners,
                old: &levels,
                new: &levels,
                outcome: &outcome,
                improved: &zero,
            },
        )
    })
}
fn code<R: Runtime<DType = DType>>(
    c: &R::Client,
    rows: &Tensor<R>,
    outcome: Outcome,
) -> HydraulicResult<Tensor<R>> {
    Ok(Tensor::full_scalar(
        rows.shape(),
        DType::I64,
        outcome as i64 as f64,
        c.device(),
    )?)
}
