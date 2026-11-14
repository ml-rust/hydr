use super::{
    paired_sources::PairedSources,
    root_invocation::RootInvocation,
    source_sonic_types::{SourceSonicDomain, SourceSonicRequest, SourceSonicStatus},
    wet_depth_types::WetDepthStatus,
};
use crate::channel::flow::surveyed::{
    equation_ops::select_interval,
    error::{HydraulicResult, checked},
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance},
};

fn deferred<R>(
    sources: &PairedSources<'_, R>,
    domain: &SourceSonicDomain<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<SourceSonicDomain<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if domain.refinement.is_some()
        || domain.iterations != 0
        || domain.callback_points != 0
        || domain.mapped_points != 0
    {
        return Err(super::point_root_session::invalid(
            "source sonic partition requires deferred state with zero native work",
        ));
    }
    let c = sources.geometry.client();
    checked::<R, _>(
        c,
        "source sonic deferred immutable operand partition",
        |a| {
            let owners = c.index_select(&domain.invocation_owner, 0, rows)?;
            let parent = c.index_select(&domain.parent_rows, 0, rows)?;
            Ok(SourceSonicDomain {
                binding: domain.binding.selected(owners.clone(), parent.clone()),
                invocation_owner: owners,
                parent_rows: parent,
                side: domain.side,
                query: domain.query.select(a, rows)?,
                epsilon: super::potential_range::select_expansion(a, &domain.epsilon, rows)?,
                classification: domain.classification,
                depth: select_interval(a, &domain.depth, rows)?,
                normalized_depth: select_interval(a, &domain.normalized_depth, rows)?,
                status: c.index_select(&domain.status, 0, rows)?,
                iterations: 0,
                callback_points: 0,
                mapped_points: 0,
                refinement: None,
            })
        },
    )
}

pub(super) fn refine<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    request: &mut SourceSonicRequest<R>,
    demand: &Tensor<R>,
    tolerance: &CertifiedRootTolerance,
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
        let local = c.index_select(demand, 0, &domain.invocation_owner)?;
        let selected = c.nonzero(&local)?.flatten()?;
        if selected.numel() == 0 || domain.classification != SourceSonicStatus::PositiveRoot {
            output.push(domain);
            continue;
        }
        request
            .binding
            .check(invocation, domain.side, &domain.binding)?;
        let todo = checked::<R, _>(c, "source sonic demanded stopping criterion", |a| {
            let normalized = select_interval(a, &domain.normalized_depth, &selected)?;
            let accepted = super::point_root_depth::satisfied(a, &normalized, tolerance)?;
            let old_status = c.index_select(&domain.status, 0, &selected)?;
            let passed = Tensor::full_scalar(
                selected.shape(),
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            let status = c.where_cond(&accepted, &passed, &old_status)?;
            domain.status = c.index_put(&domain.status, 0, &selected, &status)?;
            let zero = Tensor::zeros(accepted.shape(), a.dtype(), c.device())?;
            let needed = c.nonzero(&c.eq(&accepted, &zero)?)?.flatten()?;
            Ok(c.index_select(&selected, 0, &needed)?)
        })?;
        if todo.numel() == 0 {
            output.push(domain);
            continue;
        }
        if domain.refinement.is_some() {
            output.push(super::source_sonic_refinement::advance_selected(
                sources,
                invocation,
                &request.binding,
                domain,
                tolerance,
                &todo,
            )?);
        } else if todo.numel() == domain.parent_rows.numel() {
            output.push(super::source_sonic_refinement::advance(
                sources,
                invocation,
                &request.binding,
                domain,
                tolerance,
            )?);
        } else {
            let demanded = deferred(sources, &domain, &todo)?;
            let zero = Tensor::zeros(domain.parent_rows.shape(), local.dtype(), c.device())?;
            let mask = c.index_put(
                &zero,
                0,
                &todo,
                &Tensor::ones(todo.shape(), local.dtype(), c.device())?,
            )?;
            let complement = c.nonzero(&c.eq(&mask, &zero)?)?.flatten()?;
            output.push(deferred(sources, &domain, &complement)?);
            output.push(super::source_sonic_refinement::advance(
                sources,
                invocation,
                &request.binding,
                demanded,
                tolerance,
            )?);
        }
    }
    request.domains = output;
    Ok(())
}
