use super::{
    paired_sources::PairedSources,
    potential_range::select_expansion,
    root_invocation::RootInvocation,
    source_sonic_callback, source_sonic_charts, source_sonic_residual,
    source_sonic_types::{SonicSide, SourceSonicDomain, SourceSonicRequest, SourceSonicStatus},
    wave_curve_types::{CurveValue, scatter_value},
    wet_depth_types::WetDepthStatus,
    wet_star_endpoint::exact_zero,
};
use crate::channel::{
    flow::surveyed::{
        error::{HydraulicResult, checked},
        root_calls::resolve_root,
        work::QueryWork,
    },
    section::{PhysicalEndpoint, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::CertifiedExpansion,
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions, CertifiedUpperBracketStatus,
    },
};
use std::cell::{Cell, RefCell};

pub(super) fn locate<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    request: &mut SourceSonicRequest<R>,
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
        + CertifiedUpperBracketAlgorithms<R>,
{
    let c = sources.geometry.client();
    let side = request.binding.side;
    let (wet, parent_rows) = checked::<R, _>(c, "source sonic selected-side admission", |a| {
        let selected = sources.select_sources(a, &request.binding.rows)?;
        let zero = Tensor::zeros(request.binding.rows.shape(), a.dtype(), c.device())?;
        let wet_mask = c.eq(selected.sources[side.index()].dry_mask(), &zero)?;
        let wet = c.nonzero(&wet_mask)?.flatten()?;
        let unavailable = Tensor::full_scalar(
            request.status.shape(),
            DType::I64,
            SourceSonicStatus::DrySourceUnavailable as i64 as f64,
            c.device(),
        )?;
        request.status = c.where_cond(&wet_mask, &request.status, &unavailable)?;
        let parent_rows = c.index_select(&request.binding.rows, 0, &wet)?;
        Ok((wet, parent_rows))
    })?;
    if wet.numel() == 0 {
        return Ok(());
    }
    let endpoint_points = wet.numel().checked_mul(2).ok_or_else(|| {
        super::point_root_session::invalid("source sonic endpoint count overflows usize")
    })?;
    if endpoint_points > invocation.remaining()? {
        request.status = c.index_put(
            &request.status,
            0,
            &wet,
            &Tensor::full_scalar(
                wet.shape(),
                DType::I64,
                SourceSonicStatus::WorkLimit as i64 as f64,
                c.device(),
            )?,
        )?;
        return Ok(());
    }
    sources
        .evaluation
        .work
        .charge(QueryWork::Residual, endpoint_points)?;
    let empty = Tensor::empty(&[0], DType::I64, c.device())?;
    let values = match side {
        SonicSide::Left => sources.source_values([&parent_rows, &empty])?,
        SonicSide::Right => sources.source_values([&empty, &parent_rows])?,
    };
    let [left, right] = values;
    let potential = if side == SonicSide::Left { left } else { right };
    let (epsilon, dry) = checked::<R, _>(c, "source sonic retained outgoing invariant", |a| {
        let selected = sources.select_sources(a, &parent_rows)?;
        let velocity = if side == SonicSide::Left {
            &selected.velocity.left
        } else {
            &selected.velocity.right
        };
        let epsilon = if side == SonicSide::Left {
            a.expansion_add(velocity, &potential.retained)?
        } else {
            a.expansion_sub(&potential.retained, velocity)?
        };
        let zero =
            a.expansion_point(&Tensor::zeros(parent_rows.shape(), a.dtype(), c.device())?)?;
        let retained = a.expansion_sub(&zero, &epsilon)?;
        let dry = CurveValue {
            physical: a.expansion_interval(&retained)?,
            retained,
            available: potential.available,
        };
        Ok((epsilon, dry))
    })?;
    let one = Tensor::ones(parent_rows.shape(), epsilon.high().dtype(), c.device())?;
    let source_chart = source_sonic_charts::chart(sources, &parent_rows, side, &one, &one)?;
    let source = source_sonic_residual::endpoint(
        sources,
        &parent_rows,
        &source_chart,
        PhysicalEndpoint::Lower,
        side,
        &epsilon,
    )?;
    invocation.add_points(endpoint_points)?;
    request.endpoint_points = endpoint_points;
    let (positive, dry_contact, source_contact) =
        checked::<R, _>(c, "source sonic endpoint classification", |a| {
            let zero = Tensor::zeros(parent_rows.shape(), a.dtype(), c.device())?;
            let dry_positive = c.gt(dry.physical.lower(), &zero)?;
            let source_negative = c.lt(source.physical.upper(), &zero)?;
            let contradiction = c.mul(&dry_positive, &source_negative)?;
            let consistent = c.eq(&contradiction, &zero)?;
            let negative_dry = c.lt(dry.physical.upper(), &zero)?;
            let positive_source = c.gt(source.physical.lower(), &zero)?;
            let positive = c.mul(&negative_dry, &positive_source)?;
            let dry_contact = c.mul(&exact_zero(a, &dry.retained)?, &positive_source)?;
            let source_contact = c.mul(&exact_zero(a, &source.retained)?, &negative_dry)?;
            let mut status = Tensor::full_scalar(
                wet.shape(),
                DType::I64,
                SourceSonicStatus::Unresolved as i64 as f64,
                c.device(),
            )?;
            for (mask, code) in [
                (
                    &c.mul(&dry_positive, &consistent)?,
                    SourceSonicStatus::NoRootDry,
                ),
                (
                    &c.mul(&source_negative, &consistent)?,
                    SourceSonicStatus::NoRootSource,
                ),
                (&positive, SourceSonicStatus::PositiveRoot),
                (&dry_contact, SourceSonicStatus::DryContact),
                (&source_contact, SourceSonicStatus::SourceContact),
            ] {
                status = c.where_cond(
                    mask,
                    &Tensor::full_scalar(wet.shape(), DType::I64, code as i64 as f64, c.device())?,
                    &status,
                )?;
            }
            request.status = c.index_put(&request.status, 0, &wet, &status)?;
            let empty = super::wave_curve_types::unavailable(a, request.binding.rows.numel())?;
            let previous = std::mem::replace(&mut request.dry_residual, empty);
            request.dry_residual = scatter_value(a, previous, &wet, &dry)?;
            let empty = super::wave_curve_types::unavailable(a, request.binding.rows.numel())?;
            let previous = std::mem::replace(&mut request.source_residual, empty);
            request.source_residual = scatter_value(a, previous, &wet, &source)?;
            Ok((
                c.nonzero(&positive)?.flatten()?,
                c.nonzero(&dry_contact)?.flatten()?,
                c.nonzero(&source_contact)?.flatten()?,
            ))
        })?;
    for (selected, classification, parameter) in [
        (&dry_contact, SourceSonicStatus::DryContact, 0.0),
        (&source_contact, SourceSonicStatus::SourceContact, 1.0),
    ] {
        if selected.numel() == 0 {
            continue;
        }
        let request_rows = c.index_select(&wet, 0, selected)?;
        let parents = c.index_select(&parent_rows, 0, selected)?;
        let epsilon = checked::<R, _>(c, "source sonic contact invariant", |a| {
            select_expansion(a, &epsilon, selected)
        })?;
        let point = Tensor::full_scalar(
            selected.shape(),
            epsilon.high().dtype(),
            parameter,
            c.device(),
        )?;
        let query = source_sonic_charts::chart(sources, &parents, side, &point, &point)?;
        push_domain(
            sources,
            request,
            &request_rows,
            parents,
            query,
            epsilon,
            classification,
        )?;
    }
    if positive.numel() != 0 {
        let request_rows = c.index_select(&wet, 0, &positive)?;
        let parents = c.index_select(&parent_rows, 0, &positive)?;
        let epsilon = checked::<R, _>(c, "source sonic positive invariant", |a| {
            select_expansion(a, &epsilon, &positive)
        })?;
        let zero = Tensor::zeros(positive.shape(), epsilon.high().dtype(), c.device())?;
        let one = Tensor::ones(positive.shape(), epsilon.high().dtype(), c.device())?;
        let mut query = source_sonic_charts::chart(sources, &parents, side, &zero, &one)?;
        let admission = positive.numel().checked_mul(2).ok_or_else(|| {
            super::point_root_session::invalid("source sonic bracket admission overflows usize")
        })?;
        if admission <= invocation.remaining()? {
            let seed = source_sonic_charts::seed(sources, &parents, side, &epsilon, &query)?;
            let work = sources.evaluation.work;
            work.begin_bracket()?;
            let cause = Cell::new(None);
            let (before, result) = {
                let borrowed = RefCell::new(&mut *sources);
                let before = work.snapshot().residual_points;
                let result = c.bracket_upper_certified(
                    |value| {
                        source_sonic_callback::evaluate(
                            &borrowed, &parents, &query, &epsilon, side, value, &cause,
                        )
                    },
                    query.query.parameter.lower(),
                    &seed,
                    query.query.parameter.upper(),
                    &CertifiedUpperBracketOptions {
                        max_expansions: invocation.source_witness().config().max_bracket_expansions,
                        max_points: invocation.remaining()?,
                    },
                );
                (before, resolve_root(result, cause.take())?)
            };
            if work.snapshot().residual_points.checked_sub(before) != Some(result.callback_points())
            {
                return Err(super::point_root_session::invalid(
                    "source sonic bracket diagnostics differ from executed callbacks",
                ));
            }
            work.finish_bracket(result.expansion_rounds())?;
            invocation.add_points(result.callback_points())?;
            request.bracket_points = result.callback_points();
            request.bracket_expansions = result.expansion_rounds();
            query = checked::<R, _>(c, "source sonic retained bracket domain", |a| {
                let code = Tensor::full_scalar(
                    &[],
                    DType::I64,
                    CertifiedUpperBracketStatus::Bracketed as i64 as f64,
                    c.device(),
                )?;
                let bracketed = c.eq(result.status(), &code)?;
                let bounds = a.bounds(
                    &c.where_cond(&bracketed, result.lower(), &zero)?,
                    &c.where_cond(&bracketed, result.upper(), &one)?,
                )?;
                let selected = sources.select_sources(a, &parents)?;
                query.with_parameter_bounds(a, &selected.bands, &bounds)
            })?;
        }
        push_domain(
            sources,
            request,
            &request_rows,
            parents,
            query,
            epsilon,
            SourceSonicStatus::PositiveRoot,
        )?;
    }
    Ok(())
}

fn push_domain<R>(
    sources: &PairedSources<'_, R>,
    request: &mut SourceSonicRequest<R>,
    request_rows: &Tensor<R>,
    parent_rows: Tensor<R>,
    query: SupportedAreaQuery<R>,
    epsilon: CertifiedExpansion<R>,
    classification: SourceSonicStatus,
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
    let (invocation_owner, binding) =
        checked::<R, _>(c, "source sonic domain identity capture", |_| {
            let owners = c.index_select(&request.binding.owners, 0, request_rows)?;
            let captured_owners = Tensor::empty(owners.shape(), DType::I64, c.device())?;
            let captured_rows = Tensor::empty(parent_rows.shape(), DType::I64, c.device())?;
            c.copy_into(&captured_owners, &owners)?;
            c.copy_into(&captured_rows, &parent_rows)?;
            Ok((
                owners,
                request.binding.selected(captured_owners, captured_rows),
            ))
        })?;
    let (depth, normalized_depth) = if classification == SourceSonicStatus::DryContact {
        checked::<R, _>(c, "source sonic analytic dry contact depth", |a| {
            let zero = Tensor::zeros(parent_rows.shape(), a.dtype(), c.device())?;
            Ok((a.point(&zero)?, a.point(&zero)?))
        })?
    } else {
        super::point_root_depth::certificates(sources, &parent_rows, &query)?
    };
    let status = Tensor::full_scalar(
        parent_rows.shape(),
        DType::I64,
        WetDepthStatus::Unresolved as i64 as f64,
        c.device(),
    )?;
    request.domains.push(SourceSonicDomain {
        binding,
        invocation_owner,
        parent_rows,
        side: request.binding.side,
        query,
        epsilon,
        classification,
        depth,
        normalized_depth,
        status,
        iterations: 0,
        callback_points: 0,
        mapped_points: 0,
        refinement: None,
    });
    Ok(())
}
