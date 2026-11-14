use std::cell::{Cell, RefCell};

use super::{
    paired_sources::PairedSources,
    point_root_depth as wet_depth_mapping,
    root_invocation::RootInvocation,
    source_sonic_callback, source_sonic_charts,
    source_sonic_types::{SourceSonicBinding, SourceSonicContinuation, SourceSonicDomain},
    wet_depth_types::WetDepthStatus,
};
use crate::channel::flow::surveyed::{
    error::{HydraulicError, HydraulicResult, checked},
    root_calls::resolve_root,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms, CertifiedRootAdvanceAllowance, CertifiedRootStatus,
        CertifiedRootTolerance,
    },
};

pub(super) fn advance<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    request_binding: &SourceSonicBinding<R>,
    domain: SourceSonicDomain<R>,
    tolerance: &CertifiedRootTolerance,
) -> HydraulicResult<SourceSonicDomain<R>>
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
    advance_with_selection(
        sources,
        invocation,
        request_binding,
        domain,
        tolerance,
        None,
    )
}

pub(super) fn advance_selected<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    request_binding: &SourceSonicBinding<R>,
    domain: SourceSonicDomain<R>,
    tolerance: &CertifiedRootTolerance,
    rows: &Tensor<R>,
) -> HydraulicResult<SourceSonicDomain<R>>
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
    if domain.refinement.is_none() {
        return Err(inconsistent("selected resume has no native continuation"));
    }
    advance_with_selection(
        sources,
        invocation,
        request_binding,
        domain,
        tolerance,
        Some(rows),
    )
}

fn advance_with_selection<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    request_binding: &SourceSonicBinding<R>,
    mut domain: SourceSonicDomain<R>,
    tolerance: &CertifiedRootTolerance,
    selection: Option<&Tensor<R>>,
) -> HydraulicResult<SourceSonicDomain<R>>
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
    sources.check_root_witness(invocation.source_witness())?;
    request_binding.check(invocation, domain.side, &domain.binding)?;
    if let Some(state) = &domain.refinement {
        request_binding.check(invocation, domain.side, &state.binding)?;
    }
    let remaining = invocation.remaining()?;
    let witness = invocation.source_witness();
    let c = sources.geometry.client();
    if domain.refinement.is_none() {
        let admission = domain
            .parent_rows
            .numel()
            .checked_mul(2)
            .ok_or_else(|| inconsistent("initial endpoint reservation overflows usize"))?;
        if admission > remaining {
            domain.status = Tensor::full_scalar(
                domain.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::WorkLimit as i64 as f64,
                c.device(),
            )?;
            return Ok(domain);
        }
    }
    let SourceSonicDomain {
        binding,
        invocation_owner,
        parent_rows,
        side,
        query,
        epsilon,
        classification,
        depth: old_depth,
        normalized_depth: old_normalized,
        status: old_status,
        iterations: previous_iterations,
        callback_points: previous_points,
        mapped_points: _,
        refinement,
    } = domain;
    let work = sources.evaluation.work;
    let before = work.snapshot().residual_points;
    let (chart, progress) = match refinement {
        None => {
            let selected = sources.select_rows(&parent_rows, &query)?;
            let (chart, derivative) = checked::<R, _>(c, "source sonic native admission", |a| {
                let chart =
                    query.with_parameter_bounds(a, &selected.bands, &query.query.parameter)?;
                let derivative = source_sonic_charts::derivative(
                    a,
                    sources.geometry,
                    &selected.bands,
                    &chart,
                    witness.config().gravity_m_per_s2,
                )?;
                Ok((chart, derivative))
            })?;
            let mut options = witness.config().root.clone();
            options.atol = tolerance.atol;
            options.rtol = tolerance.rtol;
            options.max_points = remaining;
            let cause = Cell::new(None);
            work.begin_root()?;
            let borrowed = RefCell::new(&mut *sources);
            let progress = c.bisect_certified_mapped(
                |value| {
                    source_sonic_callback::evaluate(
                        &borrowed,
                        &parent_rows,
                        &chart,
                        &epsilon,
                        side,
                        value,
                        &cause,
                    )
                },
                |bounds| {
                    wet_depth_mapping::evaluate(&borrowed, &parent_rows, &chart, bounds, &cause)
                },
                chart.query.parameter.lower(),
                chart.query.parameter.upper(),
                &derivative,
                &options,
            );
            (chart, resolve_root(progress, cause.take())?)
        }
        Some(state) => {
            let SourceSonicContinuation {
                binding: _,
                continuation,
                chart,
            } = state;
            sources.select_rows(&parent_rows, &chart)?;
            let rounds = witness
                .config()
                .root
                .max_iterations
                .checked_sub(previous_iterations)
                .ok_or_else(|| {
                    inconsistent("session midpoint rounds exceed their original ceiling")
                })?;
            let allowance = CertifiedRootAdvanceAllowance {
                max_iterations: rounds,
                max_points: remaining,
            };
            let cause = Cell::new(None);
            work.begin_root()?;
            let borrowed = RefCell::new(&mut *sources);
            let residual = |value: &solvr::optimize::CertifiedRootQuery<R>| {
                source_sonic_callback::evaluate(
                    &borrowed,
                    &parent_rows,
                    &chart,
                    &epsilon,
                    side,
                    value,
                    &cause,
                )
            };
            let mapping = |bounds: &solvr::optimize::CertifiedRootDomainQuery<R>| {
                wet_depth_mapping::evaluate(&borrowed, &parent_rows, &chart, bounds, &cause)
            };
            let progress = if let Some(rows) = selection {
                c.resume_certified_mapped_selected(
                    residual,
                    mapping,
                    continuation,
                    rows,
                    tolerance,
                    &allowance,
                )
            } else {
                c.resume_certified_mapped(residual, mapping, continuation, tolerance, &allowance)
            };
            (chart, resolve_root(progress, cause.take())?)
        }
    };
    let mapped_points = progress.mapped_points();
    let (result, continuation) = progress.into_parts();
    let iterations = result.iterations();
    let callback_points = result.evaluated_points();
    let additional_iterations = iterations
        .checked_sub(previous_iterations)
        .ok_or_else(|| inconsistent("cumulative midpoint rounds decreased"))?;
    let additional_points = callback_points
        .checked_sub(previous_points)
        .ok_or_else(|| inconsistent("cumulative residual points decreased"))?;
    if work.snapshot().residual_points.checked_sub(before) != Some(additional_points) {
        return Err(inconsistent(
            "native diagnostics differ from executed callbacks",
        ));
    }
    invocation.check_add_points(additional_points)?;
    work.finish_root_progress(additional_iterations, iterations)?;
    invocation.add_points(additional_points)?;
    let (query, depth, normalized_depth, status) = if let Some(rows) = selection {
        let cached = super::source_sonic_selected_fields::Cached {
            query,
            depth: old_depth,
            normalized: old_normalized,
            status: old_status,
        };
        let updated = super::source_sonic_selected_fields::update(
            sources,
            &parent_rows,
            cached,
            &result,
            rows,
            tolerance,
        )?;
        (
            updated.query,
            updated.depth,
            updated.normalized,
            updated.status,
        )
    } else {
        let (query, mut status) =
            checked::<R, _>(c, "source sonic retained native domains", |a| {
                let lower = c.index_put(
                    query.query.parameter.lower(),
                    0,
                    result.root_owner(),
                    result.root_lower(),
                )?;
                let upper = c.index_put(
                    query.query.parameter.upper(),
                    0,
                    result.root_owner(),
                    result.root_upper(),
                )?;
                let parameter = a.bounds(&lower, &upper)?;
                let selected = sources.select_sources(a, &parent_rows)?;
                let query = query.with_parameter_bounds(a, &selected.bands, &parameter)?;
                let mut status = Tensor::full_scalar(
                    parent_rows.shape(),
                    DType::I64,
                    WetDepthStatus::Unresolved as i64 as f64,
                    c.device(),
                )?;
                for (native, physical) in [
                    (
                        CertifiedRootStatus::Converged,
                        WetDepthStatus::DepthConverged,
                    ),
                    (
                        CertifiedRootStatus::AccuracyFloor,
                        WetDepthStatus::AccuracyFloor,
                    ),
                    (CertifiedRootStatus::WorkLimit, WetDepthStatus::WorkLimit),
                ] {
                    let code =
                        Tensor::full_scalar(&[], DType::I64, native as i64 as f64, c.device())?;
                    let target = Tensor::full_scalar(
                        parent_rows.shape(),
                        DType::I64,
                        physical as i64 as f64,
                        c.device(),
                    )?;
                    status = c.where_cond(&c.eq(result.status(), &code)?, &target, &status)?;
                }
                Ok((query, status))
            })?;
        let (depth, normalized_depth) =
            wet_depth_mapping::certificates(sources, &parent_rows, &query)?;
        status = checked::<R, _>(c, "source sonic final physical criterion", |a| {
            let accepted = wet_depth_mapping::satisfied(a, &normalized_depth, tolerance)?;
            let passed = Tensor::full_scalar(
                parent_rows.shape(),
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            let converged_code = Tensor::full_scalar(
                &[],
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            let unresolved = Tensor::full_scalar(
                parent_rows.shape(),
                DType::I64,
                WetDepthStatus::Unresolved as i64 as f64,
                c.device(),
            )?;
            let other = c.where_cond(&c.eq(&status, &converged_code)?, &unresolved, &status)?;
            Ok(c.where_cond(&accepted, &passed, &other)?)
        })?;
        (query, depth, normalized_depth, status)
    };
    Ok(SourceSonicDomain {
        binding: binding.clone(),
        invocation_owner,
        parent_rows,
        side,
        query,
        epsilon,
        classification,
        depth,
        normalized_depth,
        status,
        iterations,
        callback_points,
        mapped_points,
        refinement: Some(SourceSonicContinuation {
            binding,
            continuation,
            chart,
        }),
    })
}

fn inconsistent(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!("Source sonic {message}. Discard inconsistent refinement progress"),
    }
}
