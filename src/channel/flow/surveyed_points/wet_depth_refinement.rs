use std::cell::{Cell, RefCell};

use super::{
    chart_admission, chart_root_callback, chart_root_derivative,
    paired_sources::PairedSources,
    source_binding::RootSourceWitness,
    wet_depth_mapping,
    wet_depth_types::{
        DepthRefinementState, InitializedDepthState, WetDepthDomain, WetDepthStatus,
    },
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
    witness: &RootSourceWitness,
    point_limit: usize,
    used_points: &mut usize,
    mut domain: WetDepthDomain<R>,
    tolerance: &CertifiedRootTolerance,
) -> HydraulicResult<WetDepthDomain<R>>
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
    sources.check_root_witness(witness)?;
    let remaining = point_limit
        .checked_sub(*used_points)
        .ok_or_else(|| inconsistent("callback points exceed the captured invocation limit"))?;
    let c = sources.geometry.client();
    if domain.is_deferred() {
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
    let WetDepthDomain {
        invocation_owner,
        parent_rows,
        query,
        depth: _,
        normalized_depth: _,
        status: _,
        iterations: previous_iterations,
        callback_points: previous_points,
        mapped_points: _,
        refinement,
    } = domain;
    let work = sources.evaluation.work;
    let (chart, progress) = match refinement {
        DepthRefinementState::Deferred { original } => {
            let selected = sources.select_rows(&parent_rows, &original.query)?;
            let (chart, derivative) = checked::<R, _>(c, "wet depth native admission", |a| {
                let chart = chart_admission::supported(a, &selected, &original.query)?;
                let derivative = chart_root_derivative::lower(
                    a,
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
                    chart_root_callback::evaluate(&borrowed, &parent_rows, &chart, value, &cause)
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
        DepthRefinementState::Initialized { state } => {
            let InitializedDepthState {
                continuation,
                chart,
            } = *state;
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
            let progress = c.resume_certified_mapped(
                |value| {
                    chart_root_callback::evaluate(&borrowed, &parent_rows, &chart, value, &cause)
                },
                |bounds| {
                    wet_depth_mapping::evaluate(&borrowed, &parent_rows, &chart, bounds, &cause)
                },
                continuation,
                tolerance,
                &allowance,
            );
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
    let total = used_points
        .checked_add(additional_points)
        .ok_or_else(|| inconsistent("cumulative residual points overflow usize"))?;
    if total > point_limit {
        return Err(inconsistent(
            "native residual points exceed the captured invocation limit",
        ));
    }
    work.finish_root_progress(additional_iterations, iterations)?;
    *used_points = total;
    let (query, mut status) = checked::<R, _>(c, "wet depth retained native domains", |a| {
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
            let code = Tensor::full_scalar(&[], DType::I64, native as i64 as f64, c.device())?;
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
    let (depth, normalized_depth) = wet_depth_mapping::certificates(sources, &parent_rows, &query)?;
    status = checked::<R, _>(c, "wet depth final physical criterion", |a| {
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
    Ok(WetDepthDomain {
        invocation_owner,
        parent_rows,
        query,
        depth,
        normalized_depth,
        status,
        iterations,
        callback_points,
        mapped_points,
        refinement: DepthRefinementState::Initialized {
            state: Box::new(InitializedDepthState {
                continuation,
                chart,
            }),
        },
    })
}

fn inconsistent(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!("Wet depth {message}. Discard inconsistent refinement progress"),
    }
}
