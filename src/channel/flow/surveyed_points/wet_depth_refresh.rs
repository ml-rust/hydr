use std::cell::{Cell, RefCell};

use super::{
    chart_root_callback,
    paired_sources::PairedSources,
    root_invocation::RootInvocation,
    source_sonic_selected_fields::{Cached, update},
    wet_depth_mapping,
    wet_depth_types::{DepthRefinementState, InitializedDepthState, WetDepthDomain},
};
use crate::channel::flow::surveyed::{
    error::{HydraulicError, HydraulicResult},
    root_calls::resolve_root,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms, CertifiedRootAdvanceAllowance,
        CertifiedRootDomainQuery, CertifiedRootQuery, CertifiedRootTolerance,
    },
};

/// Why the selected rows of an initialized domain continue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectedContinuation {
    /// The residual panel level changed. The domain levels already hold the new level.
    Refresh,
    /// Only the tolerance tightened. The residual and its panel level stay fixed.
    Resume,
}

/// Continues the selected local rows of an initialized domain.
/// Undemanded rows keep every stored bit.
/// An error consumes the continuation, so the caller must seal its session.
pub(super) fn refresh_selected<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    domain: WetDepthDomain<R>,
    tolerance: &CertifiedRootTolerance,
    rows: &Tensor<R>,
    mode: SelectedContinuation,
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
    sources.check_root_witness(invocation.source_witness())?;
    domain.check_invocation(invocation)?;
    let WetDepthDomain {
        invocation_witness,
        invocation_owner,
        parent_rows,
        query,
        depth,
        normalized_depth,
        status,
        levels,
        iterations: previous_iterations,
        callback_points: previous_points,
        mapped_points: _,
        refinement,
    } = domain;
    let DepthRefinementState::Initialized { state } = refinement else {
        return Err(inconsistent("refresh has no native continuation"));
    };
    let InitializedDepthState {
        invocation_witness: state_witness,
        continuation,
        chart,
        limits,
    } = *state;
    let remaining = invocation.remaining()?;
    let witness = invocation.source_witness();
    let c = sources.geometry.client();
    sources.select_rows(&parent_rows, &chart)?;
    let rounds = witness
        .config()
        .root
        .max_iterations
        .checked_sub(previous_iterations)
        .ok_or_else(|| inconsistent("session midpoint rounds exceed their original ceiling"))?;
    let allowance = CertifiedRootAdvanceAllowance {
        max_iterations: rounds,
        max_points: remaining,
    };
    let work = sources.evaluation.work;
    let before = work.snapshot().residual_points;
    let cause = Cell::new(None);
    work.begin_root()?;
    let borrowed = RefCell::new(&mut *sources);
    let residual = |value: &CertifiedRootQuery<R>| {
        chart_root_callback::evaluate(
            &borrowed,
            &parent_rows,
            Some(&levels),
            &chart,
            value,
            &cause,
        )
    };
    let mapping = |bounds: &CertifiedRootDomainQuery<R>| {
        wet_depth_mapping::evaluate(&borrowed, &parent_rows, &chart, bounds, &cause)
    };
    let progress = match mode {
        SelectedContinuation::Refresh => c.refresh_certified_mapped_selected(
            residual,
            mapping,
            continuation,
            rows,
            tolerance,
            &allowance,
        ),
        SelectedContinuation::Resume => c.resume_certified_mapped_selected(
            residual,
            mapping,
            continuation,
            rows,
            tolerance,
            &allowance,
        ),
    };
    let progress = resolve_root(progress, cause.take())?;
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
    let cached = Cached {
        query,
        depth,
        normalized: normalized_depth,
        status,
    };
    let updated = update(sources, &parent_rows, cached, &result, rows, tolerance)?;
    Ok(WetDepthDomain {
        invocation_witness,
        invocation_owner,
        parent_rows,
        query: updated.query,
        depth: updated.depth,
        normalized_depth: updated.normalized,
        status: updated.status,
        levels,
        iterations,
        callback_points,
        mapped_points,
        refinement: DepthRefinementState::Initialized {
            state: Box::new(InitializedDepthState {
                invocation_witness: state_witness,
                continuation,
                chart,
                limits,
            }),
        },
    })
}

fn inconsistent(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!("Wet star refresh {message}. Discard inconsistent refinement progress"),
    }
}
