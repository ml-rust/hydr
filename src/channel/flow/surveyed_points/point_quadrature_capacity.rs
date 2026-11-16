use super::{
    paired_sources::PairedSources, point_root_session::invalid, root_invocation::RootInvocation,
};
use crate::channel::{flow::surveyed::error::HydraulicResult, section::SupportedAreaQuery};
use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, SortingOps, TensorOps},
    runtime::Runtime,
    tensor::Tensor,
};

/// Immutable ceilings copied from the exact options supplied to the native initialization call.
pub(super) struct NativeRootLimits {
    max_iterations: usize,
    max_points: usize,
}

impl NativeRootLimits {
    pub(super) fn capture(options: &solvr::optimize::CertifiedBisectionOptions) -> Self {
        Self {
            max_iterations: options.max_iterations,
            max_points: options.max_points,
        }
    }

    pub(super) fn permits(
        &self,
        iterations: usize,
        points: usize,
        requested_points: usize,
    ) -> HydraulicResult<bool> {
        let rounds = self.max_iterations.checked_sub(iterations).ok_or_else(|| {
            invalid("root iterations exceed their actual original native ceiling")
        })?;
        let remaining = self
            .max_points
            .checked_sub(points)
            .ok_or_else(|| invalid("root points exceed their actual original native ceiling"))?;
        Ok(rounds != 0 && remaining != 0 && requested_points <= remaining)
    }
}

/// Native progress of one initialized continuation: its limits, rounds and points.
pub(super) struct NativeRootUse<'a> {
    pub(super) limits: &'a NativeRootLimits,
    pub(super) iterations: usize,
    pub(super) callback_points: usize,
}

/// Admit native work before any quadrature callback can consume it.
/// `None` marks a deferred domain, which reserves two endpoints per row.
pub(super) fn permits<R>(
    sources: &PairedSources<'_, R>,
    invocation: &RootInvocation,
    native: Option<NativeRootUse<'_>>,
    query: &SupportedAreaQuery<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<bool>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R>,
{
    let remaining = invocation.remaining()?;
    if remaining == 0 {
        return Ok(false);
    }
    if let Some(native) = native {
        if !native
            .limits
            .permits(native.iterations, native.callback_points, 0)?
        {
            return Ok(false);
        }
        let c = sources.geometry.client();
        let lower = c.index_select(query.query.parameter.lower(), 0, rows)?;
        let upper = c.index_select(query.query.parameter.upper(), 0, rows)?;
        let points = c.nonzero(&c.lt(&lower, &upper)?)?.numel();
        Ok(points <= remaining
            && native
                .limits
                .permits(native.iterations, native.callback_points, points)?)
    } else {
        if invocation.source_witness().config().root.max_iterations == 0 {
            return Ok(false);
        }
        let reservation = rows
            .numel()
            .checked_mul(2)
            .ok_or_else(|| invalid("deferred endpoint reservation overflows usize"))?;
        Ok(reservation <= remaining)
    }
}

/// The level after `index` stays within the refinement, panel and integer-doubling ceilings.
pub(super) fn next_level<R>(sources: &PairedSources<'_, R>, index: usize) -> HydraulicResult<bool>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + numr::ops::MidpointOps<R>
        + numr::ops::NextRepresentableOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::runtime::RuntimeClient<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>,
{
    let current = sources.panel_level(index)?;
    let next = index
        .checked_add(1)
        .ok_or_else(|| invalid("quadrature panel level overflows usize"))?;
    Ok(next <= sources.evaluation.config.max_refinements
        && next < usize::BITS as usize
        && current
            .panels()
            .checked_mul(2)
            .is_some_and(|panels| panels <= sources.evaluation.config.max_simpson_panels))
}
