use super::{
    paired_sources::PairedSources, root_invocation::RootInvocation,
    source_sonic_types::SourceSonicDomain,
};
use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, SortingOps, TensorOps},
    runtime::Runtime,
    tensor::Tensor,
};

/// Admit native work before source-potential cache priming can consume quadrature work.
pub(super) fn permits<R>(
    sources: &PairedSources<'_, R>,
    invocation: &RootInvocation,
    domain: &SourceSonicDomain<R>,
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
    if let Some(state) = &domain.refinement {
        if !state
            .limits
            .permits(domain.iterations, domain.callback_points, 0)?
        {
            return Ok(false);
        }
        let c = sources.geometry.client();
        let lower = c.index_select(domain.query.query.parameter.lower(), 0, rows)?;
        let upper = c.index_select(domain.query.query.parameter.upper(), 0, rows)?;
        let points = c.nonzero(&c.lt(&lower, &upper)?)?.numel();
        Ok(points <= remaining
            && state
                .limits
                .permits(domain.iterations, domain.callback_points, points)?)
    } else {
        if invocation.source_witness().config().root.max_iterations == 0 {
            return Ok(false);
        }
        let reservation = rows.numel().checked_mul(2).ok_or_else(|| {
            super::point_root_session::invalid(
                "sonic deferred endpoint reservation overflows usize",
            )
        })?;
        Ok(reservation <= remaining)
    }
}
