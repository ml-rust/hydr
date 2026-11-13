use super::{
    potential_range::select_expansion,
    wet_depth_types::{DepthRefinementState, WetDepthDomain, WetDepthStatus},
    wet_root_types::WetRootDomain,
};
use crate::channel::flow::surveyed::{
    equation_ops::select_interval,
    error::{Arithmetic, HydraulicError, HydraulicResult},
};
use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Deferred groups have no midpoint history, so native row selection preserves their complete original evidence.
pub(super) fn deferred<R>(
    a: &mut Arithmetic<'_, R>,
    domain: &WetDepthDomain<R>,
    rows: &Tensor<R>,
    outcome: WetDepthStatus,
) -> HydraulicResult<WetDepthDomain<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let DepthRefinementState::Deferred { original } = &domain.refinement else {
        return Err(HydraulicError::Numerical { context: "Initialized wet depth groups cannot split into fresh sessions. Resume their retained numerical state".into() });
    };
    let c = a.client();
    let original = WetRootDomain {
        invocation_witness: original.invocation_witness.clone(),
        invocation_owner: c.index_select(&original.invocation_owner, 0, rows)?,
        parent_rows: c.index_select(&original.parent_rows, 0, rows)?,
        query: original.query.select(a, rows)?,
        lower_residual: select_interval(a, &original.lower_residual, rows)?,
        upper_residual: select_interval(a, &original.upper_residual, rows)?,
        lower_retained: select_expansion(a, &original.lower_retained, rows)?,
        upper_retained: select_expansion(a, &original.upper_retained, rows)?,
    };
    Ok(WetDepthDomain {
        invocation_witness: domain.invocation_witness.clone(),
        invocation_owner: c.index_select(&domain.invocation_owner, 0, rows)?,
        parent_rows: c.index_select(&domain.parent_rows, 0, rows)?,
        query: domain.query.select(a, rows)?,
        depth: select_interval(a, &domain.depth, rows)?,
        normalized_depth: select_interval(a, &domain.normalized_depth, rows)?,
        status: Tensor::full_scalar(rows.shape(), DType::I64, outcome as i64 as f64, c.device())?,
        iterations: 0,
        callback_points: 0,
        mapped_points: 0,
        refinement: DepthRefinementState::Deferred {
            original: Box::new(original),
        },
    })
}
