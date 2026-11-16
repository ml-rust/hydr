use crate::channel::{
    flow::surveyed::error::{
        Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection,
    },
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn positive<R>(
    a: &Arithmetic<'_, R>,
    query: &SupportedAreaQuery<R>,
    invalid: &Tensor<R>,
    quantity: &str,
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
    let rows = c.nonzero(invalid)?.flatten()?;
    if rows.numel() == 0 {
        return Ok(());
    }
    let first = rows.narrow(0, 0, 1)?;
    let original = c
        .index_select(&query.query.original_rows, 0, &first)?
        .item::<i64>()?;
    let owner = c
        .index_select(&query.query.owners, 0, &first)?
        .item::<i64>()?;
    Err(unresolved_positive(original, owner, quantity)?)
}

/// Rejection for a selected quantity that lacks strict positivity at one original row.
pub(super) fn unresolved_positive(
    original: i64,
    owner: i64,
    quantity: &str,
) -> HydraulicResult<HydraulicError> {
    let original = usize::try_from(original).map_err(|_| {
        super::point_root_session::invalid(
            "selected point original row is negative. Preserve the bound query",
        )
    })?;
    let owner = usize::try_from(owner).map_err(|_| {
        super::point_root_session::invalid(
            "selected point section owner is negative. Preserve the bound query",
        )
    })?;
    Ok(HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::GeometryUnresolved,
        query: Some(original),
        section_owner: Some(owner),
        context: format!(
            "selected {quantity} lacks strict positivity at original row={original}, section owner={owner}. Resume the original depth continuation"
        ),
    })
}
