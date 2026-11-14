use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
        UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn inverse<R>(
    a: &Arithmetic<'_, R>,
    invocation_count: usize,
    requested: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
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
    if requested.numel() == 0 {
        return Ok(Tensor::empty(&[0], DType::I64, c.device())?);
    }
    if requested.numel() as u128 > (1_u128 << f64::MANTISSA_DIGITS) {
        return Err(super::point_root_session::invalid(
            "point flux requested count exceeds exact arange indexing. Split the request",
        ));
    }
    let empty = Tensor::from_slice(&[-1_i64], &[], c.device())?
        .broadcast_to(&[invocation_count])?
        .contiguous()?;
    let positions = c.arange(0., requested.numel() as f64, 1., DType::I64)?;
    Ok(c.index_put(&empty, 0, requested, &positions)?)
}

pub(super) fn positions<R>(
    a: &Arithmetic<'_, R>,
    inverse: &Tensor<R>,
    owners: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
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
    let positions = c.index_select(inverse, 0, owners)?;
    let zero = Tensor::from_slice(&[0_i64], &[], c.device())?;
    if c.nonzero(&c.lt(&positions, &zero)?)?.numel() != 0 {
        return Err(super::point_root_session::invalid(
            "point flux candidate owner lacks a requested position. Discard the inconsistent candidate set",
        ));
    }
    Ok(positions)
}
