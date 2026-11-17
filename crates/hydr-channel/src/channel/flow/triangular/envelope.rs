//! Four-envelope directed expansions for finite F64 primitive results.

use crate::numerics::rounding::rounding_with_client;
use numr::{
    dtype::DType,
    ops::{BinaryOps, ScalarOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

pub(in crate::channel::flow) fn upper<R>(
    client: &R::Client,
    value: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    client.add(
        value,
        &client.mul_scalar(&rounding_with_client::<R>(client, value)?, 4.0)?,
    )
}

pub(in crate::channel::flow) fn lower<R>(
    client: &R::Client,
    value: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    client.sub(
        value,
        &client.mul_scalar(&rounding_with_client::<R>(client, value)?, 4.0)?,
    )
}
