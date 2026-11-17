//! Outward rounding bound shared across numerics modules.

use numr::dtype::DType;
use numr::ops::{ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Reuse the rounding expression with a caller-specific shape and diagnostic context.
pub fn rounding_with_client<R>(
    client: &R::Client,
    value: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ScalarOps<R> + UnaryOps<R>,
{
    client.add_scalar(
        &client.mul_scalar(&client.abs(value)?, f64::EPSILON)?,
        f64::from_bits(1),
    )
}
