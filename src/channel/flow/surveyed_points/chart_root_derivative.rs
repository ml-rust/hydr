use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::{GatheredBands, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Scale the physical wet-pair derivative before projecting its positive lower bound.
pub(super) fn lower<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    query: &SupportedAreaQuery<R>,
    gravity: f64,
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
    let scale = a.expansion_interval(&query.query.scale)?;
    let two = Tensor::full_scalar(&[], a.dtype(), 2.0, a.client().device())?;
    let gravity = Tensor::full_scalar(&[], a.dtype(), gravity, a.client().device())?;
    let gravity = a.point(&gravity)?;
    let gravity = a.sqrt_nonnegative(&gravity)?;
    let cap_area = a.sqrt_nonnegative(&bands.cap_area)?;
    let cap_width = a.sqrt_nonnegative(&bands.cap_width)?;
    Ok(a.positive_product_ratio_lower(
        &[&two, scale.lower(), gravity.lower()],
        &[cap_area.upper(), cap_width.upper()],
    )?)
}
