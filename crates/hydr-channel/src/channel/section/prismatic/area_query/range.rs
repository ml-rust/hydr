use super::chart::{AreaDirection, ExpandedBounds};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) fn offsets<R>(
    a: &mut Arithmetic<'_, R>,
    scale: &CertifiedExpansion<R>,
    parameter: &Interval<R>,
    direction: AreaDirection,
) -> HydraulicResult<ExpandedBounds<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let lower = a.expansion_point(parameter.lower())?;
    let upper = a.expansion_point(parameter.upper())?;
    let lower = a.expansion_mul(scale, &lower)?;
    let upper = a.expansion_mul(scale, &upper)?;
    match direction {
        AreaDirection::Add => Ok(ExpandedBounds { lower, upper }),
        AreaDirection::Subtract => {
            let zeros = Tensor::zeros(scale.high().shape(), DType::F64, a.client().device())?;
            let zero = a.expansion_point(&zeros)?;
            Ok(ExpandedBounds {
                lower: a.expansion_sub(&zero, &upper)?,
                upper: a.expansion_sub(&zero, &lower)?,
            })
        }
    }
}

/// Retain one endpoint's transform radius while projecting its anchor-relative certificate.
pub(super) fn add_endpoint<R>(
    a: &mut Arithmetic<'_, R>,
    base: &Interval<R>,
    offset: &CertifiedExpansion<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let low = a.expansion_point(base.lower())?;
    let high = a.expansion_point(base.upper())?;
    let low = a.expansion_add(&low, offset)?;
    let high = a.expansion_add(&high, offset)?;
    let low = a.expansion_interval(&low)?;
    let high = a.expansion_interval(&high)?;
    Ok(a.bounds(low.lower(), high.upper())?)
}
