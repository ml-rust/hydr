use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::Interval;

/// Bound physical band weights without forming an overflowing quotient.
pub(super) fn band_weights<R>(
    a: &mut Arithmetic<'_, R>,
    numerator: &Interval<R>,
    denominator: &Interval<R>,
    zero: &Interval<R>,
    one: &Interval<R>,
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
    let c = a.client();
    let saturated_lower = c.ge(numerator.lower(), denominator.upper())?;
    let saturated_upper = c.ge(numerator.upper(), denominator.lower())?;
    let lower_numerator = a.point(&c.where_cond(
        &saturated_lower,
        zero.lower(),
        &c.maximum(numerator.lower(), zero.lower())?,
    )?)?;
    let lower_denominator =
        a.point(&c.where_cond(&saturated_lower, one.lower(), denominator.upper())?)?;
    let upper_numerator = a.point(&c.where_cond(
        &saturated_upper,
        zero.lower(),
        &c.maximum(numerator.upper(), zero.lower())?,
    )?)?;
    let upper_denominator =
        a.point(&c.where_cond(&saturated_upper, one.lower(), denominator.lower())?)?;
    let lower = a.div_positive_interval(&lower_numerator, &lower_denominator)?;
    let upper = a.div_positive_interval(&upper_numerator, &upper_denominator)?;
    Ok(a.bounds(
        &c.where_cond(
            &saturated_lower,
            one.lower(),
            &c.minimum(&c.maximum(lower.lower(), zero.lower())?, one.upper())?,
        )?,
        &c.where_cond(
            &saturated_upper,
            one.upper(),
            &c.minimum(&c.maximum(upper.upper(), zero.lower())?, one.upper())?,
        )?,
    )?)
}
