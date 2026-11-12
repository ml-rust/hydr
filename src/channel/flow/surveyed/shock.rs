use super::equation_ops::{choose, scalar};
use super::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn offset<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    original: &Tensor<R>,
    secant: &Interval<R>,
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
    let area = shock_area(a, target, original)?;
    offset_from_areas(a, &area, original, secant)
}

pub(crate) fn curve<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    original: &Tensor<R>,
    secant: &Interval<R>,
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
    let area = shock_area(a, target, original)?;
    let offset = offset_from_areas(a, &area, original, secant)?;
    // The contraction increases with area, so evaluate its represented endpoints independently.
    let lower = contraction_at_area(a, area.lower(), original)?;
    let upper = contraction_at_area(a, area.upper(), original)?;
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let contraction = a.bounds(
        &c.maximum(lower.lower(), zero.lower())?,
        &c.minimum(upper.upper(), one.upper())?,
    )?;
    Ok(a.mul(&contraction, &offset)?)
}

fn shock_area<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    original: &Tensor<R>,
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
    // Rarefaction-only rows retain a benign hypothetical shock at the original area.
    Ok(a.bounds(
        &c.maximum(target.lower(), original)?,
        &c.maximum(target.upper(), original)?,
    )?)
}

fn offset_from_areas<R>(
    a: &mut Arithmetic<'_, R>,
    area: &Interval<R>,
    original: &Tensor<R>,
    secant: &Interval<R>,
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
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let multiply_first = c.mul(
        &c.lt(original, one.lower())?,
        &c.ge(area.upper(), one.upper())?,
    )?;
    let divide_first = c.eq(&multiply_first, zero.lower())?;
    let original = a.point(original)?;
    let original_root = a.sqrt_nonnegative(&original)?;
    let area_root = a.sqrt_nonnegative(area)?;
    let secant_root = a.sqrt_nonnegative(secant)?;

    // Mask every unused operand before eager interval multiplication or division.
    let product_secant = choose(a, &multiply_first, &secant_root, &one)?;
    let product_area = choose(a, &multiply_first, &area_root, &one)?;
    let product_original = choose(a, &multiply_first, &original_root, &one)?;
    let product = a.mul(&product_secant, &product_area)?;
    let product_first = a.div_positive_interval(&product, &product_original)?;

    let quotient_secant = choose(a, &divide_first, &secant_root, &one)?;
    let quotient_area = choose(a, &divide_first, &area_root, &one)?;
    let quotient_original = choose(a, &divide_first, &original_root, &one)?;
    let quotient = a.div_positive_interval(&quotient_area, &quotient_original)?;
    let quotient_first = a.mul(&quotient_secant, &quotient)?;
    choose(a, &multiply_first, &product_first, &quotient_first)
}

fn contraction_at_area<R>(
    a: &mut Arithmetic<'_, R>,
    area: &Tensor<R>,
    original: &Tensor<R>,
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
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let close = c.ge(original, &c.mul_scalar(area, 0.5)?)?;
    let distant = c.eq(&close, zero.lower())?;
    let equal = c.eq(area, original)?;
    let area = a.point(area)?;
    let original = a.point(original)?;

    let close_area = choose(a, &close, &area, &one)?;
    let close_original = choose(a, &close, &original, &one)?;
    // Sterbenz subtraction is exact here, including gradual underflow and equal areas.
    let difference = a.point(&c.sub(close_area.lower(), close_original.lower())?)?;
    let close_ratio = a.div_positive_interval(&difference, &close_area)?;

    let distant_area = choose(a, &distant, &area, &one)?;
    let distant_original = choose(a, &distant, &original, &one)?;
    let quotient = a.div_positive_interval(&distant_original, &distant_area)?;
    let distant_ratio = a.sub(&one, &quotient)?;
    let ratio = choose(a, &close, &close_ratio, &distant_ratio)?;
    choose(a, &equal, &zero, &ratio)
}
