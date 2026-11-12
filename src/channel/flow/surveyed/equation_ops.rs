use super::PrismaticRiemannRejection;
use super::error::{Arithmetic, HydraulicError, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps,
    TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn scalar<R>(a: &mut Arithmetic<'_, R>, value: f64) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.point(&Tensor::from_slice(&[value], &[], a.client().device())?)?)
}

pub(crate) fn scale<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    factor: f64,
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
    let factor = scalar(a, factor)?;
    Ok(a.mul(value, &factor)?)
}

pub(crate) fn gravity_depth_root<R>(
    a: &mut Arithmetic<'_, R>,
    gravity: &Interval<R>,
    depth: &Interval<R>,
    bottom_factor: Option<&Interval<R>>,
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
    let invalid = c.maximum(
        &c.le(gravity.lower(), zero.upper())?,
        &c.lt(depth.lower(), zero.lower())?,
    )?;
    reject_mask(
        a,
        &invalid,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Surveyed gravity-depth root requires positive gravity and nonnegative depth. Check the hydraulic operands",
    )?;
    let mut product = a.mul(gravity, depth)?;
    if let Some(factor) = bottom_factor {
        reject_mask(
            a,
            &c.lt(factor.lower(), zero.lower())?,
            PrismaticRiemannRejection::QuadratureUnresolved,
            "Surveyed bottom factor requires a nonnegative value. Check the analytic bottom identity",
        )?;
        product = a.mul(&product, factor)?;
    }
    // Physical gravity and extent prove this product nonnegative, including an uncertain zero endpoint.
    let product = a.bounds(&c.maximum(product.lower(), zero.lower())?, product.upper())?;
    Ok(a.sqrt_nonnegative(&product)?)
}

pub(crate) fn select_interval<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    rows: &Tensor<R>,
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
    Ok(a.bounds(
        &c.index_select(value.lower(), 0, rows)?,
        &c.index_select(value.upper(), 0, rows)?,
    )?)
}

pub(crate) fn choose<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    yes: &Interval<R>,
    no: &Interval<R>,
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
    Ok(a.bounds(
        &c.where_cond(mask, yes.lower(), no.lower())?,
        &c.where_cond(mask, yes.upper(), no.upper())?,
    )?)
}

pub(crate) fn hull<R>(
    a: &mut Arithmetic<'_, R>,
    first: &Interval<R>,
    second: &Interval<R>,
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
    Ok(a.bounds(
        &c.minimum(first.lower(), second.lower())?,
        &c.maximum(first.upper(), second.upper())?,
    )?)
}

pub(crate) fn reject_mask<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    reason: PrismaticRiemannRejection,
    context: &str,
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
    if mask.numel() != 0
        && a.client()
            .max(&mask.flatten()?, &[0], false)?
            .item::<f64>()?
            != 0.0
    {
        return Err(HydraulicError::Rejected {
            reason,
            query: None,
            section_owner: None,
            context: context.into(),
        });
    }
    Ok(())
}
