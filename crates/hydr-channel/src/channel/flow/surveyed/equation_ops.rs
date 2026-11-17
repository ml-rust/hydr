use super::PrismaticRiemannRejection;
use super::error::{Arithmetic, HydraulicError, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps,
    TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, CertifiedExpansion, Interval};

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

/// Select `yes` where `mask` is one and `no` elsewhere, component by component.
pub(crate) fn choose_expansion<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    yes: &CertifiedExpansion<R>,
    no: &CertifiedExpansion<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
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
    a.expansion_bounds(
        &c.where_cond(mask, yes.high(), no.high())?,
        &c.where_cond(mask, yes.low(), no.low())?,
        &c.where_cond(mask, yes.abs_error(), no.abs_error())?,
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::error::checked;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    #[test]
    fn choose_expansion_picks_each_side_per_component() -> HydraulicResult<()> {
        let client = CpuClient::new(CpuDevice::new());
        checked::<CpuRuntime, _>(&client, "select expansion test", |a| {
            let device = a.client().device().clone();
            let part = |values: [f64; 2]| Tensor::<CpuRuntime>::from_slice(&values, &[2], &device);
            let yes = a.expansion_bounds(
                &part([1.0, 2.0])?,
                &part([1e-20, 2e-20])?,
                &part([0.5, 0.25])?,
            )?;
            let no = a.expansion_bounds(
                &part([3.0, 4.0])?,
                &part([3e-20, 4e-20])?,
                &part([0.75, 0.125])?,
            )?;
            let mask = part([1.0, 0.0])?;
            let picked = choose_expansion(a, &mask, &yes, &no)?;
            assert_eq!(picked.high().to_vec::<f64>(), vec![1.0, 4.0]);
            assert_eq!(picked.low().to_vec::<f64>(), vec![1e-20, 4e-20]);
            assert_eq!(picked.abs_error().to_vec::<f64>(), vec![0.5, 0.125]);
            Ok(())
        })
    }
}
