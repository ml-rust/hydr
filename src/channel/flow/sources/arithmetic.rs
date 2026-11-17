use super::super::ChannelError;
use crate::numerics::outward::{AdjustmentReduction, add_adjustment_charge};
use crate::numerics::rounding::rounding_with_client;
use crate::numerics::water_transfers::{finite, upward_add};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn scalar_charge(current: f64, step: f64) -> Result<f64, ChannelError> {
    add_adjustment_charge(current, step).map_err(|e| ChannelError::numerical(e.to_string()))
}

pub(super) fn zeros<R: Runtime<DType = DType>>(
    client: &R::Client,
    count: usize,
) -> Result<Tensor<R>, ChannelError> {
    Ok(Tensor::zeros(&[count], DType::F64, client.device())?)
}

pub(super) fn upper_sum<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: &Tensor<R>,
    reduction: AdjustmentReduction,
) -> Result<f64, ChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    if value.numel() == 0 {
        return Ok(0.0);
    }
    let reported = client.sum(value, &[0], false)?.item::<f64>()?;
    let error = reduction
        .unit_sum_error_bound(reported)
        .map_err(|e| ChannelError::numerical(e.to_string()))?;
    scalar_charge(reported, error)
}

pub(super) fn reported_sum<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: &Tensor<R>,
    reduction: AdjustmentReduction,
) -> Result<(f64, f64), ChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    if value.numel() == 0 {
        return Ok((0.0, 0.0));
    }
    let reported = client.sum(value, &[0], false)?.item::<f64>()?;
    let error = reduction
        .unit_sum_error_bound(reported)
        .map_err(|e| ChannelError::numerical(e.to_string()))?;
    Ok((reported, error))
}

pub(super) fn product_roundoff<R: Runtime<DType = DType>>(
    client: &R::Client,
    a: &Tensor<R>,
    b: &Tensor<R>,
    product: &Tensor<R>,
) -> Result<Tensor<R>, ChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    finite(client, product, "lateral momentum product")?;
    let zero = zeros(client, 1)?;
    let active = client.mul(
        &client.gt(&client.abs(a)?, &zero)?,
        &client.gt(&client.abs(b)?, &zero)?,
    )?;
    Ok(client.where_cond(&active, &rounding_with_client::<R>(client, product)?, &zero)?)
}

pub(super) fn propagated_bound<R: Runtime<DType = DType>>(
    client: &R::Client,
    bound: &Tensor<R>,
    factor: &Tensor<R>,
) -> Result<Tensor<R>, ChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    let zero = zeros(client, 1)?;
    let factor = client.abs(factor)?;
    let product = client.mul(bound, &factor)?;
    let raised = client.add(&product, &rounding_with_client::<R>(client, &product)?)?;
    let active = client.mul(&client.gt(bound, &zero)?, &client.gt(&factor, &zero)?)?;
    let result = client.where_cond(&active, &raised, &zero)?;
    finite(client, &result, "lateral propagated momentum bound")?;
    Ok(result)
}

pub(super) fn add_bounds<R: Runtime<DType = DType>>(
    client: &R::Client,
    bounds: &[&Tensor<R>],
) -> Result<Tensor<R>, ChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    let first = bounds.first().ok_or_else(|| {
        ChannelError::input("lateral bound list is empty: supply a physical arithmetic bound")
    })?;
    let mut sum = Tensor::zeros(first.shape(), DType::F64, client.device())?;
    for bound in bounds {
        sum = upward_add(client, &sum, bound)?;
    }
    Ok(sum)
}
