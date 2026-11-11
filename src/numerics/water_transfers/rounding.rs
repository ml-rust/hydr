use super::WaterTransferError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) fn finite<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: &Tensor<R>,
    field: &str,
) -> Result<(), WaterTransferError>
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
        return Ok(());
    }
    let maximum = Tensor::<R>::full_scalar(&[], DType::F64, f64::MAX, client.device())?;
    let admitted = client.le(&client.abs(value)?, &maximum)?;
    if client.min(&admitted, &[0], false)?.item::<f64>()? != 1.0 {
        return Err(WaterTransferError::numerical(format!(
            "{field} contains nonfinite values. Inspect the physical scale and arithmetic envelope"
        )));
    }
    Ok(())
}

pub(crate) fn reject<R: Runtime<DType = DType>>(
    client: &R::Client,
    mask: &Tensor<R>,
    message: &str,
) -> Result<(), WaterTransferError>
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
    if mask.numel() == 0 {
        return Ok(());
    }
    if client.max(mask, &[0], false)?.item::<f64>()? != 0.0 {
        return Err(WaterTransferError::numerical(message));
    }
    Ok(())
}

pub(crate) fn upward_add<R: Runtime<DType = DType>>(
    client: &R::Client,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, WaterTransferError>
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
    let (sum, low) = client.two_sum(a, b)?;
    let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
    let rounding = client.add_scalar(
        &client.mul_scalar(&client.abs(&sum)?, f64::EPSILON)?,
        f64::from_bits(1),
    )?;
    let raised = client.add(&sum, &rounding)?;
    let result = client.where_cond(&client.gt(&low, &zero)?, &raised, &sum)?;
    finite(client, &result, "transfer assembly error bound")?;
    Ok(result)
}
