use super::ChannelPoolError;
use crate::numerics::water_transfers::{finite, reject, upward_add};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn zero<R: Runtime<DType = DType>>(
    client: &R::Client,
) -> Result<Tensor<R>, ChannelPoolError> {
    Ok(Tensor::zeros(&[1], DType::F64, client.device())?)
}

pub(super) fn nonnegative<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: &Tensor<R>,
    field: &str,
) -> Result<(), ChannelPoolError>
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
    finite(client, value, field).map_err(ChannelPoolError::from_transfer)?;
    reject(client, &client.lt(value, &zero(client)?)?, field)
        .map_err(ChannelPoolError::from_transfer)?;
    Ok(())
}

pub(super) fn add_bounds<R: Runtime<DType = DType>>(
    client: &R::Client,
    values: &[&Tensor<R>],
) -> Result<Tensor<R>, ChannelPoolError>
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
    let mut result = zero(client)?;
    for value in values {
        nonnegative(client, value, "nonnegative accounting bound")?;
        result = upward_add(client, &result, value).map_err(ChannelPoolError::from_transfer)?;
    }
    Ok(result)
}

/// Each signed term contributes its exact TwoSum low to the report's arithmetic bound.
pub(super) fn expression<R: Runtime<DType = DType>>(
    client: &R::Client,
    terms: &[(&Tensor<R>, bool)],
) -> Result<(Tensor<R>, Tensor<R>), ChannelPoolError>
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
    let mut value = zero(client)?;
    let mut bound = zero(client)?;
    for (term, positive) in terms {
        let operand = if *positive {
            (*term).clone()
        } else {
            client.neg(term)?
        };
        let (next, low) = client.two_sum(&value, &operand)?;
        finite(client, &next, "water accounting expression")
            .map_err(ChannelPoolError::from_transfer)?;
        finite(client, &low, "water accounting expression low")
            .map_err(ChannelPoolError::from_transfer)?;
        bound = upward_add(client, &bound, &client.abs(&low)?)
            .map_err(ChannelPoolError::from_transfer)?;
        value = next;
    }
    Ok((value, bound))
}

pub(super) fn admit<R: Runtime<DType = DType>>(
    client: &R::Client,
    residual: &Tensor<R>,
    bound: &Tensor<R>,
    scale: &Tensor<R>,
    absolute: f64,
    relative: f64,
) -> Result<(), ChannelPoolError>
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
    nonnegative(client, scale, "physical water scale")?;
    let budget = client.add_scalar(&client.mul_scalar(scale, relative)?, absolute)?;
    finite(client, &budget, "physical water budget").map_err(ChannelPoolError::from_transfer)?;
    let used = upward_add(client, &client.abs(residual)?, bound)
        .map_err(ChannelPoolError::from_transfer)?;
    reject(
        client,
        &client.gt(&used, &budget)?,
        "water residual and arithmetic bound exceed the physical budget. Refine the common interval or inspect represented storage",
    )
    .map_err(ChannelPoolError::from_transfer)?;
    Ok(())
}

pub(super) fn physical_scale<R: Runtime<DType = DType>>(
    client: &R::Client,
    before: &Tensor<R>,
    after: &Tensor<R>,
    external_in: &Tensor<R>,
    external_out: &Tensor<R>,
    lateral_in: &Tensor<R>,
    lateral_out: &Tensor<R>,
) -> Result<Tensor<R>, ChannelPoolError>
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
    let incoming = client.add(external_in, lateral_in)?;
    let outgoing = client.add(external_out, lateral_out)?;
    let scale = client.maximum(
        &client.maximum(before, after)?,
        &client.maximum(&incoming, &outgoing)?,
    )?;
    nonnegative(
        client,
        &scale,
        "physical storage and external throughput scale",
    )?;
    Ok(scale)
}
