use super::super::ChannelError;
use numr::dtype::DType;
use numr::ops::{CompareOps, ReduceOps, UnaryOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn check_scalar<R: Runtime<DType = DType>>(
    client: &R::Client,
    tensor: &Tensor<R>,
    name: &str,
) -> Result<(), ChannelError> {
    if tensor.shape() != [1]
        || tensor.dtype() != DType::F64
        || !tensor.device().is_same(client.device())
    {
        return Err(ChannelError::input(format!(
            "{name} shape={:?}, dtype={:?}: supply an actual F64 scalar tensor on the configured client",
            tensor.shape(),
            tensor.dtype()
        )));
    }
    Ok(())
}

pub(super) fn all<R: Runtime<DType = DType>>(
    client: &R::Client,
    mask: &Tensor<R>,
) -> Result<bool, ChannelError>
where
    R::Client: ReduceOps<R>,
{
    Ok(client.min(mask, &[0], false)?.item::<f64>()? == 1.0)
}

pub(super) fn finite<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: &Tensor<R>,
) -> Result<bool, ChannelError>
where
    R::Client: CompareOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    let maximum = Tensor::full_scalar(&[1], DType::F64, f64::MAX, client.device())?;
    all(client, &client.le(&client.abs(value)?, &maximum)?)
}
