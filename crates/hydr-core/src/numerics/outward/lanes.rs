use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::error::OutwardError;

pub(super) fn check_tensor_finite<R>(
    client: &R::Client,
    value: &Tensor<R>,
    quantity: &str,
) -> Result<(), OutwardError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let invalid = client.maximum(
        &client.cast(&client.isnan(value)?, value.dtype())?,
        &client.cast(&client.isinf(value)?, value.dtype())?,
    )?;
    reject(
        client,
        &invalid,
        &format!("{quantity} contains nonfinite values. Rescale source inputs"),
    )
}

pub(super) fn reject<R>(
    client: &R::Client,
    mask: &Tensor<R>,
    context: &str,
) -> Result<(), OutwardError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    if let Some(index) = rejected_lane(client, mask)? {
        return Err(OutwardError::numerical(context).at_cell(index));
    }
    Ok(())
}

pub(super) fn rejected_lane<R>(
    client: &R::Client,
    mask: &Tensor<R>,
) -> Result<Option<usize>, OutwardError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    if client.any(mask, &[], false)?.item::<f64>()? == 0.0 {
        return Ok(None);
    }
    let selected = client.argmax(mask, 0, false)?.item::<i64>()?;
    let index = usize::try_from(selected)
        .ok()
        .filter(|index| *index < mask.numel())
        .ok_or_else(|| {
            OutwardError::numerical("source rejection mask contradicts its reduction")
        })?;
    if mask.narrow(0, index, 1)?.item::<f64>()? == 0.0 {
        return Err(OutwardError::numerical(
            "source rejection mask contradicts its reduction",
        ));
    }
    Ok(Some(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuClient;
    use numr::runtime::cpu::{CpuDevice, ParallelismConfig};

    #[test]
    fn source_rejection_selects_first_tied_strided_lane() -> Result<(), OutwardError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        for count in [3, 33, 257] {
            let first = if count == 3 { 1 } else { 17 };
            let mut flags = vec![0.0; count];
            flags[first] = 1.0;
            flags[count - 1] = 1.0;
            let padded: Vec<_> = flags.iter().flat_map(|value| [99.0, *value]).collect();
            let mask = Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
                &padded,
                &[count, 2],
                &device,
            )?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
            assert_eq!(rejected_lane(&client, &mask)?, Some(first));
        }
        Ok(())
    }
}
