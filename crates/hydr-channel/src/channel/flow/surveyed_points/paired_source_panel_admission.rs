use super::super::{paired_source_cache::SourcePotentialCache, paired_sources::PairedSources};
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn admit_rows<R>(
    sources: &PairedSources<'_, R>,
    rows: [&Tensor<R>; 2],
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
    let c = sources.geometry.client();
    let count = sources.original_rows.numel();
    if count as u128 > (1_u128 << f64::MANTISSA_DIGITS) {
        return Err(storage_error(count, sources.cache[0].ready.dtype(), 1));
    }
    for (side, value) in ["left", "right"].into_iter().zip(rows) {
        if value.shape() != [value.numel()]
            || value.dtype() != DType::I64
            || !value.device().is_same(c.device())
        {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "Source panel side={side}, shape={:?}, rank={}, dtype={:?}, device={}: supply rank-one I64 parent rows on device={}",
                    value.shape(),
                    value.shape().len(),
                    value.dtype(),
                    value.device().id(),
                    c.device().id(),
                ),
            });
        }
    }
    checked::<R, _>(c, "source panel parent admission", |_a| {
        let zero = Tensor::zeros(&[], DType::I64, c.device())?;
        let limit = Tensor::full_scalar(&[], DType::I64, count as f64, c.device())?;
        for (side, value) in ["left", "right"].into_iter().zip(rows) {
            if value.numel() != 0 {
                let outside = c.maximum(&c.lt(value, &zero)?, &c.ge(value, &limit)?)?;
                let outside = c.nonzero(&outside)?.flatten()?;
                if outside.numel() != 0 {
                    let first = outside.narrow(0, 0, 1)?;
                    let parent = c.index_select(value, 0, &first)?.item::<i64>()?;
                    return Err(HydraulicError::Numerical {
                        context: format!(
                            "Source panel side={side}, parent={parent}, captured_count={count}: supply parent rows in [0, {count})"
                        ),
                    });
                }
            }
        }
        Ok(())
    })
}

pub(super) fn storage<R: Runtime<DType = DType>>(
    count: usize,
    dtype: DType,
    levels: usize,
) -> HydraulicResult<()> {
    let error = || storage_error(count, dtype, levels);
    let elements = count
        .checked_mul(12)
        .and_then(|n| n.checked_mul(levels))
        .ok_or_else(error)?;
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .and_then(|bytes| {
            levels
                .checked_mul(std::mem::size_of::<(usize, [SourcePotentialCache<R>; 2])>())
                .and_then(|metadata| bytes.checked_add(metadata))
        })
        .ok_or_else(error)?;
    if bytes > isize::MAX as usize
        || count as u128 > (1_u128 << f64::MANTISSA_DIGITS)
        || count > i64::MAX as usize
    {
        return Err(error());
    }
    Ok(())
}

pub(super) fn storage_error(count: usize, dtype: DType, levels: usize) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!(
            "Source panel cache count={count}, dtype={dtype:?}, levels={levels}, numeric_fields_per_parent=12 exceeds storage/index bounds: bytes_limit={}, exact_row_limit={}, index_limit={}. Split the source graph or reduce panel levels",
            isize::MAX,
            1_u128 << f64::MANTISSA_DIGITS,
            i64::MAX,
        ),
    }
}
